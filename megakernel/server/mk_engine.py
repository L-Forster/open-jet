"""The engine's C API through ctypes."""
import contextlib
import ctypes
import json
import threading

from mk_errors import PrefillCancelled


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, mtp=None):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_set_penalty": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_prefill_ck": (ctypes.c_int, [vp, ip, ctypes.c_int, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
        }
        opt = {  # checkpoint pool (libraries before it: one checkpoint)
            "mk_prefill_cks": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int]),
            "mk_checkpoints": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_rewind_to": (ctypes.c_int, [vp, ctypes.c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.pool = all(hasattr(L, name) for name in opt)
        if self.pool:
            for name, (res, args) in opt.items():
                f = getattr(L, name)
                f.restype, f.argtypes = res, args
        # images (libraries before it: text only)
        self.images = hasattr(L, "mk_prefill_img")
        if self.images:
            L.mk_prefill_img.restype = ctypes.c_int
            L.mk_prefill_img.argtypes = [vp, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ip, ip, ip,
                                         ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            L.mk_n_embd.restype, L.mk_n_embd.argtypes = ctypes.c_int, []
        dv = draft_vocab.encode() if draft_vocab else None
        if mtp is not None:  # --mtp FILE / --no-mtp ("none"); otherwise the model file's MTP layer (or MK_MTP / MK_NO_MTP)
            if not hasattr(L, "mk_new_mtp"):
                raise RuntimeError("--mtp / --no-mtp need a newer libqwen38mk (mk_new_mtp)")
            L.mk_new_mtp.restype = vp
            L.mk_new_mtp.argtypes = [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, cp]
            self.h = L.mk_new_mtp(model.encode(), ctx, n_draft, dv, kv_mode, mtp.encode())
        else:
            self.h = L.mk_new(model.encode(), ctx, n_draft, dv, kv_mode)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.has_mtp = True  # False: no MTP layer, plain decoding at depth 0
        if hasattr(L, "mk_has_mtp"):
            L.mk_has_mtp.restype, L.mk_has_mtp.argtypes = ctypes.c_int, [vp]
            self.has_mtp = L.mk_has_mtp(self.h) == 1
        self.ctx = ctx
        self._pieces = {}
        # MK_EXACT engines (exact reuse): prefill rows do not depend on the chunking; see Server.load_prompt
        self.exact = False
        if hasattr(L, "mk_exact"):
            L.mk_exact.restype, L.mk_exact.argtypes = ctypes.c_int, [vp]
            self.exact = L.mk_exact(self.h) == 1
        # keep-warm (MK_KEEP_WARM_S; libraries before it: none)
        self._warm = hasattr(L, "mk_keep_warm")
        if self._warm:
            L.mk_keep_warm.restype, L.mk_keep_warm.argtypes = ctypes.c_double, [vp, ctypes.c_int]
        # prefill cancel (libraries before it: a disconnect during a prefill is noticed when it ends)
        self._cancel = hasattr(L, "mk_cancel")
        if self._cancel:
            L.mk_cancel.restype, L.mk_cancel.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        # prefill progress for the console (libraries before it: elapsed time only)
        self._progress = hasattr(L, "mk_prefill_progress")
        if self._progress:
            L.mk_prefill_progress.restype, L.mk_prefill_progress.argtypes = None, [vp, ip, ip]
        # decode statistics; the library's build (commit, engine flags; libraries before it: unknown)
        self._stats = hasattr(L, "mk_stats")
        if self._stats:
            lp = ctypes.POINTER(ctypes.c_long)
            L.mk_stats.restype, L.mk_stats.argtypes = None, [vp, lp, lp, lp]
        # conversation state in host memory: side requests and parked conversations (libraries before it: neither)
        self.store = hasattr(L, "mk_park")
        if self.store:
            ll, llp, ci = ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int
            for name, res, args in (("mk_store_budget", ci, [vp, ll]), ("mk_store_reserve", ci, [vp, ll]),
                                    ("mk_store_info", ci, [vp, llp, ci]), ("mk_kv_sizes", None, [vp, llp, llp, llp, llp]),
                                    ("mk_side_begin", ci, [vp]), ("mk_side_end", ci, [vp]), ("mk_side_drop", ci, [vp]),
                                    ("mk_side_active", ci, [vp]), ("mk_park", ci, [vp, ci]), ("mk_unpark", ci, [vp, ci]),
                                    ("mk_park_drop", ci, [vp, ci]), ("mk_park_need", ll, [vp, ci])):
                f = getattr(L, name)
                f.restype, f.argtypes = res, args
        # a side request's own conversation parked after it, and continued inside a later one (libraries before: neither)
        self.side_keep = self.store and hasattr(L, "mk_side_end_park") and hasattr(L, "mk_side_unpark")
        if self.side_keep:
            L.mk_side_end_park.restype, L.mk_side_end_park.argtypes = ctypes.c_int, [vp, ctypes.c_int, ip]
            L.mk_side_unpark.restype, L.mk_side_unpark.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        # pinned checkpoints: a prompt's shared prefix (a client's system prompt and tools) survives the pool's thinning
        # and parking (libraries before them: nothing is pinned, parked conversations keep their newest checkpoints only)
        self.ck_pins = hasattr(L, "mk_ck_pin") and hasattr(L, "mk_ck_pinned")
        if self.ck_pins:
            L.mk_ck_pin.restype, L.mk_ck_pin.argtypes = ctypes.c_int, [vp, ip, ctypes.c_int]
            L.mk_ck_pinned.restype, L.mk_ck_pinned.argtypes = ctypes.c_int, [vp, ip, ctypes.c_int]
        # parallel slots (--agents: SlotScheduler; open-jet's slot API, MKEngine::slots_open; libraries before them: none)
        self.slots = all(hasattr(L, n) for n in ("mk_slots_open2", "mk_slot_begin2", "mk_slot_prefill", "mk_slots_step"))
        if self.slots:
            ci, cf = ctypes.c_int, ctypes.c_float
            for name, args in (("mk_slots_open2", [vp, ci, ci, ci]),
                               ("mk_slot_begin2", [vp, ci, ip, ci, cf, cf, ci, cf, ctypes.c_ulonglong, ci, ci]),
                               ("mk_slot_prefill", [vp, ci, ci]), ("mk_slots_step", [vp, ip, ci, ip, ci, ip])):
                f = getattr(L, name)
                f.restype, f.argtypes = ci, args
        self.n_vocab = 0  # (libraries before mk_n_vocab: unknown)
        if hasattr(L, "mk_n_vocab"):
            L.mk_n_vocab.restype, L.mk_n_vocab.argtypes = ctypes.c_int, [vp]
            self.n_vocab = L.mk_n_vocab(self.h)
        # tokenizer flags and token attributes (libraries before them: special tokens always parsed, no BOS / EOS)
        self.tok_flags = hasattr(L, "mk_tokenize2") and hasattr(L, "mk_token_attr") and self.n_vocab > 0
        if self.tok_flags:
            L.mk_tokenize2.restype = ctypes.c_int
            L.mk_tokenize2.argtypes = [vp, cp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            L.mk_token_attr.restype, L.mk_token_attr.argtypes = ctypes.c_int, [vp, ctypes.c_int]
        self._controls = None
        self.build = None
        if hasattr(L, "mk_build_info"):
            L.mk_build_info.restype, L.mk_build_info.argtypes = ctypes.c_char_p, []
            raw = L.mk_build_info().decode(errors="replace")
            try:
                self.build = json.loads(raw)
            except ValueError:
                self.build = {"raw": raw}

    def stats(self):
        """(decode cycles, drafted tokens, accepted drafts) since the load, or None (a library without mk_stats)"""
        if not self._stats:
            return None
        c, d, a = ctypes.c_long(), ctypes.c_long(), ctypes.c_long()
        self.L.mk_stats(self.h, ctypes.byref(c), ctypes.byref(d), ctypes.byref(a))
        return c.value, d.value, a.value

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text, special=True, add_special=False):
        """text -> token ids. special: the text of control tokens (<|im_start|>, ...) becomes those tokens, else plain text;
        add_special: the vocabulary's BOS / EOS where it adds them (llama_tokenize's parse_special / add_special;
        other than the defaults they need mk_tokenize2: ValueError without it)"""
        b = text.encode()
        if special and not add_special:
            def run(out, cap):
                return self.L.mk_tokenize(self.h, b, len(b), out, cap)
        elif self.tok_flags:
            def run(out, cap):
                return self.L.mk_tokenize2(self.h, b, len(b), out, cap, int(add_special), int(special))
        else:
            raise ValueError("parse_special false / add_special need a newer libqwen38mk (mk_tokenize2)")
        cap = min(len(b) + 16, 1 << 20)  # (a longer text is tokenized twice: the first call returns its count)
        out = (ctypes.c_int * cap)()
        n = run(out, cap)
        if -2 ** 31 < n < 0:  # (-count: a longer buffer; INT_MIN: an error)
            cap = -n
            out = (ctypes.c_int * cap)()
            n = run(out, cap)
        if n < 0:
            raise self._err("tokenize")
        return list(out[:n])

    def controls(self):
        """{text: id} of the vocabulary's control (and unknown) tokens: the ones whose text tokenize(special=False) keeps
        as text (user-defined ones such as <think> are tokens either way); {} without mk_token_attr. Read once (one call
        per token id, ~0.1 s)."""
        if self._controls is None:
            self._controls = {}
            if self.tok_flags:
                attr = self.L.mk_token_attr
                for t in range(self.n_vocab):
                    a = attr(self.h, t)
                    if a > 0 and a & 0x9:  # (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN)
                        text = self.piece(t).decode(errors="replace")
                        if text:
                            self._controls.setdefault(text, t)
        return self._controls

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            if -2 ** 31 < n < 0:  # (-size: a longer piece, read again into its size, as tokenize does; INT_MIN: an error)
                buf = ctypes.create_string_buffer(-n)
                n = self.L.mk_piece(self.h, tok, buf, -n)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def keep_warm(self, on=True):
        """on: keep the GPU's clocks up for MK_KEEP_WARM_S seconds from now (the engine stops the load before its next
        GPU work); off: stop now. Returns the window in seconds (0: off)."""
        return self.L.mk_keep_warm(self.h, 1 if on else 0) if self._warm else 0.0

    @contextlib.contextmanager
    def cancel_when(self, gone, every=0.05):
        """runs the block (a prefill call) next to a watcher thread that polls gone() every `every` seconds: once it
        returns True, the engine stops the prefill at its next chunk boundary (mk_cancel: a host flag) and the call
        raises PrefillCancelled, with the engine back at its last checkpoint before the prefill"""
        if not self._cancel or gone is None:
            yield
            return
        self.L.mk_cancel(self.h, 0)
        done = threading.Event()

        def watch():
            while not done.wait(every):
                if gone():
                    self.L.mk_cancel(self.h, 1)
                    return
        th = threading.Thread(target=watch, name="mk-cancel-watch", daemon=True)
        th.start()
        try:
            yield
        finally:
            done.set()
            th.join()
            self.L.mk_cancel(self.h, 0)

    def prefill_progress(self):
        """(done, total) positions of the running prefill's batched part, callable from any thread while it runs; None
        for a library without mk_prefill_progress"""
        if not self._progress:
            return None
        d, t = ctypes.c_int(), ctypes.c_int()
        self.L.mk_prefill_progress(self.h, ctypes.byref(d), ctypes.byref(t))
        return d.value, t.value

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def set_penalty(self, presence, frequency):
        if self.L.mk_set_penalty(self.h, presence, frequency) < 0:
            raise self._err("penalty")

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        """cks: more checkpoint positions (absolute; the engine moves them onto its scan grid), ck_at stays exact.
        images: (pos, nx, ny, rows address) of each image the fed positions overlap (absolute positions)"""
        arr = (ctypes.c_int * len(toks))(*toks)
        if images:
            k = len(images)
            ca = (ctypes.c_int * max(len(cks), 1))(*cks)
            pos, nx, ny = ((ctypes.c_int * k)(*[im[j] for im in images]) for j in range(3))
            rows = (ctypes.c_void_p * k)(*[im[3] for im in images])
            r = self.L.mk_prefill_img(self.h, arr, len(toks), ca, len(cks), ck_at, pos, nx, ny, rows, k)
        elif self.pool and cks:
            ca = (ctypes.c_int * len(cks))(*cks)
            r = self.L.mk_prefill_cks(self.h, arr, len(toks), ca, len(cks), ck_at)
        else:
            r = self.L.mk_prefill_ck(self.h, arr, len(toks), ck_at)
        if r == -2:  # (mk_cancel)
            raise PrefillCancelled(self.L.mk_last_error().decode())
        if r < 0:
            raise self._err("prefill")
        return r

    def step(self):
        out = (ctypes.c_int * 16)()
        n = self.L.mk_step(self.h, out, 16)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0

    def checkpoints(self):
        if not self.pool:
            ck = self.checkpoint_pos()
            return [ck] if ck >= 0 else []
        n = self.L.mk_checkpoints(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_checkpoints(self.h, out, n)
        return list(out[:n])

    def rewind_to(self, pos):
        if not self.pool:
            return pos == self.checkpoint_pos() and self.rewind_to_checkpoint()
        return self.L.mk_rewind_to(self.h, pos) == 0

    def ck_pin(self, positions):
        """pins, for each position P, the checkpoint at the last position in (P - 32, P]: one there now, else the one the
        next prefill saves there (MKEngine::ck_pin). Pinned checkpoints are never evicted from the pool and are parked
        with the conversation (besides its newest --park-cks)"""
        arr = (ctypes.c_int * max(len(positions), 1))(*positions)
        if self.L.mk_ck_pin(self.h, arr, len(positions)) < 0:
            raise self._err("checkpoint pin")

    def pinned(self):
        """positions of the pinned checkpoints (ascending)"""
        n = self.L.mk_ck_pinned(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_ck_pinned(self.h, out, n)
        return list(out[:n])

    # ---- host store: side requests, parked conversations (MKEngine::store_budget, side_begin, park) ----
    def store_budget(self, nbytes):
        if self.L.mk_store_budget(self.h, nbytes) < 0:
            raise self._err("host store")

    def store_info(self):
        """{budget, allocated, used, parked, last_bytes, last_ms, side_ms, kv_token_bytes, side_park_bytes, side_park_ms}"""
        out = (ctypes.c_longlong * 10)()
        n = self.L.mk_store_info(self.h, out, 10)
        keys = ("budget", "allocated", "used", "parked", "last_bytes", "last_us", "side_us", "kv_token_bytes",
                "side_park_bytes", "side_park_us")
        d = dict(zip(keys, out[:n]))
        d["last_ms"], d["side_ms"] = d.pop("last_us", 0) / 1e3, d.pop("side_us", 0) / 1e3
        d["side_park_ms"] = d.pop("side_park_us", 0) / 1e3
        return d

    def kv_sizes(self):
        """bytes: (per KV token, the window rings, the recurrent state, one checkpoint)"""
        v = [ctypes.c_longlong() for _ in range(4)]
        self.L.mk_kv_sizes(self.h, *[ctypes.byref(x) for x in v])
        return tuple(x.value for x in v)

    def side_begin(self):
        """True: the sequence is kept aside (the engine looks reset); False: no room in the host store"""
        r = self.L.mk_side_begin(self.h)
        if r < 0:
            raise self._err("side request")
        return r == 0

    def side_end(self):
        """True: the kept sequence is back bit for bit; False: it was dropped (the store ran full: the engine is reset)"""
        r = self.L.mk_side_end(self.h)
        if r < 0:
            raise self._err("side request")
        return r == 1

    def side_end_park(self, max_cks):
        """side_end, with the side request's own conversation parked first: (kept back, its entry id or None). Kept back
        False: the kept sequence was lost (the store ran full) and the side request's conversation stays live"""
        pid = ctypes.c_int(0)
        r = self.L.mk_side_end_park(self.h, max_cks, ctypes.byref(pid))
        if r < 0:
            raise self._err("side request")
        return r == 1, pid.value or None

    def side_unpark(self, pid):
        """right after side_begin: the side request continues parked conversation pid (its entry is freed); False: no
        room in the host store for what that moves aside (nothing changed)"""
        r = self.L.mk_side_unpark(self.h, pid)
        if r < 0:
            raise self._err("side request")
        return r == 0

    def side_active(self):
        return self.L.mk_side_active(self.h) == 1

    def side_drop(self):
        self.L.mk_side_drop(self.h)

    def park(self, max_cks):
        """a snapshot of the sequence in the host store: its id, or None (no room)"""
        r = self.L.mk_park(self.h, max_cks)
        if r == -3:
            return None
        if r < 0:
            raise self._err("park")
        return r or None

    def unpark(self, pid):
        if self.L.mk_unpark(self.h, pid) < 0:
            raise self._err("restore")

    def park_drop(self, pid):
        self.L.mk_park_drop(self.h, pid)

    def park_need(self, max_cks):
        return self.L.mk_park_need(self.h, max_cks)

    # ---- parallel slots (--agents: SlotScheduler drives them from its thread) ----
    SLOT_PER = 16  # tokens a slot can emit in one cycle that the output arrays hold (1 + its drafts, at most MAXD)

    def slots_open(self, n, ctx):
        """opens n slots of ctx positions each (0: the most that fit, leaving MK_SLOT_RESERVE_MIB of the device memory
        free); returns a slot's positions"""
        r = self.L.mk_slots_open2(self.h, n, ctx, 0)
        if r < 0:
            raise self._err("parallel slots")
        return r

    def slot_begin(self, slot, toks, temp, top_p, top_k, min_p, seed):
        """slot <- the prompt and its sampling, reusing what the slot's sequence has (truncated at the divergence, else
        back to its newest checkpoint before it, else from scratch); returns the reused tokens: slot_prefill feeds the
        rest"""
        arr = (ctypes.c_int * len(toks))(*toks)
        r = self.L.mk_slot_begin2(self.h, slot, arr, len(toks), temp, top_p, top_k, min_p, seed, 0, 0)
        if r < 0:
            raise self._err("slot prompt")
        return r

    def slot_prefill(self, slot, budget):
        """feeds up to `budget` of the slot's prompt tokens; returns how many are left (0: the slot decodes from the next
        cycle on, which emits its first token). Budget 0: the rest is abandoned (the slot keeps what was fed)"""
        r = self.L.mk_slot_prefill(self.h, slot, budget)
        if r < 0:
            raise self._err("slot prefill")
        return r

    def _slot_tokens(self, n, out, cnt):
        return [list(out[i * self.SLOT_PER:i * self.SLOT_PER + cnt[i]]) for i in range(n)]

    def slots_step(self, slots):
        """one engine cycle of the slots (one batched launch; a lone slot: the solo megakernel): the tokens each one
        emitted ([]: its context is full)"""
        n = len(slots)
        sa, out, cnt = (ctypes.c_int * n)(*slots), (ctypes.c_int * (n * self.SLOT_PER))(), (ctypes.c_int * n)()
        if self.L.mk_slots_step(self.h, sa, n, out, self.SLOT_PER, cnt) < 0:
            raise self._err("decode")
        return self._slot_tokens(n, out, cnt)
