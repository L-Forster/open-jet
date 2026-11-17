"""Server's prompt path: rendering, tokenizing, checkpoints, prefill and its progress."""
import contextlib
import itertools
import threading
import time

import jinja2

import mk_limits
from mk_chat import MEDIA_MARKER, TEMPLATE_KWARGS, normalize_messages, soft_switch
from mk_errors import BadRequest, ContextOverflow, PrefillCancelled
from mk_log import CON, secs
from mk_media import VideoBytes

PIN_MIN = 1024  # a shared prefix shorter than this is not pinned (Server.ck_pins: it is prefilled again in under ~0.5 s)


class PromptMixin:
    def prompt_tokens(self, body):
        """the token count of the request's prompt (rendered with the chat template, images as their tokens), without
        generating: /v1/messages/count_tokens, /v1/chat/completions/input_tokens"""
        text, _, media = self.render(body)
        if not media:
            return len(self.tokenize_prompt(text, media)[0])
        with self.lock:  # (the image cache belongs to the running request too)
            toks, images = self.tokenize_prompt(text, media)
            self.vision.release([im for _, im in images])
        return len(toks)

    def render(self, body):
        """(prompt text, thinking, image file bytes in prompt order: one MEDIA_MARKER in the text each)"""
        kw = body.get("chat_template_kwargs") or {}
        if not isinstance(kw, dict):
            raise BadRequest("chat_template_kwargs must be an object")
        kw = dict(kw)
        eff = body.get("reasoning_effort")  # (OpenAI's top-level field; chat_template_kwargs win)
        if isinstance(eff, str) and eff:
            if eff == "none":  # (as llama-server)
                kw.setdefault("enable_thinking", False)
            else:  # (the template knows low / medium / high / xhigh)
                if eff in ("max", "ultra"):  # (Codex offers them for its own models: the template's highest instead)
                    self.warn_once(f"effort-{eff}", f"reasoning effort {eff!r} is taken as 'xhigh' (the chat "
                                                    f"template's highest)")
                    eff = "xhigh"
                kw.setdefault("reasoning_effort", {"minimal": "low"}.get(eff, eff))
        if "enable_thinking" not in kw:  # (Qwen3's soft switches, which Qwen3.8's template ignores: the request's own)
            soft = soft_switch(body.get("messages") or [])
            if soft is not None:
                self.warn_once("soft-switch", "a request's trailing /no_think or /think (last user message, else the "
                                              "system message) or an empty <think></think> after the last user text "
                                              "turns its thinking off / on (chat_template_kwargs and reasoning_effort "
                                              "none win)")
                kw["enable_thinking"] = soft
        media, notes = [] if self.vision else None, []
        msgs = normalize_messages(body.get("messages") or [], media, notes)
        if "<|channel|>" not in (getattr(self, "template_src", None) or ""):  # (llama-server's mapping: a template
            for m in msgs:  # without GPT-OSS's channels has no developer role, so OpenAI's newer name for system is one)
                if m.get("role") == "developer":
                    m["role"] = "system"
        ctx = dict(messages=msgs, add_generation_prompt=True)
        for key, msg in notes:
            self.warn_once(key, msg)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k, v in (getattr(self.args, "chat_template_kwargs", None) or {}).items():  # (--chat-template-kwargs: under them)
            kw.setdefault(k, v)
        for k in TEMPLATE_KWARGS:
            if k in kw:
                ctx[k] = kw[k]
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        except Exception as e:  # (the template's own expressions failing on the request's data: a 400, as llama-server)
            raise BadRequest(f"chat template: {type(e).__name__}: {e}")
        thinking = text.endswith("<think>\n")
        n_img = sum(1 for d in media or [] if not isinstance(d, VideoBytes))
        if n_img > mk_limits.MAX_IMAGES:
            raise BadRequest(f"{n_img} images in one request: at most {mk_limits.MAX_IMAGES} (MK_MAX_IMAGES)")
        if n_img < len(media or []) and not self.video_ok:
            raise BadRequest("video input is not available: " + (
                "the server found no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if self.vision is not None and self.vision.frames_ok else "the vision library has no video frames"))
        if media and text.count(MEDIA_MARKER) != len(media):
            raise BadRequest(f"the rendered prompt has {text.count(MEDIA_MARKER)} media markers for {len(media)} images / videos")
        return text, thinking, media or []

    def tokenize_prompt(self, text, media):
        """the prompt's token ids and its images [(pos, Vision.Image)]: each MEDIA_MARKER becomes <|vision_start|>, the
        image's rows (<|image_pad|> placeholders, nx * ny of them), <|vision_end|>; the text between is tokenized piece
        by piece, as mtmd_tokenize does. A video's marker becomes its parts (Vision.video: texts, and frames / frame
        pairs each like an image). The images are pinned (Vision.image / video): the caller releases them. Special-token
        text from the request's content (a Prompt of LiteralTemplate) is tokenized as text (tokenize_text)."""
        lit = getattr(text, "lit", None)
        if not media:
            return self.tokenize_text(lit[0] if lit else text, lit), []
        toks, images = [], []
        segs = (lit[0] if lit else text).split(MEDIA_MARKER)  # (the same media markers: LiteralTemplate checked it)
        frames_left = mk_limits.MAX_VIDEO_FRAMES

        def add_image(im, j=None):  # (j: its slot in images, else appended)
            toks.append(self.vis_start)
            if j is None:
                images.append((len(toks), im))
            else:
                images[j] = (len(toks), im)
            toks.extend([self.img_pad] * im.n)
            toks.append(self.vis_end)
        try:
            for i, seg in enumerate(segs):
                if seg:
                    toks += self.tokenize_text(seg, lit)
                if i < len(media):
                    if isinstance(media[i], VideoBytes):
                        # llama.cpp's layout: "Video:", frames / frame pairs as images, timestamp texts between (each
                        # text tokenized on its own, as mtmd_tokenize does)
                        v = self.vision.video(media[i], self.decoder, self.args.video_fps, self.args.video_max_tokens,
                                              frames_left)
                        j = len(images)
                        images.extend((None, im) for im in v.images)  # (all pinned now: released below on an error)
                        frames_left -= v.n_frames
                        for part in v.parts:
                            if isinstance(part, str):
                                toks += self.eng.tokenize(part)
                            else:
                                add_image(part, j)
                                j += 1
                    else:
                        add_image(self.vision.image(media[i]))
                    if len(toks) + ContextOverflow.RESERVE >= self.args.ctx:  # (no more images decoded for it)
                        raise ContextOverflow(len(toks), self.args.ctx, at_least=True)
        except BaseException:
            self.vision.release([im for _, im in images])
            raise
        return toks, images

    def tokenize_text(self, text, lit=None):
        """the token ids of rendered prompt text (a piece between media). With lit (Prompt.lit) the text comes from the
        prompt rendered with the request content's special-token texts as placeholders: it is split at the template's
        own control tokens (kept), and each piece between them is tokenized with its texts put back, as plain text where
        it has any (parse_special false: the piece is one run of text, as llama.cpp tokenizes the text between special
        tokens, so those texts never become control tokens). Pieces without any, and every prompt without such text
        in its content, are tokenized with control-token text parsed as special tokens (Engine.tokenize's default)."""
        if not lit:
            return self.eng.tokenize(text)
        _, back, lt = lit
        toks, pos = [], 0
        for m in itertools.chain(lt.ctl_re.finditer(text), [None]):
            piece = text[pos:m.start() if m else len(text)]
            if piece:
                orig = piece.translate(back)
                toks += self.eng.tokenize(orig, special=False) if orig != piece else self.eng.tokenize(piece)
            if m:
                toks.append(lt.controls[m.group()])
                pos = m.end()
        return toks

    def ck_positions(self, toks, start):
        """checkpoints for a prefill of toks[start:] besides the generation prompt: each message start and every
        --ck-every tokens (absolute positions; the engine keeps the most useful ones in its pool)"""
        cks = set()
        every = self.args.ck_every
        if every > 0:
            cks.update(range((start // every + 1) * every, len(toks) - 1, every))
        if self.args.ck_msgs and self.im_start is not None:
            cks.update(k for k in range(start + 1, len(toks) - 1) if toks[k] == self.im_start)
        return sorted(cks)

    def ck_pins(self, toks, lcp, spans):
        """checkpoints the engine pins for this prompt (--ck-pin; Engine.ck_pin: never evicted from the pool, parked with
        the conversation): the end of its system block (the start of the first message after it), which every session
        of the same client shares, tools included; and, when the prompt leaves the cached conversation (lcp) or a parked
        one inside that block, the point where it does: a client whose system prompt ends with per-session text (a date,
        a directory, a session id) shares up to there with its other sessions. That one comes first (it serves every
        session of the client). A prompt without a system message, or a prefix under PIN_MIN tokens: none."""
        role = getattr(self, "sys_role", None)
        if (not getattr(self.args, "ck_pin", 0) or not getattr(self.eng, "ck_pins", False) or self.im_start is None
                or not role or toks[:1 + len(role)] != [self.im_start] + role):
            return []
        end = next((k for k in range(1 + len(role), len(toks) - 1) if toks[k] == self.im_start), 0)
        if end < PIN_MIN:
            return []
        left = [lcp]
        if lcp < end:
            for e in (self.parked or {}).values():
                if not e.side:
                    r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                    left.append(r[0] if r[1] is None else r[1])
        cut = max((x for x in left if PIN_MIN <= x < end), default=0)
        return [cut, end] if cut else [end]

    @contextlib.contextmanager
    def prefill_status(self, n_new):
        """the console's status line during a prefill call: the engine's progress (mk_prefill_progress), its rate and the
        time left, every 0.25 s from a watcher thread (nothing for a prefill shorter than that, or off a terminal)"""
        if not CON.tty():
            yield
            return
        stop, t0 = threading.Event(), time.monotonic()
        prog = getattr(self.eng, "prefill_progress", None)

        def show():
            while not stop.wait(0.25):
                CON.status(self.prefill_text(n_new, prog() if prog else None, time.monotonic() - t0))
        th = threading.Thread(target=show, name="mk-status", daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()

    @contextlib.contextmanager
    def prompt_progress(self, cached, total):
        """a request's return_progress (self.on_progress set): on_progress(cached, processed, total) when the prefill
        starts, from a watcher thread every 0.5 s while it runs (the engine's mk_prefill_progress; ctypes releases the
        GIL) and when it is done, as llama-server's prompt_progress events. The watcher is joined before the request
        thread goes on, so the two never write to the client at once."""
        cb = getattr(self, "on_progress", None)
        if cb is None:
            yield
            return
        prog, stop, seen = getattr(self.eng, "prefill_progress", None), threading.Event(), [cached]

        def watch():
            while not stop.wait(0.5):
                p = prog() if prog else None
                if p and p[1] > 0 and cached + p[0] > seen[0]:
                    seen[0] = cached + p[0]
                    cb(cached, seen[0], total)
        cb(cached, cached, total)
        th = threading.Thread(target=watch, name="mk-progress", daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()
        cb(cached, total, total)

    def timings_now(self):
        """timings_per_token: the timings of the reply so far (llama-server's fields)"""
        prompt_n, prompt_ms, t1, n_gen = self.tick
        ms = max((time.time() - t1) * 1e3, 1e-6)
        return {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": n_gen, "predicted_ms": ms,
                "predicted_per_second": n_gen * 1e3 / ms}

    def prefill_text(self, n_new, prog, el):
        """prog: the engine's (done, total) of the batched part, or None (a library without it: elapsed time only)"""
        if prog and prog[1] > 0:
            done, total = prog
            s = f"#{self.rid} prefill {done:,} / {total:,} tok ({100 * done // total}%)"
            if done:
                rate = done / max(el, 1e-9)
                s += f", {rate:,.0f} tok/s, ~{secs((total - done) / rate)} left"
            return f"{s} | {secs(el)}"
        return f"#{self.rid} prefill {n_new:,} tok | {secs(el)}"

    def decode_text(self, n_gen, phase, steps, stepped, el):
        s = f"#{self.rid} decoding {n_gen:,} tok, {n_gen / max(el, 1e-9):.1f} tok/s"
        if steps:
            s += f", {stepped / steps:.1f} tok/cycle"
        ph = ", ".join(f"{k} {v:,}" for k, v in phase.items() if v)
        return s + (f" | {ph}" if ph else "") + f" | {secs(el)}"

    def load_prompt(self, toks, images=(), gone=None, first_turn=False, max_tokens=-1, tools=True):
        """brings the engine to `toks` processed, reusing what it has; returns (first token, reused count).
        images: [(pos, Vision.Image)] of the prompt. gone (the client left: () -> bool) is watched during the prefill,
        which then stops at a chunk boundary with PrefillCancelled (Engine.cancel_when). first_turn: the prompt has no
        earlier assistant reply; max_tokens: the request's (-1: none); tools: the request has tools (not tool_choice
        "none"); all three for cache_switch's side request rule"""
        eng = self.eng
        self.cache_pre = ""
        if self.parked is not None:  # (the host cache: --side-max / --park-mib)
            self.cache_switch(toks, [(p, im.n, im.key) for p, im in images], first_turn, max_tokens, tools)
        seq = eng.sequence()
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        match, img_cut = lcp, None
        # rewind checkpoint at the last <|im_start|> (the generation prompt): a follow-up turn re-renders this turn
        # from there (thinking mode: <think>\n\n</think> instead of <think>\n), so it must not sit at the prompt's end
        gen = -1
        if self.im_start is not None:
            for k in range(len(toks) - 1, -1, -1):
                if toks[k] == self.im_start:
                    gen = k
                    break
        # images: their rows' placeholder ids are alike, so the processed sequence matches only up to the first image
        # that is not the same image at the same position in both
        spans = [(p, im.n, im.key) for p, im in images]
        for p, _, _ in set(spans) ^ set(self.img_spans):
            if p < lcp:
                lcp, img_cut = p, p

        def feed(start):  # prefill toks[start:] with the images it overlaps (encoded now, each once: Vision's cache)
            fed = [(p, im) for p, im in images if p + im.n > start]
            if any(im.encode_s is None for _, im in fed):
                # the CPU encodes leave the GPU idle (up to seconds): keep its clocks up until the prefill's first launch
                eng.keep_warm()
            imgs, armed = [], time.monotonic()
            todo, k = sum(1 for _, im in fed if im.encode_s is None), 0
            for p, im in fed:  # (a video's frame pairs encode for seconds to minutes: a client that left stops them)
                if imgs and im.encode_s is None:
                    if gone is not None and gone():
                        raise PrefillCancelled(f"the client left during the image encodes ({len(imgs)} of {len(fed)} done)")
                    if time.monotonic() - armed > 30:  # (keep-warm re-armed through long encodes)
                        eng.keep_warm()
                        armed = time.monotonic()
                if im.encode_s is None:
                    k += 1
                    CON.status(f"#{self.rid} encoding image {k} / {todo} on the CPU ({im.nx}x{im.ny} tok)")
                imgs.append((p, im.nx, im.ny, im.rows()))
            # (a side request saves the generation prompt's checkpoint only: where its own conversation's follow-up resumes)
            cks = self.ck_positions(toks, start) if self.side is None else []
            pins = self.ck_pins(toks, lcp, spans) if self.side is None else []
            if pins:
                eng.ck_pin(pins)
            with eng.cancel_when(gone), self.prefill_status(len(toks) - start), self.prompt_progress(start, len(toks)):
                if imgs:
                    return eng.prefill(toks[start:], gen, cks, imgs), start
                return eng.prefill(toks[start:], gen, cks), start

        self.img_spans = []  # (until the prefill succeeds)
        had = f"the prompt matches {match:,} of the {len(seq):,} cached tok" + (
            f" (to the image at {img_cut:,}, which differs)" if img_cut is not None else "")
        # exact reuse (an MK_EXACT engine, --exact-reuse): never append to the sequence. Its last prompt row and the
        # reply were processed by the decode kernels, whose arithmetic differs from the batched prefill's; restoring
        # the last checkpoint at or before the divergence (all of them sit inside earlier batched prefills, on the
        # absolute 32-token scan grid) and prefilling from there computes what a prefill from 0 computes, bit for bit.
        if seq and lcp >= eng.committed() and not eng.exact:
            eng.truncate(lcp)
            self.cache_note = f"{had}: continued from there"
            r = feed(lcp)
        else:
            # the last checkpoint at or before the divergence (one inside a kept image: the image is fed again)
            ck = max((p for p in eng.checkpoints() if 0 < p <= lcp), default=0)
            if ck and eng.rewind_to(ck):
                self.cache_note = f"{had}: resumed at the checkpoint at {ck:,}"
                r = feed(ck)
            else:
                eng.reset()
                self.cache_note = f"CACHE MISS: {had}, no checkpoint before that" if seq else "nothing cached"
                r = feed(0)
        self.img_spans = spans
        self.live_gen, self.live_tools, self.live_low = gen, tools, self.next_low
        self.side_ready = self.side is not None
        self.cache_note = self.cache_pre + self.cache_note
        return r
