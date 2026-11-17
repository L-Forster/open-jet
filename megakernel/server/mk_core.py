"""Server (setup and state; its methods are in the mk_srv_* mixins) and its --agents form, SlotServer."""
import collections
import contextlib
import itertools
import os
import threading
import time

import mk_limits
from mk_engine import Engine
from mk_errors import BadRequest
from mk_log import log, mib, plural, secs
from mk_media import VIDEO_PARTS, VideoDecoder, Vision, find_ffmpeg
from mk_slots import FifoLock, SLOT_OWN_CTX, SlotScheduler
from mk_srv_cache import CacheMixin, SIDE_STORE_MIB
from mk_srv_gen import GenerateMixin
from mk_srv_info import InfoMixin
from mk_srv_prompt import PromptMixin
from mk_template import CHATML_TEMPLATE, LiteralTemplate, make_template


class Server(CacheMixin, InfoMixin, PromptMixin, GenerateMixin):
    vision = None   # Vision (--mmproj), else text only
    decoder = None  # VideoDecoder (an ffmpeg was found): video input with the vision encoder
    video_ok = False  # video input works in this process (mmproj loaded, frame-pair encoder, ffmpeg): /props
    img_spans = ()  # the images of the engine's sequence: (pos, n, key); replaced, never mutated
    stop_word = None  # the stop string that ended the last reply (None: it ended otherwise)
    last_prompt = (0, 0)  # the running request's prompt: (tokens, reused tokens), once it is processed
    reasoning_tokens = 0  # the last reply's tokens generated inside its thinking (a Response's reasoning_tokens)
    answer_tokens = 0  # the running (or last) reply's tokens of its answer (not thinking, not a tool call)
    rids = itertools.count(1)  # request numbers for the console
    rid = 0  # the running (or last) request's number: "#rid" on its console lines
    req = ("chat", None, 0.0)  # its (route, model field, start time), set by the handler when it gets the engine
    cache_note = ""  # how load_prompt matched the last prompt against the engine's sequence (the console's prompt line)
    # host cache (Engine.store): parked conversations by id, least recently parked first (None: no store); the running
    # side request's kept state (None: none) and whether its own prompt was processed (side_ready: its conversation can be
    # parked after it); the live sequence's generation prompt position (-1: none; a request whose prompt matches up to it
    # continues the live conversation)
    parked = None
    store_mib = 0
    side = None
    side_ready = False
    live_gen = -1
    # whether the live conversation's last request had tools (None: unknown), and why the live conversation is parked
    # with low priority when another one replaces it ("": it is not); cache_switch decides next_low for the request it
    # runs, which becomes live_low once its prompt is processed
    live_tools = None
    live_low = next_low = ""
    # the parked conversation cache_switch marked low because the running request (no tools) replaced it: debris_finish
    # takes the mark back if the reply's answer is shorter than a summary's (None: no such mark)
    debris = None
    asst_marker = ()  # the tokens of "<|im_start|>assistant": where a prompt's first reply starts (Server.first_reply)
    # (parked, bytes in use, bytes allocated, their token counts, how many are side requests' own, how many have low
    # priority): /metrics and /props read it
    store_gauge = (0, 0, 0, (), 0, 0)
    cache_pre = ""  # what the host cache did for the running request (before cache_note on the prompt line)
    tick = (0, 0.0, 0.0, 0)  # the running reply's (prompt tokens, prompt ms, its start, generated tokens): timings_now
    # --agents (SlotServer): the slot scheduler (None: one request at a time on the engine's own sequence); the running
    # request's return_progress callback (its handler sets it), slot job, slot sampling and wait for a free slot (seconds)
    sched = None
    on_progress = None
    job = None
    slot_sp = None
    queued_s = 0.0

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, self.engine_ctx(args), args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv], "none" if args.no_mtp else args.mtp)
        if not self.eng.has_mtp:
            log("no MTP layer: plain decoding (depth 0)")
        # --chat-template-file / --chat-template (read and checked in main) replace the GGUF's template everywhere it is
        # used: chat and Anthropic requests, /apply-template, token counts, /props; a GGUF without one gets ChatML
        src = getattr(args, "chat_template_src", None)
        self.template_src = src if src is not None else self.eng.chat_template()
        if not self.template_src:
            log("the GGUF has no chat template: using ChatML (as llama-server; --chat-template-file to give one)")
            self.template_src = CHATML_TEMPLATE
        self.template = make_template(self.template_src)
        self.setup_specials(args)
        im = self.eng.tokenize("<|im_start|>")
        self.im_start = im[0] if len(im) == 1 else None  # checkpoint at the start of the generation prompt
        if self.im_start is not None:
            self.asst_marker = tuple(self.eng.tokenize("<|im_start|>assistant"))
        self.sys_role = self.eng.tokenize("system")  # (a prompt that opens with a system message: ck_pins)
        if getattr(args, "ck_pin", 0) and not getattr(self.eng, "ck_pins", False):
            log(f"pinned checkpoints off: {os.path.basename(args.lib)} has no mk_ck_pin (rebuild it): a new session "
                f"re-reads a client's system prompt once the cached conversation has grown or was parked")
        if args.mmproj:
            if not self.eng.images:
                raise RuntimeError(f"{args.lib} has no image input (mk_prefill_img): rebuild it")
            self.vision = Vision(args.vision_lib, args.mmproj, args.model, args.vision_threads, args.image_min_tokens,
                                 args.image_max_tokens, args.image_cache, args.vision_backend, args.video_cache)
            if self.vision.n_embd != self.eng.L.mk_n_embd():
                raise RuntimeError(f"mmproj rows of {self.vision.n_embd} for a model of {self.eng.L.mk_n_embd()}")
            ids = [self.eng.tokenize(t) for t in ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError("the vocabulary has no <|vision_start|> / <|vision_end|> / <|image_pad|> tokens")
            self.vis_start, self.vis_end, self.img_pad = (i[0] for i in ids)
            self.setup_video(args)
        self.warm_up()
        self.setup_store(args)
        self.model_id = os.path.basename(args.model)
        self.lock = FifoLock()
        self.warned = set()
        self.setup_slots(args)
        vis = (f", vision on the CPU ({os.path.basename(args.mmproj)}, {self.vision.backend}, {self.vision.n_threads} threads)"
               if self.vision else "")
        self.t_start = time.time()
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts{vis}")
        self.warm_s = self.eng.keep_warm()  # (MK_KEEP_WARM_S: the first reply starts at full clocks too)

    def setup_specials(self, args):
        """--content-specials text (the default; env MK_CONTENT_SPECIALS): special-token text inside the request's
        messages and tools is tokenized as plain text (LiteralTemplate, tokenize_prompt); parse: as the special tokens,
        as llama-server does"""
        if getattr(args, "content_specials", "parse") != "text":
            log("special-token text in messages is tokenized as special tokens (--content-specials parse, as llama-server)")
            return
        controls = self.eng.controls()
        if not controls:
            log(f"special-token text in messages is tokenized as special tokens: {os.path.basename(args.lib)} has no "
                f"mk_tokenize2 / mk_token_attr (rebuild it)")
            return
        self.template = LiteralTemplate(self.template, controls)

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and then requests fail with a CUDA out-of-memory error. Warming up here moves that
        failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        self.eng.prefill(toks)
        self.eng.step()
        self.eng.reset()
        if self.vision:
            self.vision.warm_up()

    def engine_ctx(self, args):
        """the positions of the engine's own sequence (SlotServer: the slots have their own)"""
        return args.ctx

    def setup_slots(self, args):
        """(SlotServer: opens the engine's parallel slots and starts their scheduler)"""

    def setup_store(self, args):
        """host memory for side requests and parked conversations (--side-max, --park-mib; an engine without the host
        store: neither)"""
        side, park = getattr(args, "side_max", 0), getattr(args, "park_mib", 0)
        if not getattr(self.eng, "store", False) or not (side > 0 or park > 0):
            if (side > 0 or park > 0) and not getattr(self.eng, "store", False):
                log(f"side requests / parked conversations off: {args.lib} has no host store (mk_park): rebuild it")
            return
        self.parked = collections.OrderedDict()
        self.store_mib = max(park, SIDE_STORE_MIB if side > 0 else 0)
        self.eng.store_budget(self.store_mib << 20)
        pt, rings, state, ck = self.eng.kv_sizes()
        log(f"host cache (allocated on first use, at most {self.store_mib:,} MiB): side requests "
            f"{f'up to {side:,} tok' if side > 0 else 'off'}, parked conversations "
            f"{f'up to {park:,} MiB, {args.park_cks} checkpoints each' if park > 0 else 'off'} ({pt:,} B per token, "
            f"{mib(state)} recurrent state, {mib(ck)} per checkpoint)")
        if side > 0 and park > 0 and not self.side_keep():
            log(f"a side request's own conversation is not kept: {os.path.basename(args.lib)} has no mk_side_end_park "
                f"(rebuild it): its next turn is prefilled in full")

    def setup_video(self, args):
        """video input (input_video / video_url parts): needs the vision library's frame pairs (mkv_frames_load) and an
        ffmpeg (find_ffmpeg); without them video requests get a 400 and video_ok stays False"""
        if args.video_max_tokens and self.vision.backend != "vit_cpu":
            raise RuntimeError("--video-max-tokens needs the cpu vision backend (libmtmd resizes frames as images)")
        if not self.vision.frames_ok:
            log(f"video input off: {args.vision_lib} has no video frames (mkv_frames_load): rebuild it")
            return
        exe = find_ffmpeg(args.ffmpeg)
        if not exe:
            log("video input off: no ffmpeg (--ffmpeg / MK_FFMPEG, ffmpeg on PATH, or pip install imageio-ffmpeg)"
                if not (args.ffmpeg or os.environ.get("MK_FFMPEG")) else
                f"video input off: ffmpeg not found at {args.ffmpeg or os.environ.get('MK_FFMPEG')}")
            return
        dec = VideoDecoder(exe)
        try:
            dec.check()
        except RuntimeError as e:
            log(f"video input off: {e}")
            return
        self.decoder, self.video_ok = dec, True
        log(f"video input: {exe} ({dec.version}), {args.video_fps:g} fps, frames of at most "
            f"{args.video_max_tokens or args.image_max_tokens or 'the default'} tokens, {mk_limits.MAX_VIDEO_FRAMES} frames per request")


class PerRequest:
    """a Server attribute that belongs to the running request: in a SlotServer each request thread has its own (Server's
    class value until the thread sets one)"""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, srv, owner=None):
        if srv is None:
            return self
        return getattr(srv.local, self.name, getattr(Server, self.name))

    def __set__(self, srv, value):
        setattr(srv.local, self.name, value)


class NoLock(contextlib.nullcontext):
    """SlotServer.lock: requests do not take turns on the engine (the slot scheduler's thread runs it)"""

    @staticmethod
    def locked():
        return False


class SlotServer(Server):
    """--agents N: up to N requests are served at once, each in a parallel slot of the engine (SlotScheduler), with the
    reply it gets when it runs alone. Everything above the engine is Server's: the request threads render, tokenize,
    parse, stream and report as a request running alone does, with their per-request state of their own (PerRequest).
    What the slots cannot serve is refused with a 400 (images and video: slot prompts are token ids; the presence /
    frequency penalties: slots sample with temperature, top_p, top_k, min_p and seed); the features of the engine's
    one sequence (side requests, parked conversations, pinned checkpoints) are off."""

    stop_word = PerRequest()
    last_prompt = PerRequest()
    reasoning_tokens = PerRequest()
    answer_tokens = PerRequest()
    rid = PerRequest()
    req = PerRequest()
    tick = PerRequest()
    cache_note = PerRequest()
    cache_pre = PerRequest()
    on_progress = PerRequest()
    job = PerRequest()
    slot_sp = PerRequest()
    queued_s = PerRequest()

    def __init__(self, args):
        self.local = threading.local()
        if args.mmproj:
            log("--mmproj ignored: --agents serves text only (images and video need the server without --agents)")
            args.mmproj = None
        super().__init__(args)

    def engine_ctx(self, args):
        return SLOT_OWN_CTX

    def setup_store(self, args):
        """(no host cache: side requests and parked conversations move the engine's one sequence aside)"""

    def setup_slots(self, args):
        if not getattr(self.eng, "slots", False):
            raise RuntimeError(f"--agents needs parallel slots: {os.path.basename(args.lib)} has no mk_slots_open2 "
                               f"(rebuild it)")
        args.ctx = self.eng.slots_open(args.agents, args.ctx)
        self.lock = NoLock()
        self.sched = SlotScheduler(self.eng, args.agents, args.ctx)
        s = self.sched
        kv = " ".join(f"{k}={os.environ[k]}" for k in ("MK_KV", "MK_KV_WIN") if os.environ.get(k))
        if s.mix:
            feed = f"{s.mix} rows of a waiting prompt ride along with each decode cycle"
        elif s.chunk:
            feed = f"a waiting prompt is fed {s.chunk} tokens at a time between the decode cycles"
        else:
            feed = "a waiting prompt is fed whole between two decode cycles (the other slots wait meanwhile)"
        if s.solo:
            feed += f", {s.solo} tokens at a time while no slot decodes"
        log(f"--agents {args.agents}: {plural(args.agents, 'slot')} of {args.ctx:,} positions each (q8_0 KV{', ' + kv if kv else ''}"
            f"{', MK_EXACT: prefill rows independent of the chunking' if self.eng.exact else ''}); {feed}; side requests, "
            f"parked conversations, pinned checkpoints and images are off; presence / frequency penalties are refused")

    def status(self, text):
        pass

    def request_done(self):
        """(the keep-warm window is the slot scheduler's: armed when it goes idle)"""

    def busy(self):
        return self.sched.full()

    def processing(self):
        return self.sched.counts()[1]

    def counters(self):
        m = super().counters()
        m["deferred"] = self.sched.counts()[0]  # (requests waiting for a free slot)
        m["decode_cycles"] = self.sched.cycles  # (engine cycles of all slots together; drafts are not counted per slot)
        return m

    def eng_stats(self):
        return None  # (the engine counts the cycles of every slot together)

    def props(self):
        p = super().props()
        p["total_slots"] = self.sched.n
        return p

    def engine_info(self):
        info = super().engine_info()
        waiting, running = self.sched.counts()
        s = self.sched
        info["agents"] = {"slots": s.n, "slot_ctx": self.args.ctx, "running": running, "waiting": waiting,
                          "prefill_mix_rows": s.mix, "prefill_chunk": s.chunk, "prefill_solo": s.solo, "cycles": s.cycles}
        return info

    def slots(self):
        """GET /slots: llama-server's fields for each slot, and its state in the scheduler"""
        spec = bool(getattr(self.eng, "has_mtp", True)) and bool(getattr(self.args, "draft", 0))
        return [{"id": k, "n_ctx": self.args.ctx, "is_processing": d["state"] != "idle", "speculative": spec, **d}
                for k, d in enumerate(self.sched.snapshot())]

    def render(self, body):
        for m in body.get("messages") or []:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", *VIDEO_PARTS) for p in c):
                raise BadRequest("image and video input are not available with --agents: the parallel slots take text "
                                 "prompts only (run the server without --agents for them)")
        return super().render(body)

    def sampling(self, body):
        temp, top_p, top_k, min_p, seed, pen, desc = self.sampling_params(body)
        for k, v in zip(("presence_penalty", "frequency_penalty"), pen):
            if v:
                raise BadRequest(f"{k} is not available with --agents: the parallel slots sample with temperature, "
                                 f"top_p, top_k, min_p and seed only")
        self.slot_sp = (temp, top_p, top_k, min_p, seed)
        return desc

    def start_reply(self, toks, max_new, images, gone, first_turn, asked, tools):
        job = self.job = self.sched.submit(toks, self.slot_sp, max_new, self.rid)
        tok, reused = job.prompt(gone, self.on_progress)
        self.queued_s = job.t_begin - job.t_submit
        self.cache_note = f"slot {job.slot}" + (f" (after {secs(self.queued_s)} in the queue)" if self.queued_s >= 0.01 else "")
        return tok, reused, job.step
