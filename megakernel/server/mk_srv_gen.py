"""Server's replies: sampling parameters and the generation loop."""
import codecs
import random
import time

from mk_chat import OutputParser, StopMatcher, check_params, stop_strings
from mk_errors import BadRequest, ClientGone, ContextOverflow, PrefillCancelled
from mk_limits import DEF_MIN_P, DEF_TEMP, DEF_TOP_K, DEF_TOP_P, FLT_MAX, TOPK_MAX
from mk_log import CON, log, plural, secs
from mk_media import VideoBytes


class GenerateMixin:
    def sampling(self, body):
        """sets the request's sampling on the engine (sampling_params); returns it as text for the console"""
        temp, top_p, top_k, min_p, seed, pen, desc = self.sampling_params(body)
        self.eng.set_penalty(*pen)
        self.eng.set_sampling(temp, top_p, top_k, min_p, seed)
        return desc

    def sampling_params(self, body):
        """the request's sampling: (temp, top_p, top_k, min_p, seed, (presence, frequency penalty), text for the
        console), greedy (temperature <= 0) as the engine takes it.
        top_k: absent -> DEF_TOP_K; <= 0 ("off" for llama.cpp / OpenAI-style clients) -> TOPK_MAX, the engine's
        maximum; above TOPK_MAX -> TOPK_MAX with a one-time warning"""
        try:
            temp = float(body["temperature"] if body.get("temperature") is not None else DEF_TEMP)
            top_p = float(body["top_p"] if body.get("top_p") is not None else DEF_TOP_P)
            top_k = int(body["top_k"]) if body.get("top_k") is not None else DEF_TOP_K
            min_p = float(body["min_p"] if body.get("min_p") is not None else DEF_MIN_P)
            seed = body.get("seed")
            seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("temperature, top_p, top_k, min_p and seed must be numbers")
        if not all(abs(v) <= FLT_MAX for v in (temp, top_p, min_p)):  # (NaN would run as greedy, inf at its own meaning)
            raise BadRequest("temperature, top_p and min_p must be finite numbers")
        if top_k <= 0:
            top_k = TOPK_MAX
        elif top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        # presence / frequency penalty, vLLM's semantics: tokens generated in this response so far (not the prompt)
        pen = []
        for k in ("presence_penalty", "frequency_penalty"):
            try:
                v = float(body.get(k) or 0)
            except (TypeError, ValueError, OverflowError):
                raise BadRequest(f"{k} must be a number")
            if not -2.0 <= v <= 2.0:
                raise BadRequest(f"{k} must be in [-2, 2]")
            pen.append(v)
        try:
            rep = float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("repetition_penalty must be a number")
        if rep != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            temp, top_p, top_k, min_p = 0.0, 1.0, TOPK_MAX, 0.0
            desc = "greedy"
        else:
            desc = f"temp {temp:g} top_p {top_p:g} top_k {top_k} min_p {min_p:g}"
            if body.get("seed") is not None and int(body["seed"]) >= 0:
                desc += f" seed {seed}"
        if any(pen):
            desc += f" presence {pen[0]:g} frequency {pen[1]:g}"
        return temp, top_p, top_k, min_p, seed, tuple(pen), desc

    def start_reply(self, toks, max_new, images, gone, first_turn, asked, tools):
        """processes the prompt (load_prompt): (the reply's first token, the prompt tokens reused, the engine cycle: a
        function returning the next tokens). SlotServer: in a slot, by the slot scheduler's thread"""
        tok, reused = self.load_prompt(toks, images, gone, first_turn, asked, tools)
        return tok, reused, self.eng.step

    def eng_stats(self):
        """the engine's (decode cycles, drafted, accepted) counters, or None. They move in its decode cycles only and
        restart at a reset (a full re-prefill in load_prompt): read after the prompt, the difference is the reply's"""
        return self.eng.stats() if getattr(self.eng, "stats", None) is not None else None

    def status(self, text):
        """the console's status line for the running request (SlotServer: none, its requests run at once)"""
        CON.status(text)

    def request_done(self):
        """after each generation request (the handler): the status line goes, the keep-warm window restarts"""
        CON.status("")
        self.eng.keep_warm()  # (MK_KEEP_WARM_S: the window restarts after every request)

    def generate(self, body, emit, gone=lambda: False):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas, and in a
        stream (body["stream"]) with ('tool_call', delta.tool_calls[] entry) as the tool calls are generated.
        Returns (finish_reason, tool_calls, usage, timings). emit returning False, or gone() returning True (the client
        closed its connection; asked before the prefill, during it by a watcher thread (the prefill stops at its next
        chunk) and before each engine cycle) stops it with ClientGone."""
        check_params(body)
        self.answer_tokens = 0
        self.status(f"#{self.rid} preparing the prompt (template, tokens, media)")
        text, thinking, media = self.render(body)
        toks, images = self.tokenize_prompt(text, media)
        replied = False
        try:
            out = self._generate(body, emit, gone, thinking, toks, images, media)
            replied = True
            return out
        finally:
            if images:  # (the request's images stay pinned until here)
                self.vision.release([im for _, im in images])
            self.side_finish()  # (a side request: the cached sequence comes back, also after an error or a disconnect)
            self.debris_finish(replied)
            if self.job is not None:  # (--agents: its slot leaves the next cycle, if the reply did not end there)
                self.job.close()
                self.job = None

    def _generate(self, body, emit, gone, thinking, toks, images, media=()):
        max_new = next((body[k] for k in ("max_tokens", "max_completion_tokens", "n_predict") if body.get(k) is not None), -1)
        try:
            max_new = int(max_new)
        except (TypeError, ValueError, OverflowError):
            raise BadRequest("max_tokens must be an integer")
        if max_new == 0:  # (OpenAI's and Anthropic's minimum is 1)
            raise BadRequest("max_tokens must be at least 1 (-1: up to the context)")
        asked = max_new
        limit = self.args.ctx - len(toks) - ContextOverflow.RESERVE
        if limit <= 0:
            raise ContextOverflow(len(toks), self.args.ctx)
        max_new = limit if max_new <= 0 else min(max_new, limit)
        # tool_choice "none": the tools stay in the prompt, calls are not parsed (returned as text), as llama-server does
        tools = (body.get("tools") or []) if body.get("tool_choice") != "none" else []
        stopper = StopMatcher(stop_strings(body.get("stop")))
        self.stop_word = None
        samp = self.sampling(body)
        rid = self.rid
        route, model, t_req = self.req
        n_vid = sum(1 for d in media if isinstance(d, VideoBytes))
        what = [f"{len(toks):,} tok", plural(len(body.get("messages") or []), "message")]
        if tools:
            what.append(plural(len(tools), "tool"))
        if len(media) > n_vid:
            what.append(plural(len(media) - n_vid, "image"))
        if n_vid:
            what.append(plural(n_vid, "video"))
        how = ["thinking" if thinking else "no thinking"] + ([f"max_tokens {asked:,}"] if asked > 0 else [])
        log(f"#{rid} {route}{' stream' if body.get('stream') else ''}"
            f"{f' {str(model)[:48]!r}' if isinstance(model, str) and model else ''}: "
            f"{', '.join(what)} | {', '.join(how)} | {samp}")
        if gone():  # (it left while the request waited for the engine)
            log(f"#{rid} the client disconnected before its prompt was processed")
            raise ClientGone()
        self.status(f"#{rid} prefill")
        t0 = time.time()
        enc0 = [im.encode_s for _, im in images]
        first_turn = not any(isinstance(m, dict) and m.get("role") == "assistant" for m in body.get("messages") or [])
        try:
            tok, reused, step = self.start_reply(toks, max_new, images, gone, first_turn, asked, bool(tools))
        except PrefillCancelled as e:  # (the engine kept nothing of it)
            log(f"#{rid} the client disconnected during the prefill of {len(toks):,} tok: stopped after "
                f"{secs(time.time() - t0)} ({e})")
            raise ClientGone()
        t1 = time.time()
        t0 += self.queued_s  # (--agents: the wait for a free slot is not prompt time)
        st0 = self.eng_stats()

        def spent():
            st1 = self.eng.stats() if st0 is not None else None
            d = tuple(b - a for a, b in zip(st0, st1)) if st0 and st1 else (0, 0, 0)
            self.count(decode_cycles=d[0], draft_tokens=d[1], draft_tokens_accepted=d[2])
            return d
        self.last_prompt = (len(toks), reused)  # (an Anthropic stream's message_start reports it)
        enc = [im.encode_s for (_, im), e in zip(images, enc0) if e is None and im.encode_s is not None]
        n_new, enc_s = len(toks) - reused, sum(enc)
        pf_s = max(t1 - t0 - enc_s, 1e-9)
        line = (f"#{rid} prompt: {reused:,} cached + {n_new:,} prefilled in {secs(pf_s)} ({n_new / pf_s:,.0f} tok/s)"
                f" | {self.cache_note}")
        if images:
            grids = sorted({f"{im.nx}x{im.ny}" for _, im in images})
            line += (f" | images: {len(images)} ({', '.join(grids[:3])}{', ...' if len(grids) > 3 else ''} tok grids), "
                     f"{len(enc)} encoded on the CPU in {secs(enc_s)}")
        self.status(f"#{rid} decoding")
        log(line)
        # a stream gets the tool calls as they are generated (ToolCallParser's events). reasoning_format (OpenAI route; an
        # Anthropic reply has its thinking blocks): none -> the thinking stays in the content, deepseek-legacy -> both
        # (a stream: in the content only, as llama-server)
        fmt = "auto" if route == "messages" else body.get("reasoning_format") or getattr(self.args, "reasoning_format", "auto")
        inline = {"none": "only", "deepseek-legacy": "only" if body.get("stream") else "both"}.get(fmt)
        parser = OutputParser(thinking, tools, bool(body.get("stream")), body.get("parallel_tool_calls") is False, inline)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, pending, aborted, dropped = 0, "length", [tok], False, 0
        # t_max_predict_ms (llama-server's): past it, the reply ends at the next token with a newline (finish "length")
        t_max, timed_out = float(body.get("t_max_predict_ms") or 0) / 1e3, False
        self.tick = (n_new, (t1 - t0) * 1e3, t1, 0)  # (timings_now)
        phase = {"thinking": 0, "answer": 0, "tool call": 0}  # generated tokens by the part of the reply they are in
        steps = stepped = 0  # engine cycles, the tokens they produced
        live, shown = CON.tty(), time.monotonic()
        while n_gen < max_new and not aborted:
            if not pending:
                if gone():  # once per engine cycle: a closed connection stops the GPU work here
                    aborted = True
                    break
                pending = step()
                if not pending:  # (--agents: the client left during the cycle (None), or the slot's context is full)
                    aborted = pending is None
                    break
                steps += 1
                stepped += len(pending)
                if live and time.monotonic() - shown >= 0.25:
                    shown = time.monotonic()
                    self.status(self.decode_text(n_gen, phase, steps, stepped, time.time() - t1))
            t = pending.pop(0)
            n_gen += 1
            if self.eng.is_eog(t):
                finish = "stop"
                break
            mode0 = parser.mode
            self.tick = self.tick[:3] + (n_gen,)
            piece = dec.decode(self.eng.piece(t))
            for kind, delta in parser.feed(stopper.feed(piece)):
                if emit(kind, delta) is False:
                    aborted = True
                    break
            phase["thinking" if mode0 == "reasoning" else "tool call" if parser.mode == "tools" else "answer"] += 1
            if stopper.word is not None:
                # a stop string: the reply ends at it, and the tokens after this one (the rest of the cycle's accepted
                # drafts) are dropped. The engine's sequence keeps them, as after max_tokens: the next request finds
                # the divergence (load_prompt) and continues from the reply that was sent.
                finish, dropped = "stop", len(pending)
                break
            if t_max > 0 and "\n" in piece and time.time() - t1 > t_max:  # (the rest of the cycle: as after a stop)
                finish, dropped, timed_out = "length", len(pending), True
                break
        if aborted:  # (the engine's state is that of a reply cut by max_tokens: the next request reuses it as usual)
            spent()
            log(f"#{rid} the client disconnected after {n_gen:,} generated tok ({secs(time.time() - t1)}): stopped")
            raise ClientGone()
        # (bytes the decoder still holds start a character the reply was cut inside: dropped, not sent as U+FFFD)
        for kind, delta in parser.feed(stopper.feed("", final=True), final=True):
            emit(kind, delta)
        self.stop_word = stopper.word
        self.reasoning_tokens, self.answer_tokens = phase["thinking"], phase["answer"]
        calls = parser.calls
        if body.get("parallel_tool_calls") is False:
            calls = calls[:1]  # (the first call only)
        rest = parser.unparsed()
        lost = parser.stray()
        if lost:  # (Qwen's template tells the model not to write after a call: rare, but never silent)
            log(f"#{rid} dropped {sum(map(len, lost)):,} chars of text the model wrote between or after its tool calls "
                f"(a reply carries the calls only)")
        shown = parser.unparsed(streamed=True) if body.get("stream") else rest
        if shown:
            emit("content", shown)
        # a call cut by max_tokens is text (a stream's client has it as a call it started, arguments incomplete), and the
        # reply ends with "length" (complete calls before it are still sent)
        if calls and not (rest and finish == "length"):
            finish = "tool_calls"
        t2 = time.time()
        dt = max(t2 - t1, 1e-9)
        # (one token is the prefill's: no decode rate)
        parts = [f"{n_gen:,} tok in {secs(dt)}" + (f" = {n_gen / dt:.1f} tok/s" if n_gen > 1 else "")]
        ph = ", ".join(f"{k} {v:,}" for k, v in phase.items() if v)
        if ph:
            parts.append(ph)
        _, drafted, accepted = spent()
        if steps:  # (MTP: each engine cycle verifies the drafts and keeps the accepted ones plus one token)
            spec = f"{stepped / steps:.2f} tok/cycle"
            if drafted > 0:
                spec += f", {100 * accepted / drafted:.0f}% of {drafted:,} drafts accepted"
            parts.append(spec)
        parts.append(f"finish {finish}" + (f" ({plural(len(calls), 'call')})" if calls else "")
                     + (f" (stop string; {dropped} tok of the cycle dropped)" if stopper.word is not None else "")
                     + (f" (t_max_predict_ms; {dropped} tok of the cycle dropped)" if timed_out else ""))
        if t_req:
            parts.append(f"request {secs(t2 - t_req)}")
        log(f"#{rid} reply: " + " | ".join(parts))
        self.count(prompt_tokens=len(toks) - reused, prompt_tokens_reused=reused, prompt_seconds=t1 - t0,
                   tokens_predicted=n_gen, predicted_seconds=t2 - t1)
        usage = {"prompt_tokens": len(toks), "completion_tokens": n_gen, "total_tokens": len(toks) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused},
                 "completion_tokens_details": {"reasoning_tokens": phase["thinking"]}}
        timings = {"prompt_n": len(toks) - reused, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - t1) * 1e3, "predicted_per_second": n_gen / max(t2 - t1, 1e-9)}
        return finish, calls, usage, timings
