"""The engine's request lock, and the slot scheduler behind --agents."""
import collections
import itertools
import queue
import threading
import time
import traceback

from mk_errors import ContextOverflow, PrefillCancelled
from mk_limits import env_int
from mk_log import log


class FifoLock:
    """the engine's lock: requests get it in the order they arrive (threading.Lock wakes an arbitrary waiter)"""

    def __init__(self):
        self._cv = threading.Condition()
        self._tickets = itertools.count()
        self._turn = 0
        self._held = False

    def acquire(self):
        with self._cv:
            t = next(self._tickets)
            self._cv.wait_for(lambda: self._turn == t and not self._held)
            self._held = True
        return True

    def release(self):
        with self._cv:
            self._held = False
            self._turn += 1
            self._cv.notify_all()

    def locked(self):
        return self._held

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


SLOT_OWN_CTX = 1024  # --agents: the positions of the engine's own sequence (the warm-up's; the slots have their own)


def common_prefix(a, b):
    """the length of the common prefix of two token lists"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SlotJob:
    """a request in the slot scheduler (--agents): its prompt, sampling and token limit go in; each engine cycle's tokens
    come out through q (a list per cycle; []: the reply cannot go on, its slot's context is full; an exception: the
    engine failed). state: queued (waiting for a free slot), prefill, decode, done"""

    POLL_S = 0.05  # how often a request waiting for its tokens asks whether its client is still there

    def __init__(self, sched, toks, sp, max_new, rid=0):
        self.sched, self.toks, self.sp, self.max_new, self.rid = sched, toks, sp, max_new, rid
        self.q = queue.Queue()
        self.state, self.slot, self.reused, self.left, self.n = "queued", None, 0, len(toks), 0
        self.closed = False
        self.t_submit, self.t_begin = time.time(), None
        self.gone, self.rest = (lambda: False), []

    def close(self):
        """the request is done with the job (its reply ended, it failed, or its client left): the scheduler drops it
        from the queue, abandons its prefill, or takes its slot out of the next cycle"""
        if not self.closed:
            self.closed = True
            self.sched.wake()

    def _get(self, tick=None):
        """the next item of q; None once gone() is True (the job is closed)"""
        while True:
            try:
                item = self.q.get(timeout=self.POLL_S)
            except queue.Empty:
                if self.gone():
                    self.close()
                    return None
                if tick is not None:
                    tick()
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    def prompt(self, gone, progress=None):
        """waits until the prompt is processed: (the reply's first token, the prompt tokens the slot reused). gone() (the
        client left) is asked every POLL_S meanwhile: PrefillCancelled once it is True. progress(cached, processed,
        total): a request's return_progress (Server.prompt_progress's events), called when the prefill starts, as it
        goes (every 0.5 s at most) and when it is done"""
        self.gone = gone
        total, seen = len(self.toks), [None, 0.0]

        def tick():
            if progress is None or self.state != "prefill":
                return
            done, now = total - self.left, time.monotonic()
            if seen[0] is None or (done > seen[0] and now - seen[1] >= 0.5):
                seen[:] = [done, now]
                progress(self.reused, done, total)
        first = self._get(tick)
        if first is None:
            raise PrefillCancelled("while it waited for a free slot" if self.state == "queued" else
                                   f"at {total - self.left:,} of {total:,} tok")
        if not first:  # (Server._generate leaves the reply room in the context before it submits)
            raise RuntimeError("the slot had no room for the reply")
        if progress is not None:
            if seen[0] is None:
                progress(self.reused, self.reused, total)
            progress(self.reused, total, total)
        self.rest = first[1:]
        return first[0], self.reused

    def step(self):
        """the tokens of the job's next engine cycle; None: the client left meanwhile (the job is closed); []: the slot's
        context is full"""
        if self.rest:
            out, self.rest = self.rest, []
            return out
        return self._get()


class SlotScheduler:
    """--agents N: one thread owns the engine's parallel slots (MKEngine::slots_open: up to N conversations, each with
    its own KV cache and recurrent state) and decodes every admitted request together, one engine cycle for all of them
    (mk_slots_step: one batched launch; a lone slot runs the solo megakernel); each request thread (Server._generate) gets
    the tokens of its cycles through its SlotJob. Requests wait for a free slot in arrival order and take the one whose
    sequence shares the most with the prompt, less what that discards of the slot's last prompt (a conversation's next
    turn goes back to its own slot and prefills only its new tokens; an empty slot goes before another conversation's).
    While slots decode, a waiting prompt is fed in chunks, one per cycle right before their step (a failed chunk fails
    only its own request: the step is a call of its own; MK_PF_MIX rows, default 256; MK_PF_MIX=0: MK_PF_CHUNK tokens
    between the cycles, default 512, 0: the whole prompt), so they keep decoding while a long prompt is read; while none decodes,
    MK_PF_SOLO tokens per call (default 0: the whole prompt). With MK_EXACT the chunks change no row (the engine feeds
    every chunk but a prompt's last as batched rows, each ending on the 32-token scan grid: up to 31 tokens off the
    budget; a last chunk of 32 tokens or fewer joins the one before, so the prompt keeps its checkpoint), so a prompt that
    arrives while others decode gets the reply it gets alone. A closed job is dropped from the queue, its prefill abandoned at the next chunk, or its slot left out of the
    next cycle. Open-jet's SlotScheduler (--parallel / --agents), with jobs that can be closed at any point.

    With MK_EXACT (exact reuse; --agents sets it), a prompt never reuses the rows a slot decoded: those depend on how many
    drafts each cycle verified, which depends on the slots decoding beside it. A prompt that reaches into the slot's last
    reply first takes the slot back to a checkpoint inside its last prompt (slot_begin with that prompt's tokens), as
    Server.load_prompt restores a checkpoint with --exact-reuse (the engine does the same itself under MK_EXACT, so a slot
    whose bookkeeping a failure cleared stays exact too).

    The engine's keep-warm load (MK_KEEP_WARM_S) is armed when the scheduler goes idle and stopped before its next GPU
    work: the slot calls do not stop it themselves."""

    def __init__(self, eng, n, ctx):
        self.eng, self.n, self.ctx = eng, n, ctx
        self.exact = bool(getattr(eng, "exact", False))
        self.mix = env_int("MK_PF_MIX", 256, 0)
        self.chunk = env_int("MK_PF_CHUNK", 512, 0)
        self.solo = env_int("MK_PF_SOLO", 0, 0)
        self.cv = threading.Condition()
        self.pending = collections.deque()  # jobs waiting for a slot (under cv)
        self.free = list(range(n))
        self.filling, self.active = {}, {}  # slot -> job: its prompt is fed (the oldest first) / it decodes
        self.seq = [[] for _ in range(n)]  # each slot's tokens as the engine holds them (prompt, reply): the slot choice
        self.held = [0] * n  # the length of each slot's last prompt (what another conversation's prompt would discard)
        self.last = [None] * n  # each slot's latest job (GET /slots)
        self.warm = False  # the keep-warm load is armed
        self.cycles = 0
        threading.Thread(target=self.run, name="mk-slots", daemon=True).start()

    def submit(self, toks, sp, max_new, rid=0):
        """a request: toks, sp = (temp, top_p, top_k, min_p, seed), at most max_new tokens; returns its SlotJob"""
        job = SlotJob(self, toks, sp, max_new, rid)
        with self.cv:
            self.pending.append(job)
            self.cv.notify()
        return job

    def wake(self):
        with self.cv:
            self.cv.notify()

    def counts(self):
        """(requests waiting for a slot, requests in a slot)"""
        return len(self.pending), len(self.filling) + len(self.active)

    def full(self):
        """a request sent now would wait for a slot"""
        return len(self.pending) >= len(self.free)

    def snapshot(self):
        """each slot's state for GET /slots (read from the HTTP threads while the scheduler runs)"""
        out = []
        for s in range(self.n):
            j = self.last[s]
            st = j.state if j is not None and j.state in ("prefill", "decode") else "idle"
            out.append({"state": st, "id_task": j.rid if j else -1, "n_prompt_tokens": len(j.toks) if j else 0,
                        "n_cached": j.reused if j else 0, "prefill_left": j.left if st == "prefill" else 0,
                        "n_decoded": j.n if j else 0, "n_past": len(self.seq[s])})
        return out

    def _end(self, job):
        job.state = "done"
        job.q.put([])  # (a request still waiting for tokens: the reply ends here)

    def _finish(self, slot):
        job = self.active.pop(slot)
        self.free.append(slot)
        self._end(job)

    def _fail(self, slot, job, e):
        """the engine failed on the job's slot: the request gets the error; the slot's state is unknown (its next prompt
        is matched against nothing)"""
        job.state = "done"
        job.q.put(e)
        self.seq[slot], self.held[slot] = [], 0
        self.free.append(slot)

    def _prefilled(self, slot, job):
        """the job's prompt is in: it decodes from the next cycle on"""
        del self.filling[slot]
        job.max_new = min(job.max_new, self.ctx - len(job.toks) - ContextOverflow.RESERVE)
        if job.max_new <= 0:
            self.free.append(slot)
            self._end(job)
        else:
            job.state = "decode"
            self.active[slot] = job

    def run(self):
        while True:
            try:
                self.cycle()
            except Exception as e:  # (a fault of the scheduler's own: its requests get it, the next ones are served)
                log(f"slot scheduler: {e!r}")
                traceback.print_exc()
                for s in list(self.filling):
                    self._fail(s, self.filling.pop(s), e)
                for s in list(self.active):
                    self._fail(s, self.active.pop(s), e)

    def cycle(self):
        """admits waiting requests to free slots, feeds one prompt chunk, runs one engine cycle of the decoding slots"""
        eng = self.eng
        with self.cv:
            while True:
                for job in [j for j in self.pending if j.closed]:  # (its client left while it waited)
                    self.pending.remove(job)
                    self._end(job)
                if self.pending or self.filling or self.active:
                    break
                if not self.warm:  # idle: the GPU's clocks stay up for MK_KEEP_WARM_S, as after a request one at a time
                    self.warm = True
                    eng.keep_warm(True)
                self.cv.wait()
            admit = []
            while self.pending and self.free:
                job = self.pending.popleft()
                # the slot holding this conversation; else the one whose loss costs least (an empty one before evicting
                # another conversation for the few template tokens it shares): reused - discarded tokens of the slot's
                # last prompt (open-jet counts its reply too, which the conversation's own next turn re-renders: then a
                # long reply could send that turn to another conversation's slot)
                slot = max(self.free, key=lambda s: 2 * common_prefix(self.seq[s], job.toks) - self.held[s])
                self.free.remove(slot)
                admit.append((slot, job))
        if self.warm:
            self.warm = False
            eng.keep_warm(False)
        for slot, job in admit:
            try:
                p = self.held[slot]
                if self.exact and p and common_prefix(self.seq[slot], job.toks) >= p:  # (exact reuse: see the class)
                    eng.slot_begin(slot, job.toks[:p], *job.sp)
                job.reused = eng.slot_begin(slot, job.toks, *job.sp)
            except Exception as e:  # (reported to the request; the others go on)
                self._fail(slot, job, e)
                continue
            job.slot, job.left, job.t_begin, job.state = slot, len(job.toks) - job.reused, time.time(), "prefill"
            self.seq[slot], self.held[slot], self.last[slot] = list(job.toks), len(job.toks), job
            self.filling[slot] = job
        for slot in [s for s, j in self.active.items() if j.closed]:
            self._finish(slot)
        slots = None  # (the slots this cycle steps)
        if self.filling:  # one chunk of the oldest admitted prompt
            slot = next(iter(self.filling))
            job = self.filling[slot]
            try:
                if job.closed:  # (the slot keeps what was fed)
                    eng.slot_prefill(slot, 0)
                    del self.filling[slot]
                    self.seq[slot] = job.toks[:len(job.toks) - job.left]
                    self.held[slot] = len(self.seq[slot])
                    self.free.append(slot)
                    self._end(job)
                else:
                    mixed = self.active and self.mix > 0
                    if mixed:  # (a slot whose prompt this chunk completes decodes from the next cycle on)
                        slots = sorted(self.active)
                    job.left = eng.slot_prefill(slot, self.mix if mixed else (self.chunk if self.active else self.solo)
                                                or 1 << 30)
                    if job.left == 0:
                        self._prefilled(slot, job)
            except Exception as e:  # (the chunk's own request fails: the decoding slots were not stepped, they go on)
                self.filling.pop(slot, None)
                self._fail(slot, job, e)
        if slots is None:
            if not self.active:
                return
            slots = sorted(self.active)
        try:
            stepped = (slots, eng.slots_step(slots))
        except Exception as e:
            for s in slots:
                self._fail(s, self.active.pop(s), e)
            return
        self.cycles += 1
        for s, toks in zip(*stepped):
            job = self.active[s]
            if not toks:  # (its context is full: the reply ends at its length)
                self._finish(s)
                continue
            self.seq[s] += toks
            job.n += len(toks)
            job.q.put(toks)
            if job.n >= job.max_new or any(eng.is_eog(t) for t in toks):
                self._finish(s)
