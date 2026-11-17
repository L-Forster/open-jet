"""Server's host cache: side requests and parked conversations."""
import time

from mk_log import log, mib


class Parked:
    """a conversation parked in the engine's host store (Engine.park): what the server matches prompts against. side: a
    side request's own conversation (Engine.side_end_park); low: one that is likely never continued (a summary request's
    and the conversation it summarised: Server.cache_switch); both kinds are evicted before the others. tools: its last
    request had tools (None: unknown). pins: its pinned checkpoints (in cks too: Server.ck_pins)"""

    def __init__(self, pid, seq, cks, committed, img_spans, gen, nbytes, side=False, low=False, tools=None, pins=()):
        self.id, self.seq, self.cks, self.committed = pid, seq, cks, committed
        self.img_spans, self.gen, self.nbytes, self.t, self.side = img_spans, gen, nbytes, time.time(), side
        self.low, self.tools, self.pins = low, tools, tuple(pins)


SIDE_STORE_MIB = 1024  # host store of side requests when parking is off (--park-mib 0)
SIDE_REPLY_MAX = 1024  # with parking on, a side request with tools asks for at most this many tokens (cache_switch)
SIDE_REPLY_ROOM = 8192  # store room a side request takes for the rows of a reply it does not cap (Server.side_start)
SIDE_PARK_CKS = 1  # checkpoints kept with a side request's conversation (its generation prompt's: where a follow-up resumes)
SIDE_KEEP_MAX = 4  # side requests' conversations kept at most (the oldest goes first): most are titles, never continued
# a side request without tools that quotes at least --park-min tokens and answers at least this many (thinking aside) is
# taken for a summary of the conversation beside it (titles, tags, follow-ups and commit messages answer in tens)
SUMMARY_ANSWER_MIN = 256


class CacheMixin:
    def side_keep(self):
        """a side request's own conversation is parked after it (evicted before the other parked conversations) and a
        later bounded request continuing it runs as a side request again (needs parking on and the library's support)"""
        a = self.args
        return self.parked is not None and a.side_max > 0 and a.park_mib > 0 and getattr(self.eng, "side_keep", False)

    # ---- host cache: side requests and parked conversations, switched at request boundaries ----
    def resume_point(self, seq, committed, cks, cached_spans, toks, spans):
        """how load_prompt would continue `toks` (image spans `spans`) on the cached `seq`: (common prefix, the image
        position that cut it or None, the position it resumes at, 'append' | 'checkpoint' | 'reset')"""
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)
        match, img_cut = lcp, None
        for p, _, _ in set(spans) ^ set(cached_spans):
            if p < lcp:
                lcp, img_cut = p, p
        if seq and lcp >= committed and not self.eng.exact:
            return match, img_cut, lcp, "append"
        ck = max((p for p in cks if 0 < p <= lcp), default=0)
        return match, img_cut, ck, "checkpoint" if ck else "reset"

    def cache_switch(self, toks, spans, first_turn, max_tokens=-1, tools=True):
        """before load_prompt's own matching: brings back a parked conversation that this prompt continues better than
        the live one (parking the live one), or runs the request as a side request, or parks the live conversation when
        the request would discard it. Requests switch only here, between replies. With parking on, a side request is a
        one-shot: a first turn without tools (a client's title, tags, commit message, session name, summary; every
        agent harness sends its sessions' first turns with tools) whatever its max_tokens, or one whose reply is bounded
        (0 < max_tokens <= SIDE_REPLY_MAX); its own conversation is parked after it (side_finish), so a chat without
        tools continues as side requests, and any request without tools that would displace a conversation with tools
        runs beside it too. A new agent session's first turn (tools, a long reply) parks the live conversation like any
        conversation, so its next turns find it cached."""
        eng, a = self.eng, self.args
        self.cache_pre, self.next_low, self.debris = "", "", None
        seq = eng.sequence()
        live = self.resume_point(seq, eng.committed(), eng.checkpoints(), self.img_spans, toks, spans)
        lost = len(seq) - live[2]  # cached positions the normal path would discard
        cont = self.live_gen >= 0 and live[0] >= self.live_gen  # (diverges in the live one's generation prompt / reply)
        # a side request's own conversation that this one-shot prompt continues (it matches up to that conversation's
        # generation prompt): a side request again, resumed there, while the live conversation stays aside
        side_ok = (a.side_max > 0 and len(toks) <= a.side_max and not cont and lost >= max(2 * len(toks), a.park_min)
                   and (a.park_mib <= 0 or not tools or 0 < max_tokens <= SIDE_REPLY_MAX))
        if side_ok and self.side_keep() and self.parked:
            best = None
            for e in self.parked.values():
                if not e.side or e.gen < 0:
                    continue
                r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                if r[0] >= e.gen and r[2] > live[2] and (best is None or r[2] > best[1][2]):
                    best = (e, r)
            if best:
                self.side_start(seq, len(toks), max_tokens, resume=best[0])
                return
        if a.park_mib > 0 and self.parked:
            best = None
            for e in self.parked.values():
                r = self.resume_point(e.seq, e.committed, e.cks, e.img_spans, toks, spans)
                if r[2] >= live[2] + a.park_min and (best is None or r[2] > best[1][2]):
                    best = (e, r)
            if best:
                self.park_live(seq, "another conversation came back", keep=best[0].id, low=self.live_low)
                self.unpark(best[0])
                e, r = best
                if r[2] in e.pins and r[0] < e.gen:  # (a new session that shares only its system prompt: it stays parked)
                    self.park_live(eng.sequence(), "a new session reuses its system prompt",
                                   low="it was parked with low priority" if e.low else "")
                return
        # a first turn, or (parking on) a request without tools that would displace a conversation with tools: an agent
        # session stays live while a chat without tools or a client's summary (the conversation as messages) runs
        # beside it
        if side_ok and (first_turn or (not tools and self.live_tools and a.park_mib > 0)):
            # a summary of the live conversation (a long one without tools that answers at length: side_finish): if the
            # next request replaces the live conversation instead of continuing it, it was the summarised one, parked
            # with low priority
            after = "replaced right after a summary request beside it" if (
                not tools and len(toks) >= a.park_min and a.park_mib > 0) else ""
            self.side_start(seq, len(toks), max_tokens, after=after)
            return
        if a.park_mib > 0 and not cont and lost >= a.park_min:
            k = self.first_reply(toks)
            if k is not None and live[0] >= k + len(self.asst_marker):
                # everything up to the cached conversation's first reply is unchanged: this is that conversation with
                # its later history edited (a client moving or dropping a block, clearing old tool results, a chat
                # message edited); the request resumes from the engine's state either way, and the copy a park would
                # keep is never continued
                log(f"#{self.rid} not parked: the request continues the cached conversation with its history edited "
                    f"(the prompt matches {live[0]:,} of the {len(seq):,} cached tok, past its first reply at {k:,})")
                return
            # a request without tools that replaces a conversation whose last request had tools is how clients summarise
            # a conversation (compaction); the compacted conversation then replaces the summary's: both copies are
            # likely dead, parked with low priority (evicted before real parked conversations). Clients' other one-shots
            # that quote a conversation (titles, tags) look the same until they answer: the replaced conversation's mark
            # holds only for a summary's answer (debris_finish); the one-shot's own conversation stays low either
            # way, and takes no room from a conversation of normal priority when it is parked (park_live)
            debris = "" if tools or not self.live_tools else "replaced by a request without tools"
            mark = bool(debris) and not self.live_low  # (provisional: debris_finish may take it back)
            e = self.park_live(seq, "a new conversation", low=self.live_low or debris, provisional=mark)
            self.debris = e if mark else None
            self.next_low = debris and "the conversation of a request without tools that replaced one with tools"

    def first_reply(self, toks):
        """the position of the prompt's first assistant message (its <|im_start|>assistant tokens; a prompt without
        one: its generation prompt), or None"""
        m = self.asst_marker
        if not m:
            return None
        k = 0
        try:
            while True:
                k = toks.index(m[0], k)
                if tuple(toks[k:k + len(m)]) == m:
                    return k
                k += 1
        except ValueError:
            return None

    def store_update(self):
        """refreshes store_gauge (on the request thread: /metrics and /props never call into the engine)"""
        info = self.eng.store_info()
        self.store_gauge = (len(self.parked), info["used"], info["allocated"], tuple(len(e.seq) for e in self.parked.values()),
                            sum(e.side for e in self.parked.values()), sum(e.low for e in self.parked.values()))
        return info

    def make_room(self, need, budget, keep=None, side_only=False):
        """evicts parked conversations, side requests' own ones and low-priority ones first, each tier least recently
        parked first (never `keep`, the one about to be restored; side_only: only those), until `need` more bytes fit in
        `budget`; False: they do not"""
        while self.eng.store_info()["used"] + need > budget:
            pid = next((p for p, e in self.parked.items() if p != keep and (e.side or e.low)), None)
            if pid is None and not side_only:
                pid = next((p for p in self.parked if p != keep), None)
            if pid is None:
                return False
            e = self.parked[pid]
            kind = ", a side request" if e.side else ", low priority" if e.low else ""
            self.evict(pid, f"least recently parked{kind}) to make room")
        return True

    def evict(self, pid, why):
        e = self.parked.pop(pid)
        self.eng.park_drop(pid)
        self.count(park_evictions=1)
        self.store_update()
        log(f"#{self.rid} evicted the parked conversation of {len(e.seq):,} tok ({mib(e.nbytes)}, {why}")

    def park_live(self, seq, why, keep=None, low="", provisional=False):
        """a snapshot of the live conversation into the host store (the engine keeps it too); keep: a parked one that must
        stay (it is restored next); low: why it is parked with low priority ("": it is not), provisional: that mark may
        be taken back (Server.debris). Returns its Parked entry (None: not parked)"""
        eng, a = self.eng, self.args
        if len(seq) < a.park_min:
            return
        need, budget = eng.park_need(a.park_cks), a.park_mib << 20
        if need > budget:
            log(f"#{self.rid} not parked: the cached {len(seq):,} tok need {mib(need)} of host memory "
                f"(--park-mib {a.park_mib:,})")
            return
        if low and not provisional:
            # a low-priority conversation would be evicted before any normal one: as a side request's own conversation
            # (side_finish), it only takes room that side and other low-priority ones leave, and evicts nothing when
            # that is not enough
            spare = budget - eng.store_info()["used"] + sum(e.nbytes for p, e in self.parked.items()
                                                             if p != keep and (e.side or e.low))
            if need > spare or not self.make_room(need, budget, keep, side_only=True):
                log(f"#{self.rid} not parked: the cached {len(seq):,} tok ({mib(need)}) have low priority ({low}) and "
                    f"do not fit in the host store without evicting a conversation of normal priority")
                return
        else:
            self.make_room(need, budget, keep)
        # (the engine parks the newest --park-cks checkpoints and the pinned ones: Engine.ck_pin)
        pins = eng.pinned() if a.park_cks > 0 and getattr(eng, "ck_pins", False) else []
        cks = sorted(set(sorted(eng.checkpoints())[-a.park_cks:]) | set(pins)) if a.park_cks > 0 else []
        pid = eng.park(a.park_cks)
        if pid is None:
            log(f"#{self.rid} not parked: no room in the host store for the cached {len(seq):,} tok ({mib(need)})")
            return
        e = Parked(pid, seq, cks, eng.committed(), self.img_spans, self.live_gen, need, low=bool(low),
                   tools=self.live_tools, pins=pins)
        self.parked[pid] = e
        info = self.store_update()
        self.count(parks=1)
        why += f", low priority: {low}" if low else ""
        log(f"#{self.rid} parked the cached conversation ({why}): {len(seq):,} tok, {mib(info['last_bytes'])} copied in "
            f"{info['last_ms']:.0f} ms | {len(self.parked)} parked, {mib(info['used'])} of {a.park_mib:,} MiB")
        return e

    def unpark(self, e):
        """the engine becomes the parked conversation e (its entry is freed)"""
        eng = self.eng
        try:
            eng.unpark(e.id)
        except RuntimeError:  # (an engine error: the entry goes too, with its host memory)
            self.parked.pop(e.id, None)
            eng.park_drop(e.id)
            self.store_update()
            raise
        del self.parked[e.id]
        self.img_spans, self.live_gen, self.live_tools, self.live_low = e.img_spans, e.gen, e.tools, ""
        info = self.store_update()
        self.count(park_restores=1)
        self.cache_pre = f"restored a parked conversation ({len(e.seq):,} tok); "
        log(f"#{self.rid} restored the parked conversation of {len(e.seq):,} tok: {mib(info['last_bytes'])} copied in "
            f"{info['last_ms']:.0f} ms | {len(self.parked)} parked, {mib(info['used'])} in use")

    def side_start(self, seq, n, max_tokens=-1, resume=None, after=""):
        """runs the request as a side request: the live sequence stays cached (Engine.side_begin; side_finish brings it
        back after the reply). resume: a side request's parked conversation it continues (Engine.side_unpark); after:
        the low-priority reason the live conversation gets when it is back ("": its own)"""
        eng = self.eng
        # room for what the request may overwrite: its prompt and its reply (max_tokens, else SIDE_REPLY_ROOM: past that
        # the engine saves more rows while there is room, else the kept sequence is dropped), and the checkpoint blocks
        # of the live one in the slots it saves into
        pt, rings, state, ck = eng.kv_sizes()
        reply = min(max_tokens, SIDE_REPLY_ROOM) if max_tokens > 0 else SIDE_REPLY_ROOM
        self.make_room(state + rings + pt * min(len(seq), n + reply) + 2 * ck, self.store_mib << 20,
                       keep=resume.id if resume else None)
        if not eng.side_begin():
            log(f"#{self.rid} side request: no room in the host store: the cached {len(seq):,} tok are not kept")
            return
        self.side = (len(seq), self.img_spans, self.live_gen, self.live_tools, self.live_low, after)
        self.side_ready = False
        self.count(side_requests=1)
        self.cache_pre = f"side request: the cached {len(seq):,} tok are kept aside; "
        if resume is not None:
            if not eng.side_unpark(resume.id):
                log(f"#{self.rid} side request: no room in the host store to continue its parked conversation of "
                    f"{len(resume.seq):,} tok")
                return
            del self.parked[resume.id]
            self.img_spans, self.live_gen, self.live_tools = resume.img_spans, resume.gen, resume.tools
            info = self.store_update()
            self.count(side_resumed=1)
            self.cache_pre += f"its parked conversation ({len(resume.seq):,} tok) restored in {info['last_ms']:.0f} ms; "

    def side_finish(self):
        """after a side request (also one that failed or whose client left): the kept sequence comes back; with
        side_keep(), the side request's own conversation is parked first (when its prompt was processed and it has a
        checkpoint to resume from)"""
        if self.side is None:
            return
        n, spans, gen, tools, low, after = self.side
        self.side, eng, a = None, self.eng, self.args
        ok, pid, own, room = False, None, None, True
        try:
            if eng.side_active():
                if self.side_keep() and self.side_ready and eng.checkpoints():
                    sides = [p for p, e in self.parked.items() if e.side]  # (least recently parked first)
                    for p in sides[:max(0, len(sides) - SIDE_KEEP_MAX + 1)]:
                        self.evict(p, f"the oldest of {SIDE_KEEP_MAX} kept side request conversations)")
                    own = (list(eng.sequence()), sorted(eng.checkpoints())[-SIDE_PARK_CKS:], eng.committed(), self.img_spans,
                           self.live_gen, eng.park_need(SIDE_PARK_CKS))
                    # (no room: other kinds of parked conversations are not evicted for it)
                    room = own[5] <= a.park_mib << 20 and self.make_room(own[5], a.park_mib << 20, side_only=True)
                if own is not None and room:
                    ok, pid = eng.side_end_park(SIDE_PARK_CKS)
                else:
                    ok = eng.side_end()
        except RuntimeError as e:
            log(f"#{self.rid} side request: {e}")
            eng.side_drop()  # (whatever is left of it: the engine starts over)
            own = None
        info = self.store_update()
        kept = ""
        if pid is not None:
            self.parked[pid] = Parked(pid, *own, side=True, tools=self.live_tools)
            info = self.store_update()
            self.count(side_kept=1)
            kept = (f"; its own conversation ({len(own[0]):,} tok) parked: {mib(info.get('side_park_bytes', 0))} in "
                    f"{info.get('side_park_ms', 0.0):.0f} ms | {len(self.parked)} parked, {mib(info['used'])} in use")
        elif own is not None and ok:
            kept = "; its own conversation was not parked (no room in the host store)"
        if ok:
            self.img_spans, self.live_gen, self.live_tools = spans, gen, tools
            self.live_low = after if after and self.answer_tokens >= SUMMARY_ANSWER_MIN else low
            log(f"#{self.rid} side request done: the cached {n:,} tok are back ({mib(info['last_bytes'])}: kept in "
                f"{info['side_ms']:.0f} ms, back in {info['last_ms']:.0f} ms){kept}")
        elif own is not None and room:  # (side_end_park: the kept sequence was lost, the side request's conversation stays)
            self.count(side_dropped=1)
            log(f"#{self.rid} side request: the cached {n:,} tok were dropped (the host store ran full): the side "
                f"request's own conversation ({len(own[0]):,} tok) stays cached")
        else:
            self.img_spans, self.live_gen, self.live_tools, self.live_low = [], -1, None, ""
            self.count(side_dropped=1)
            log(f"#{self.rid} side request: the cached {n:,} tok were dropped (the host store ran full, or an engine "
                f"error): the next request prefills its prompt in full")

    def debris_finish(self, replied):
        """after a request without tools that parked the conversation with tools it replaced with low priority
        (cache_switch: likely a summary of it, the compacted conversation comes next): a reply whose answer is shorter
        than a summary's (SUMMARY_ANSWER_MIN tokens, thinking aside, as side_finish counts) gives that conversation, if
        it is still parked, normal priority again (a client's title or tags task quoting it). The mark stays after a
        summary's answer and without a reply (replied False: an error, or the client left): clients cut long summaries
        and move on or retry, and a retry continues the cut request's prompt, which parks nothing and decides nothing"""
        e, self.debris = self.debris, None
        if e is None or not replied or self.answer_tokens >= SUMMARY_ANSWER_MIN or self.parked.get(e.id) is not e:
            return
        e.low = False
        self.store_update()
        log(f"#{self.rid} the parked conversation of {len(e.seq):,} tok keeps normal priority: the request answered in "
            f"{self.answer_tokens:,} tok (< {SUMMARY_ANSWER_MIN}: not a summary)")
