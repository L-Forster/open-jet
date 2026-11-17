"""Server's status views: counters, /props, /v1/models, /slots, /metrics."""
import os
import threading
import time

import mk_limits
from mk_chat import MEDIA_MARKER
from mk_limits import DEF_MIN_P, DEF_TEMP, DEF_TOP_K, DEF_TOP_P, MAX_STOP, WARN_KEYS, WARN_KEY_LEN, WARN_TEXT_LEN
from mk_log import log


class InfoMixin:
    def warn_once(self, key, msg):
        """a console warning, once per key; bounded, as a request's fields can be in the key and the text"""
        key = key[:WARN_KEY_LEN]
        if key not in self.warned and len(self.warned) < WARN_KEYS:
            self.warned.add(key)
            log("warning:", msg[:WARN_TEXT_LEN])

    # ---- reporting: /props, /v1/models, /slots, /metrics ----
    METRICS = ("requests", "requests_bad", "requests_failed", "requests_cancelled", "deferred", "prompt_tokens",
               "prompt_tokens_reused", "prompt_seconds", "tokens_predicted", "predicted_seconds", "decode_cycles",
               "draft_tokens", "draft_tokens_accepted", "side_requests", "side_dropped", "side_kept", "side_resumed", "parks",
               "park_restores", "park_evictions")
    _mlock = threading.Lock()
    t_start = 0.0
    warm_s = 0.0

    def count(self, **kw):
        """adds to the counters of /metrics"""
        with self._mlock:
            m = self.__dict__.setdefault("metrics", dict.fromkeys(self.METRICS, 0))
            for k, v in kw.items():
                m[k] += v

    def counters(self):
        with self._mlock:
            return dict(self.__dict__.get("metrics") or dict.fromkeys(self.METRICS, 0))

    def busy(self):
        return self.lock.locked()

    def processing(self):
        """requests the engine works on (/metrics)"""
        return int(self.busy())

    def n_vocab(self):
        return getattr(self.eng, "n_vocab", 0)

    def env_number(self, key, default):
        """an integer setting from the environment as /props shows it (default when unset); main() checks only those it
        uses, so one that is no integer is shown as the default, with a one-time warning (not a 500 on every /props)"""
        v = os.environ.get(key) or default
        try:
            return int(v)
        except ValueError:
            self.warn_once(f"env-{key}", f"{key}={str(v)[:40]!r} is not an integer: /props shows {default}")
            return default

    def engine_info(self):
        """what the server knows for sure about its engine: arguments, environment, the library's own answers"""
        a, e, v = self.args, self.eng, self.vision
        g = lambda o, k, d=None: getattr(o, k, d)  # noqa: E731 (fake engines in the tests have fewer attributes)
        return {
            "build": g(e, "build"), "lib": os.path.basename(str(g(a, "lib", ""))),
            "n_ctx": a.ctx, "kv": g(a, "kv"),  # (the KV format asked for; "auto" is resolved inside the engine)
            "mtp": g(e, "has_mtp", True), "mtp_file": g(a, "mtp"),
            "draft_depth": g(a, "draft") if g(e, "has_mtp", True) else 0,
            "draft_vocab": os.path.basename(a.draft_vocab) if isinstance(g(a, "draft_vocab"), str) else None,
            "exact_reuse": bool(g(e, "exact", False)),
            "checkpoints": {"pool": bool(g(e, "pool", False)), "every": g(a, "ck_every"), "messages": bool(g(a, "ck_msgs")),
                            "pinned": bool(g(a, "ck_pin") and g(e, "ck_pins", False)),
                            "slots": self.env_number("MK_CK_SLOTS", 8)},
            "keep_warm_s": self.warm_s, "prefill_cancel": bool(g(e, "_cancel", False)),
            "host_cache": None if self.parked is None else {
                "budget_mib": self.store_mib, "side_max": g(a, "side_max"), "park_mib": g(a, "park_mib"),
                "park_min": g(a, "park_min"), "park_checkpoints": g(a, "park_cks"), "parked": self.store_gauge[0],
                "parked_tokens": list(self.store_gauge[3]), "parked_side": self.store_gauge[4],
                "parked_low": self.store_gauge[5],
                "side_conversations": bool(self.side_keep()), "used_mib": round(self.store_gauge[1] / 2**20, 1),
                "allocated_mib": round(self.store_gauge[2] / 2**20, 1)},
            "vision": None if v is None else {
                "mmproj": os.path.basename(str(g(a, "mmproj", ""))), "backend": g(v, "backend"),
                "threads": g(v, "n_threads"), "image_min_tokens": g(a, "image_min_tokens"),
                "image_max_tokens": g(a, "image_max_tokens"), "image_cache": g(v, "cache_max")},
            "limits": {"max_body_mb": mk_limits.MAX_BODY_MB, "max_inflight_mb": mk_limits.MAX_INFLIGHT_MB,
                       "max_connections": mk_limits.MAX_CONNECTIONS, "request_timeout_s": mk_limits.REQUEST_TIMEOUT_S,
                       "max_images": mk_limits.MAX_IMAGES, "max_stop": MAX_STOP,
                       "image_max_pixels": self.env_number("MK_IMAGE_MAX_PIXELS", mk_limits.IMAGE_MAX_PIXELS)},
            "env": {k: "(set)" if k == "MK_API_KEY" and val else val  # (not the key: /props ends up in bug reports)
                    for k, val in sorted(os.environ.items()) if k.startswith("MK_")},
            "busy": self.busy(), "uptime_s": round(time.time() - self.t_start, 1) if self.t_start else 0.0,
        }

    def props(self):
        """the fields of llama-server's GET /props that the server has, plus `engine` (engine_info)"""
        b = getattr(self.eng, "build", None) or {}
        params = {"temperature": DEF_TEMP, "top_k": DEF_TOP_K, "top_p": DEF_TOP_P, "min_p": DEF_MIN_P, "seed": -1,
                  "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": -1, "n_predict": -1, "stop": [],
                  "stream": False, "n_probs": 0}
        return {
            "default_generation_settings": {"n_ctx": self.args.ctx, "params": params},
            "total_slots": 1, "model_alias": self.model_id, "model_path": getattr(self.args, "model", None),
            "modalities": {"vision": self.vision is not None, "video": bool(getattr(self, "video_ok", False)),
                           "audio": False},
            "media_marker": MEDIA_MARKER, "endpoint_slots": True, "endpoint_props": False, "endpoint_metrics": True,
            "chat_template": getattr(self, "template_src", None),
            "build_info": f"{b.get('commit', 'unknown')} {b.get('flags', '')}".strip() if b else "unknown",
            "is_sleeping": False, "engine": self.engine_info(),
        }

    def models(self):
        """GET /v1/models: the one model, with llama-server's meta where the server has it"""
        meta = {"n_ctx": self.args.ctx}
        try:
            meta["size"] = os.path.getsize(self.args.model)
        except (OSError, TypeError):
            pass
        if getattr(self.eng, "images", False):
            meta["n_embd"] = self.eng.L.mk_n_embd()
        if self.n_vocab():
            meta["n_vocab"] = self.n_vocab()
        caps = ["completion"] + (["multimodal"] if self.vision is not None else [])
        return {"object": "list",
                "data": [{"id": self.model_id, "object": "model", "created": int(self.t_start), "owned_by": "openjet",
                          "meta": meta}],
                "models": [{"name": self.model_id, "model": self.model_id, "type": "model", "capabilities": caps,
                            "details": {"format": "gguf"}}]}

    def slots(self):
        """GET /slots: the one slot (the llama-server fields the server has)"""
        return [{"id": 0, "n_ctx": self.args.ctx, "is_processing": self.busy(),
                 "speculative": bool(getattr(self.eng, "has_mtp", True)) and bool(getattr(self.args, "draft", 0))}]

    def metrics_text(self):
        """GET /metrics: Prometheus text; llama-server's names (llamacpp:) where they mean the same, megakernel: else"""
        m = self.counters()
        cyc, drafted, accepted = m["decode_cycles"], m["draft_tokens"], m["draft_tokens_accepted"]
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed (prefilled; reused ones not).",
             m["prompt_tokens"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt process time", m["prompt_seconds"]),
            ("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", m["tokens_predicted"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Predict process time", m["predicted_seconds"]),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.",
             m["prompt_tokens"] / m["prompt_seconds"] if m["prompt_seconds"] else 0.0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.",
             m["tokens_predicted"] / m["predicted_seconds"] if m["predicted_seconds"] else 0.0),
            ("llamacpp:requests_processing", "gauge", "Number of requests processing.", self.processing()),
            ("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", m["deferred"]),
            ("megakernel:prompt_tokens_reused_total", "counter", "Prompt tokens reused from the engine's sequence.",
             m["prompt_tokens_reused"]),
            ("megakernel:decode_cycles_total", "counter", "Engine decode cycles (one verify + MTP draft each).", cyc),
            ("megakernel:draft_tokens_total", "counter", "MTP draft tokens produced.", drafted),
            ("megakernel:draft_tokens_accepted_total", "counter", "MTP draft tokens accepted.", accepted),
            ("megakernel:requests_total", "counter", "Generation requests received.", m["requests"]),
            ("megakernel:requests_bad_total", "counter", "Generation requests answered with a 4xx.", m["requests_bad"]),
            ("megakernel:requests_failed_total", "counter", "Generation requests that failed (5xx).",
             m["requests_failed"]),
            ("megakernel:requests_cancelled_total", "counter", "Generation requests whose client disconnected.",
             m["requests_cancelled"]),
            ("megakernel:side_requests_total", "counter", "Side requests (the cached sequence kept in host memory).",
             m["side_requests"]),
            ("megakernel:side_requests_dropped_total", "counter", "Side requests after which the cached sequence was lost.",
             m["side_dropped"]),
            ("megakernel:side_conversations_parked_total", "counter", "Side requests whose own conversation was parked.",
             m["side_kept"]),
            ("megakernel:side_conversations_resumed_total", "counter",
             "Side requests that continued a parked side request's conversation.", m["side_resumed"]),
            ("megakernel:parks_total", "counter", "Conversations parked in host memory.", m["parks"]),
            ("megakernel:park_restores_total", "counter", "Parked conversations brought back.", m["park_restores"]),
            ("megakernel:park_evictions_total", "counter", "Parked conversations evicted (host memory budget).",
             m["park_evictions"]),
            ("megakernel:parked_conversations", "gauge", "Conversations parked in host memory now.", self.store_gauge[0]),
            ("megakernel:host_cache_bytes", "gauge", "Host memory in use by parked conversations.", self.store_gauge[1]),
        ]
        return "".join(f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v:g}\n" if isinstance(v, float) else
                       f"# HELP {n} {h}\n# TYPE {n} {t}\n{n} {v}\n" for n, t, h, v in rows)
