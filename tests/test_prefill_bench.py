import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid


SPEC = importlib.util.spec_from_file_location(
    "prefill_bench", Path(__file__).resolve().parents[1] / "megakernel" / "bench" / "prefill_bench.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
GPU = "GPU-00000000-0000-0000-0000-000000000003"


def response(n=2048):
    return dict(timings=dict(prompt_n=n, prompt_ms=2000.0, predicted_n=1),
                tokens_evaluated=n, tokens_predicted=1, tokens_cached=n, truncated=False)


class PrefillBenchTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(__file__).parent / (".prefill-bench-test-" + uuid.uuid4().hex)
        self.work.mkdir()
        self.addCleanup(shutil.rmtree, self.work)
        self.model = dict(sha256="mock-model-hash", size_bytes=123)
        self.workload = bench.workload_from_ids([i % 256 for i in range(8192)], self.model, {"description": "mock"})
        self.args = SimpleNamespace(gpu=GPU, llama_server="mock-server", model="mock.gguf",
                                    megakernel="mock-mk", megakernel_lib="mock-mk.so",
                                    draft_vocab="mock-draft-vocab.bin",
                                    context=16384, draft=4, port=8095, startup_timeout=10)

    def test_gpu_uuid_pin_and_profiling_cleanup_preserve_parent_environment(self):
        inherited = dict(PATH="mock-path", CUDA_VISIBLE_DEVICES="0,1", CUDA_DEVICE_ORDER="FASTEST_FIRST",
                         MK_PREFILL="mk", MK_PROFILE="1", MK_PF_PROFILE="1")
        with mock.patch.dict(os.environ, inherited, clear=True):
            env = bench.benchmark_env(GPU)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU)
            self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(env["PATH"], "mock-path")
            self.assertFalse({"MK_PREFILL", "MK_PROFILE", "MK_PF_PROFILE"} & env.keys())
            self.assertEqual(dict(os.environ), inherited)
        for pin in ("3", "0,1", "GPU-invalid", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.benchmark_env(pin)

    def test_workload_exact_shared_prefixes_and_model_identity(self):
        path = self.work / "workload.json"
        path.write_text(json.dumps(self.workload))
        loaded = bench.load_workload(path, self.model)
        self.assertEqual(loaded["prompts"][0]["token_ids"], loaded["prompts"][2]["token_ids"][:2048])
        with self.assertRaisesRegex(ValueError, "model differs"):
            bench.load_workload(path, dict(sha256="other", size_bytes=123))
        self.workload["prompts"][0]["token_ids"][0] = 999
        path.write_text(json.dumps(self.workload))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            bench.load_workload(path, self.model)

    def test_workload_rejects_count_or_token_type_changes(self):
        path = self.work / "workload.json"
        for bad in ([1] * 2047, [True] * 2048, [-1] * 2048):
            self.workload["prompts"][0]["token_ids"] = bad
            path.write_text(json.dumps(self.workload))
            with self.assertRaisesRegex(ValueError, "token IDs or prompt count"):
                bench.load_workload(path, self.model)
        with self.assertRaisesRegex(ValueError, "at least"):
            bench.workload_from_ids([1], self.model, {})

    def test_mk_native_prompt_timer_not_decode_or_reported_rounded_rate(self):
        raw = "[mk] prompt 2048 tok in 2.000 s (1023.9 tok/s)\n[mk] decode 0 tok in 0.000 s = nan tok/s\n"
        sample = bench.parse_mk(raw, 2048)
        self.assertEqual(sample["seconds"], 2)
        self.assertEqual(sample["tokens_per_second"], 1024)
        self.assertEqual(sample["stderr"], raw)
        for bad in (raw.replace("2048", "2047"), raw.replace("2.000", "0.000"),
                    raw.replace("decode 0", "decode 1"), raw + raw, "decode only"):
            with self.assertRaises(ValueError):
                bench.parse_mk(bad, 2048)

    def test_llama_requires_full_uncached_prompt_and_one_output(self):
        valid = response()
        self.assertEqual(bench.parse_llama(valid, 2048)["tokens_per_second"], 1024)
        # Final occupancy can equal the input length even when nearly all input was reused.
        for timings in (dict(prompt_n=1), dict(prompt_n=2047), dict(prompt_ms=0),
                        dict(prompt_ms=float("nan")), dict(prompt_ms=float("inf")),
                        dict(predicted_n=2)):
            invalid = response()
            invalid["timings"].update(timings)
            with self.assertRaises(ValueError):
                bench.parse_llama(invalid, 2048)
        for patch in (dict(truncated=True), dict(tokens_evaluated=2049),
                      dict(tokens_cached=2047), dict(tokens_predicted=2)):
            invalid = response()
            invalid.update(patch)
            with self.assertRaises(ValueError):
                bench.parse_llama(invalid, 2048)

    def test_aggregation_uses_every_processed_token_and_native_second(self):
        result = bench.aggregate([dict(n=2048, seconds=1), dict(n=2048, seconds=3)])
        self.assertEqual(result["tokens_per_second"], 1024)
        self.assertEqual(result["sample_count"], 2)
        for samples in ([], [dict(n=0, seconds=1)], [dict(n=1, seconds=0)],
                        [dict(n=1, seconds=float("inf"))]):
            with self.assertRaises(ValueError):
                bench.aggregate(samples)

    def test_llama_slot_erase_every_request_and_unreported_warmup(self):
        proc = mock.Mock()
        proc.poll.return_value = None

        def post(url, payload):
            if "/slots/" in url:
                return dict(id_slot=0, n_erased=2048)
            return response(len(payload["prompt"]))

        with mock.patch.object(bench, "llama_server") as server, \
                mock.patch.object(bench, "post", side_effect=post) as http, \
                contextlib.redirect_stdout(io.StringIO()):
            server.return_value.__enter__.return_value = ("http://mock", proc, ["mock-server"])
            samples, _ = bench.run_llama(self.args, self.workload, self.work)
        self.assertEqual(len(samples), 9)
        self.assertEqual(http.call_count, 24)
        for offset, prompt in zip((0, 8, 16), self.workload["prompts"]):
            for i in range(4):
                erase, completion = http.call_args_list[offset + 2 * i:offset + 2 * i + 2]
                self.assertTrue(erase.args[0].endswith("/slots/0?action=erase"))
                payload = completion.args[1]
                self.assertEqual(payload["prompt"], prompt["token_ids"])
                self.assertFalse(payload["cache_prompt"])
                self.assertEqual(payload["n_predict"], 1)
                self.assertEqual(payload["temperature"], 0)
        self.assertEqual([s["repetition"] for s in samples], [1, 2, 3] * 3)

    def test_llama_startup_failure_and_occupied_port_do_not_use_external_server(self):
        with mock.patch.object(bench.socket, "socket") as sock, \
                mock.patch.object(bench.subprocess, "Popen") as launch:
            sock.return_value.__enter__.return_value.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                with bench.llama_server(self.args, self.work):
                    pass
            launch.assert_not_called()
        proc = mock.Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with mock.patch.object(bench.socket, "socket"), \
                mock.patch.object(bench.subprocess, "Popen", return_value=proc) as launch, \
                mock.patch.object(bench.urllib.request, "urlopen") as health:
            with self.assertRaisesRegex(RuntimeError, "exited 1"):
                with bench.llama_server(self.args, self.work):
                    pass
            health.assert_not_called()
            proc.terminate.assert_not_called()
            proc.wait.assert_called_once()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--slot-save-path") + 1], str(self.work.resolve()))

    def fake_engine(self):
        engine = mock.Mock()
        state = {"sequence": []}
        engine.reset.side_effect = lambda: state.update(sequence=[])

        def prefill(ids):
            state["sequence"] = ids + [123]
            return 123

        engine.prefill.side_effect = prefill
        engine.sequence.side_effect = lambda: state["sequence"]
        engine.committed.side_effect = lambda: len(state["sequence"]) - 1
        return engine

    def test_warmed_mk_api_reset_count_validation_and_cleanup(self):
        engine = self.fake_engine()
        env = os.environ.copy()
        ticks = [tick for i in range(12) for tick in (i * 3.0, i * 3.0 + 2.0)]
        with mock.patch.object(bench, "engine_class", return_value=mock.Mock(return_value=engine)) as factory, \
                mock.patch.object(bench.time, "perf_counter", side_effect=ticks), \
                contextlib.redirect_stdout(io.StringIO()):
            samples, _ = bench.run_mk_api(self.args, self.workload, self.work)
        self.assertEqual(len(samples), 9)
        self.assertEqual(engine.reset.call_count, 12)
        self.assertEqual(engine.prefill.call_count, 12)
        self.assertEqual(samples[0]["seconds"], 2.0)
        self.assertEqual(samples[0]["n"], 2048)
        self.assertEqual(factory.return_value.call_args.args,
                         ("mock-mk.so", "mock.gguf", 16384, 4, "mock-draft-vocab.bin", 1))
        engine.L.mk_free.assert_called_once_with(engine.h)
        self.assertEqual(dict(os.environ), env)

    def test_api_failure_frees_own_engine_and_restores_environment(self):
        engine = self.fake_engine()
        engine.committed.side_effect = None
        engine.committed.return_value = 2047
        before = os.environ.copy()
        with mock.patch.object(bench, "engine_class", return_value=mock.Mock(return_value=engine)):
            with self.assertRaisesRegex(ValueError, "processed-token count"):
                bench.run_mk_api(self.args, self.workload, self.work)
        engine.L.mk_free.assert_called_once_with(engine.h)
        self.assertEqual(dict(os.environ), before)

    def test_preparation_tokenizes_once_without_prefill_and_releases_engine(self):
        engine = self.fake_engine()
        engine.tokenize.return_value = list(range(9000))
        before = os.environ.copy()
        with mock.patch.object(bench, "engine_class", return_value=mock.Mock(return_value=engine)):
            data = bench.prepare_workload(self.args, self.model)
        self.assertEqual([p["n"] for p in data["prompts"]], [2048, 4096, 8192])
        engine.tokenize.assert_called_once_with(bench.source_text())
        engine.prefill.assert_not_called()
        engine.L.mk_free.assert_called_once_with(engine.h)
        self.assertEqual(dict(os.environ), before)
        engine.tokenize.return_value = [1]
        with mock.patch.object(bench, "engine_class", return_value=mock.Mock(return_value=engine)):
            with self.assertRaisesRegex(ValueError, "at least"):
                bench.prepare_workload(self.args, self.model)
        self.assertEqual(dict(os.environ), before)

    def test_cli_keeps_all_reps_and_passes_exact_shared_ids(self):
        def run(command, **kwargs):
            ids = Path(command[command.index("--prompt-ids") + 1]).read_text().split()
            n = len(ids)
            self.assertEqual(list(map(int, ids)), self.workload["prompts"][(2048, 4096, 8192).index(n)]["token_ids"])
            self.assertEqual(command[command.index("-n") + 1], "1")
            self.assertEqual(command[command.index("--kv") + 1], "q8")
            self.assertEqual(command[command.index("--draft-vocab") + 1], "mock-draft-vocab.bin")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU)
            return SimpleNamespace(returncode=0, stdout="",
                                   stderr=f"[mk] prompt {n} tok in 2.000 s (1000.0 tok/s)\n[mk] decode 0 tok")

        with mock.patch.object(bench.subprocess, "run", side_effect=run) as child, \
                contextlib.redirect_stdout(io.StringIO()):
            samples, commands = bench.run_mk(self.args, self.workload, self.work)
        self.assertEqual(len(samples), 9)
        self.assertEqual(len(commands), 3)
        self.assertEqual(child.call_count, 12)

    def test_cli_report_persists_raw_samples_hashes_and_weighted_aggregates(self):
        output = self.work / "result.json"
        samples = [dict(n=p["n"], repetition=rep, seconds=float(rep),
                        token_sha256=p["token_sha256"], first_token=123)
                   for p in self.workload["prompts"] for rep in (1, 2, 3)]
        metadata = dict(path="mock-file", size_bytes=123, sha256="mock-model-hash", mtime_ns=0)
        argv = ["prefill_bench", "--backend", "megakernel", "--gpu", GPU,
                "--model", "mock.gguf", "--megakernel-lib", "mock-mk.so",
                "--workload", "mock-workload.json", "--output", str(output)]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(bench, "fingerprint", return_value=metadata), \
                mock.patch.object(bench, "load_workload", return_value=self.workload), \
                mock.patch.object(bench, "run_mk_api", return_value=(samples, {"api": "mock"})) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            bench.main()
        run.assert_called_once()
        report = json.loads(output.read_text())
        self.assertEqual(report["samples"], samples)
        self.assertEqual(report["aggregate"]["processed_tokens"], (2048 + 4096 + 8192) * 3)
        self.assertEqual(report["aggregate"]["seconds"], 18)
        self.assertEqual(report["by_length"]["2048"]["tokens_per_second"], 1024)
        self.assertEqual(report["executable"]["sha256"], metadata["sha256"])
        self.assertEqual(report["inputs"][0]["token_sha256"], self.workload["prompts"][0]["token_sha256"])
        self.assertIn("synchronous Engine.prefill", report["settings"]["timing_scope"])


if __name__ == "__main__":
    unittest.main()
