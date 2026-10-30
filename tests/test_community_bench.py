import asyncio
import importlib.util
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from src import cli, community_bench

_spec = importlib.util.spec_from_file_location(
    "bench_table", Path(__file__).resolve().parents[1] / "scripts" / "bench_table.py"
)
bench_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_table)


class FakeServer(BaseHTTPRequestHandler):
    """Answers chat completions with llama.cpp-style timings: 3 characters per prompt token at 2 ms each,
    10 ms per generated token. A recall question is answered with the code from the prompt."""

    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.requests.append(body)
        prompt = body["messages"][-1]["content"]
        match = re.search(r"deployment is (\d+)", prompt)
        answer = match.group(1) if match and "What is the code" in prompt else "def test_it(): pass"
        n_prompt, n_gen = max(1, len(prompt) // 3), body["max_tokens"]
        data = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "timings": {"prompt_n": n_prompt, "prompt_ms": n_prompt * 2.0, "predicted_n": n_gen,
                        "predicted_ms": n_gen * 10.0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _record(results: dict, **overrides) -> dict:
    record = {
        "schema": community_bench.SCHEMA, "created": "2026-10-29T12:00:00Z", "contributor": "octo-cat",
        "openjet_version": "0.5.2",
        "system": {"gpu": {"name": "NVIDIA GeForce RTX 4090", "vram_mb": 24564, "driver": "580.1"},
                   "gpu_count": 1, "os": "Ubuntu 24.04 LTS", "arch": "x86_64", "cpu": "cpu", "ram_gb": 64},
        "model": {"file": "Qwen3.8-27B-Q4_K_M.gguf", "size_bytes": 1},
        "runtime": {"engine": "openjet", "context_tokens": 131072, "variant": "OpenJet engine, 1.5x faster"},
        "results": results,
    }
    record.update(overrides)
    return record


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeServer.requests = []

    def run_workload(self, context: int) -> dict:
        return asyncio.run(community_bench.run_workload(self.url, "m", context, log=lambda _m: None))

    def test_runs_the_fixed_workload_and_passes_validation(self):
        results = self.run_workload(131072)
        self.assertEqual(results["reasoning"]["tok_s"], 100.0)
        self.assertEqual(len(results["reasoning"]["runs"]), 4)
        self.assertEqual([d["depth"] for d in results["depth"]], list(community_bench.DEPTHS))
        self.assertTrue(all(d["recall"] for d in results["depth"]))
        self.assertEqual(results["depth"][0]["prefill_tok_s"], 500.0)
        self.assertEqual(bench_table.validate(_record(results)), [])
        reasoning = [b for b in FakeServer.requests if b["chat_template_kwargs"]["enable_thinking"]][0]
        self.assertEqual(reasoning["temperature"], 1.0)
        self.assertEqual(reasoning["top_k"], 20)
        self.assertEqual(reasoning["seed"], community_bench.SEED)

    def test_depths_that_do_not_fit_the_context_are_skipped(self):
        results = self.run_workload(40000)
        self.assertEqual([d["depth"] for d in results["depth"]], [4096, 32768])

    def test_recall_question_reuses_the_cached_source(self):
        self.run_workload(8192)
        depth, recall = FakeServer.requests[-2:]
        self.assertFalse(depth["cache_prompt"])
        self.assertTrue(recall["cache_prompt"])
        source = depth["messages"][0]["content"].rsplit("\n", 1)[0]
        self.assertTrue(recall["messages"][0]["content"].startswith(source))

    def test_depth_text_hides_one_needle_halfway(self):
        text = community_bench.depth_text(4096, "salt", "NEEDLE")
        self.assertTrue(text.startswith("# benchmark source salt\n"))
        self.assertEqual(text.count("NEEDLE"), 1)
        self.assertAlmostEqual(text.index("NEEDLE") / len(text), 0.5, delta=0.05)
        self.assertAlmostEqual(len(text) / community_bench._CHARS_PER_TOKEN, 4096, delta=100)
        # the code is the same on every machine; only the salt line differs
        self.assertEqual(text.split("\n", 1)[1], community_bench.depth_text(4096, "other", "NEEDLE").split("\n", 1)[1])

    def test_gpu_slug(self):
        self.assertEqual(community_bench.gpu_slug("NVIDIA GeForce RTX 3090"), "rtx-3090")
        self.assertEqual(community_bench.gpu_slug("Apple M3 Max"), "apple-m3-max")


class TableTests(unittest.TestCase):
    def results(self, reasoning_tok_s: float = 80.0) -> dict:
        runs = [{"prompt": i, "prompt_tokens": 40, "prompt_ms": 20.0, "tokens": 512, "ms": 512 / reasoning_tok_s * 1e3}
                for i in range(4)]
        greedy = [{"prompt": i, "prompt_tokens": 30, "prompt_ms": 15.0, "tokens": 256, "ms": 2560.0} for i in range(4)]
        return {
            "reasoning": community_bench._decode_summary(runs),
            "greedy": community_bench._decode_summary(greedy),
            "depth": [{"depth": 32768, "prompt_tokens": 32000, "prompt_ms": 20000.0, "tokens": 256, "ms": 4000.0,
                       "prefill_tok_s": 1600.0, "decode_tok_s": 64.0, "recall": True}],
        }

    def test_tampered_speed_is_rejected(self):
        results = self.results()
        results["reasoning"]["tok_s"] = 200.0
        self.assertIn("results.reasoning.tok_s does not match its runs", bench_table.validate(_record(results)))

    def test_bad_handle_and_schema_are_rejected(self):
        self.assertTrue(bench_table.validate(_record(self.results(), contributor="<script>")))
        self.assertTrue(bench_table.validate(_record(self.results(), schema="other/1")))

    def test_runs_are_grouped_with_the_median_and_credited(self):
        records = [_record(self.results(80.0)), _record(self.results(90.0), contributor="b"),
                   _record(self.results(100.0), contributor=None)]
        records.append(_record(self.results(50.0), runtime={"engine": "llama.cpp", "device": "cuda"}))
        rows = bench_table.rows(records)
        self.assertEqual(len(rows), 2)
        gpu, model, engine, os_family, reasoning, _greedy, prompt32, decode32, recall, runs, credit = rows[0]
        self.assertEqual((gpu, engine, os_family), ("RTX 4090 24 GB", "OpenJet", "Linux"))
        self.assertEqual((reasoning, prompt32, decode32, recall, runs), ("90.0", "1600", "64.0", "3/3", "3"))
        self.assertEqual(credit, "[b](https://github.com/b), [octo-cat](https://github.com/octo-cat)")
        self.assertEqual(rows[1][2], "llama.cpp (cuda)")

    def test_readme_table_is_rewritten_between_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(f"intro\n{bench_table.START}\nold\n{bench_table.END}\nafter\n")
            table = bench_table.render([_record(self.results())])
            self.assertTrue(bench_table.update_readme(table, readme))
            text = readme.read_text()
            self.assertIn("| RTX 4090 24 GB | Qwen3.8-27B-Q4_K_M | OpenJet | Linux | 80.0 |", text)
            self.assertNotIn("old", text)
            self.assertTrue(text.endswith(f"{bench_table.END}\nafter\n"))
            self.assertFalse(bench_table.update_readme(table, readme))

    def test_files_in_the_repo_are_valid(self):
        _records, errors = bench_table.load()
        self.assertEqual(errors, [])


class CliTests(unittest.TestCase):
    def test_bench_is_its_own_command(self):
        with mock.patch("src.community_bench.run_bench") as run:
            cli.main(["bench", "--github", "octo-cat", "--notes", "300 W"])
        run.assert_called_once_with(output=None, contributor="octo-cat", notes="300 W")

    def test_benchmark_still_runs_llama_bench(self):
        with mock.patch("src.benchmark.run_benchmark") as run:
            cli.main(["benchmark", "-r", "1"])
        self.assertEqual(run.call_args.kwargs["repetitions"], 1)


if __name__ == "__main__":
    unittest.main()
