import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock
import urllib.error
import uuid


SPEC = importlib.util.spec_from_file_location(
    "think_bench", Path(__file__).resolve().parents[1] / "megakernel" / "bench" / "think_bench.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
LLAMA_GPU = "GPU-00000000-0000-0000-0000-000000000002"
MK_GPU = "GPU-00000000-0000-0000-0000-000000000003"


class ThinkBenchTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(__file__).parent / (".think-bench-test-" + uuid.uuid4().hex)
        self.work.mkdir()
        self.addCleanup(shutil.rmtree, self.work)
        patch = mock.patch.object(bench, "TMP", str(self.work))
        patch.start()
        self.addCleanup(patch.stop)
        sockets = mock.patch.object(bench.socket, "socket")
        self.socket = sockets.start()
        self.addCleanup(sockets.stop)

    def test_independent_gpu_environments_and_unpinned_defaults(self):
        inherited = {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST", "PATH": "/bin"}
        with mock.patch.dict(os.environ, inherited, clear=True):
            llama = bench.gpu_env("2")
            mk = bench.gpu_env(MK_GPU)
            self.assertEqual(llama["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(mk["CUDA_VISIBLE_DEVICES"], MK_GPU)
            self.assertEqual(llama["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(mk["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(bench.gpu_env(None), inherited)
            self.assertEqual(dict(os.environ), inherited)
        for value in ("2,3", "-1", "", "GPU-invalid"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.gpu_identifier(value)

    def test_megakernel_paired_seeds_sampling_and_raw_timing(self):
        stderr = "[mk] decode 8 tok in 2.000 s = 4.00 tok/s | cycles 4, accepted 3/8 drafts (37.5%), 2.00 tok/cycle\n"
        with mock.patch.object(bench.subprocess, "run", return_value=mock.Mock(stderr=stderr)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            samples = bench.run_mk(["prompt"], 2, 9, 3, model="same.gguf", megakernel="custom-mk",
                                   gpu=MK_GPU, kv="q8", context=8192)
        for seed, call in enumerate(run.call_args_list, 1000):
            cmd = call.args[0]
            self.assertEqual(cmd[:3], ["custom-mk", "-m", "same.gguf"])
            self.assertEqual(cmd[cmd.index("--seed") + 1], str(seed))
            self.assertEqual(cmd[cmd.index("-n") + 1], "9")
            self.assertEqual(cmd[cmd.index("--kv") + 1], "q8")
            self.assertEqual(cmd[cmd.index("-c") + 1], "8192")
            for flag, value in (("--temp", "1.0"), ("--top-p", "0.95"), ("--top-k", "20"), ("--min-p", "0.0")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertEqual(Path(cmd[cmd.index("-f") + 1]).read_text(), bench.render("prompt"))
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], MK_GPU)
        self.assertEqual(samples[(0, 0)]["n"], 8)
        self.assertEqual(samples[(0, 0)]["generated_n"], 9)
        self.assertEqual(samples[(0, 0)]["seconds"], 2.0)
        self.assertEqual(samples[(0, 0)]["stderr"], stderr)

    def test_megakernel_failure_is_not_a_timing_sample(self):
        failure = subprocess.CalledProcessError(1, ["custom-mk"], stderr="decode 8 tok in 2.000 s = 4.00 tok/s")
        with mock.patch.object(bench.subprocess, "run", side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError):
                bench.run_mk(["prompt"], 1, 9, 3)

    def test_llama_uses_same_prompt_sampling_seeds_and_separate_gpu(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        health = mock.MagicMock()
        health.__enter__.return_value.status = 200
        timings = dict(predicted_n=8, predicted_per_second=4.0, predicted_ms=2000)
        with mock.patch.object(bench.subprocess, "Popen", return_value=proc) as popen, \
                mock.patch.object(bench.urllib.request, "urlopen", return_value=health), \
                mock.patch.object(bench, "post", return_value={"timings": timings}) as post, \
                contextlib.redirect_stdout(io.StringIO()):
            samples = bench.run_llama(["prompt"], 2, 9, model="same.gguf", llama_server="custom-llama",
                                      gpu=LLAMA_GPU, context=8192)
        self.assertEqual(popen.call_args.args[0][:3], ["custom-llama", "-m", "same.gguf"])
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-c") + 1], "8192")
        self.assertEqual(cmd[cmd.index("-ctk") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("-ctv") + 1], "q8_0")
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], LLAMA_GPU)
        for seed, call in enumerate(post.call_args_list, 1000):
            payload = call.args[1]
            self.assertEqual(payload["prompt"], bench.render("prompt"))
            self.assertEqual(payload["seed"], seed)
            self.assertEqual(payload["n_predict"], 9)
            self.assertFalse(payload["cache_prompt"])
            self.assertEqual({k: payload[k] for k in bench.SAMPLING}, bench.SAMPLING)
        self.assertEqual(samples[(0, 0)]["timings"], timings)
        self.assertEqual(samples[(0, 0)]["seconds"], 2.0)
        proc.terminate.assert_called_once()

    def test_llama_startup_exit_and_health_timeout_fail_explicitly(self):
        proc = mock.Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with mock.patch.object(bench.subprocess, "Popen", return_value=proc), \
                mock.patch.object(bench, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "status 1"):
                bench.run_llama(["prompt"], 2, 9)
            post.assert_not_called()
        proc.poll.return_value = None
        with mock.patch.object(bench.subprocess, "Popen", return_value=proc), \
                mock.patch.object(bench.urllib.request, "urlopen", side_effect=urllib.error.URLError("not ready")), \
                mock.patch.object(bench.time, "sleep"), mock.patch.object(bench, "post") as post:
            with self.assertRaisesRegex(TimeoutError, "did not become healthy"):
                bench.run_llama(["prompt"], 2, 9)
            post.assert_not_called()

    def test_llama_refuses_occupied_port_before_launch_or_health_request(self):
        self.socket.return_value.__enter__.return_value.bind.side_effect = OSError("address already in use")
        with mock.patch.object(bench.subprocess, "Popen") as popen, \
                mock.patch.object(bench.urllib.request, "urlopen") as health, \
                mock.patch.object(bench, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "port 8094 is unavailable"):
                bench.run_llama(["prompt"], 2, 256)
            popen.assert_not_called()
            health.assert_not_called()
            post.assert_not_called()

    def test_cli_paths_and_output_keep_every_seed(self):
        output = self.work / "results.json"
        samples = {(i, s): dict(n=8, tps=4.0 + s, seconds=2.0, acc=(1, 2))
                   for i in range(5) for s in range(2)}
        argv = ["think_bench", "--model", "same.gguf", "--llama-server", "custom-llama",
                "--megakernel", "custom-mk", "--llama-gpu", LLAMA_GPU,
                "--megakernel-gpu", MK_GPU, "-n", "256", "--kv", "q8", "--context", "8192",
                "--output", str(output)]
        with mock.patch("sys.argv", argv), mock.patch.object(bench, "file_prompt", return_value="file prompt"), \
                mock.patch.object(bench, "run_llama", return_value=samples) as llama, \
                mock.patch.object(bench, "run_mk", return_value=samples) as mk, \
                contextlib.redirect_stdout(io.StringIO()):
            bench.main()
        self.assertEqual(llama.call_args.kwargs,
                         dict(model="same.gguf", llama_server="custom-llama", gpu=LLAMA_GPU, context=8192))
        self.assertEqual(mk.call_count, 2)
        self.assertEqual(mk.call_args.kwargs,
                         dict(model="same.gguf", megakernel="custom-mk", gpu=MK_GPU, kv="q8", context=8192))
        raw = json.loads(output.read_text())
        self.assertEqual(raw["settings"]["seeds_used"], [1000, 1001])
        self.assertEqual(raw["settings"]["n"], 256)
        self.assertEqual(raw["settings"]["prompt_count"], 5)
        self.assertEqual(raw["settings"]["context"], 8192)
        self.assertEqual(raw["settings"]["llama_kv"], "q8_0")
        self.assertEqual(raw["settings"]["megakernel_kv"], "q8")
        self.assertEqual(raw["settings"]["llama_draft_max"], 2)
        self.assertEqual(raw["settings"]["megakernel_drafts"], [3, 4])
        self.assertEqual(raw["rendered_prompts"], [bench.render(p) for p in raw["prompts"]])
        for results in raw["samples"].values():
            self.assertEqual(len(results), 10)
            self.assertEqual({r["seed"] for r in results}, {1000, 1001})
        for args in (["--seeds", "0"], ["-n", "-1"], ["--context", "0"], ["--kv", "q4"]):
            with mock.patch("sys.argv", ["think_bench"] + args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.main()

    def test_cli_preserves_legacy_defaults(self):
        samples = {(i, s): dict(n=8, tps=4.0, seconds=2.0, acc=(1, 2))
                   for i in range(5) for s in range(2)}
        with mock.patch("sys.argv", ["think_bench"]), \
                mock.patch.object(bench, "file_prompt", return_value="file prompt"), \
                mock.patch.object(bench, "run_llama", return_value=samples) as llama, \
                mock.patch.object(bench, "run_mk", return_value=samples) as mk, \
                contextlib.redirect_stdout(io.StringIO()):
            bench.main()
        self.assertEqual(llama.call_args.args[1:], (2, 1024))
        self.assertEqual(llama.call_args.kwargs,
                         dict(model=bench.MODEL, llama_server=bench.LLAMA_SERVER, gpu=None, context=16384))
        self.assertEqual([call.args[3] for call in mk.call_args_list], [3, 4])
        self.assertEqual(mk.call_args.kwargs,
                         dict(model=bench.MODEL, megakernel=bench.MK, gpu=None, kv="auto", context=16384))


if __name__ == "__main__":
    unittest.main()
