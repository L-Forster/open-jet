import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import megakernel_build as mb

MODEL = "/models/Qwen3.8-27B-Q4_K_M.gguf"


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)


def _run(setup: dict, log: _Log) -> dict:
    return asyncio.run(mb.ensure_megakernel(setup, log=log, set_status=lambda _s: None, clear_status=lambda: None))


class EnsureMegakernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "megakernel"
        self.root.mkdir()
        (self.root / "Makefile").write_text("")
        self.llama = Path(self.tmp.name) / "llama.cpp"
        (self.llama / "include").mkdir(parents=True)
        (self.llama / "build" / "bin").mkdir(parents=True)
        (self.llama / "include" / "llama.h").write_text("")
        (self.llama / "build" / "bin" / "libllama.so").write_text("")
        self.patches = [
            mock.patch.object(mb.sys, "platform", "linux"),
            mock.patch.object(mb, "_megakernel_root", return_value=self.root),
            mock.patch.object(mb, "_megakernel_gpu_info", return_value=("0", "8.9")),
            mock.patch.object(mb, "_nvcc_path", return_value="/usr/local/cuda/bin/nvcc"),
            mock.patch.object(mb, "LLAMA_CPP_DIR", self.llama),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def _fake_make(self, rc: int = 0):
        async def run(*args, cwd=None):
            if rc == 0:
                (self.root / "libqwen38mk.so").write_text("")
            return rc, "", "nvcc fatal : boom"
        return mock.patch.object(mb, "_run_exec", side_effect=run)

    def test_builds_for_the_detected_gpu_architecture(self) -> None:
        with self._fake_make() as run:
            _run({"llama_model": MODEL}, _Log())
        args = run.call_args.args
        self.assertEqual(args[:4], ("make", "-B", "-C", str(self.root)))
        self.assertIn("ARCH=sm_89", args)
        self.assertIn(f"LLAMA={self.llama}", args)
        self.assertEqual(json.loads((self.root / ".openjet-build.json").read_text()), {"arch": "sm_89"})

    def test_skips_other_models_gpus_and_the_llama_backend(self) -> None:
        with self._fake_make() as run:
            _run({"llama_model": "/models/Qwen3.6-27B-UD-IQ3_XXS.gguf"}, _Log())
            _run({"llama_model": MODEL, "llama_backend": "llama"}, _Log())
            with mock.patch.object(mb, "_megakernel_gpu_info", return_value=None):
                _run({"llama_model": MODEL}, _Log())
        run.assert_not_called()

    def test_missing_nvcc_or_llama_cpp_is_reported_not_raised(self) -> None:
        log = _Log()
        with self._fake_make() as run, mock.patch.object(mb, "_nvcc_path", return_value=None):
            _run({"llama_model": MODEL}, log)
        (self.llama / "build" / "bin" / "libllama.so").unlink()
        with self._fake_make() as run2:
            _run({"llama_model": MODEL}, log)
        run.assert_not_called()
        run2.assert_not_called()
        self.assertIn("CUDA toolkit", log.lines[0])
        self.assertIn("llama.cpp", log.lines[1])

    def test_failed_build_falls_back_without_raising(self) -> None:
        log = _Log()
        setup = {"llama_model": MODEL}
        with self._fake_make(rc=2):
            self.assertEqual(_run(setup, log), setup)
        self.assertTrue(any("build failed" in line for line in log.lines))
        self.assertFalse((self.root / ".openjet-build.json").exists())

    def test_current_build_is_not_rebuilt_but_a_new_gpu_arch_is(self) -> None:
        with self._fake_make() as run, mock.patch.object(mb, "_megakernel_dir", return_value=self.root):
            _run({"llama_model": MODEL}, _Log())
            _run({"llama_model": MODEL}, _Log())
            self.assertEqual(run.call_count, 1)
            with mock.patch.object(mb, "_megakernel_gpu_info", return_value=("0", "12.0")):
                _run({"llama_model": MODEL}, _Log())
            self.assertEqual(run.call_count, 2)
            self.assertIn("ARCH=sm_120", run.call_args.args)


if __name__ == "__main__":
    unittest.main()
