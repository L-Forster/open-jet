import asyncio
import json
import os
import struct
import tarfile
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
        self.model = str(Path(self.tmp.name) / Path(MODEL).name)
        types = {"token_embd.weight": 12, "output.weight": 14}
        with open(self.model, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, len(types), 0))
            for name, kind in types.items():
                name = name.encode()
                f.write(struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 256, kind, 0))
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
            mock.patch.object(mb, "install_prebuilt_megakernel", mock.AsyncMock(return_value=False)),
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
            _run({"llama_model": self.model}, _Log())
        args = run.call_args.args
        self.assertEqual(args[:4], ("make", "-B", "-C", str(self.root)))
        self.assertIn("ARCH=sm_89", args)
        self.assertIn(f"LLAMA={self.llama}", args)
        self.assertEqual(json.loads((self.root / ".openjet-build.json").read_text()), {"arch": "sm_89"})

    def test_skips_other_models_gpus_and_the_llama_backend(self) -> None:
        with self._fake_make() as run:
            _run({"llama_model": "/models/Qwen3.6-27B-UD-IQ3_XXS.gguf"}, _Log())
            _run({"llama_model": self.model, "llama_backend": "llama"}, _Log())
            with mock.patch.object(mb, "_megakernel_gpu_info", return_value=None):
                _run({"llama_model": self.model}, _Log())
        run.assert_not_called()

    def test_builds_for_a_pinned_rtx_8000(self) -> None:
        with self._fake_make() as run, \
                mock.patch.object(mb, "_megakernel_gpu_info", return_value=("3", "7.5")):
            _run({"llama_model": self.model}, _Log())
        self.assertIn("ARCH=sm_75", run.call_args.args)
        self.assertEqual(json.loads((self.root / ".openjet-build.json").read_text()), {"arch": "sm_75"})

    def test_missing_nvcc_or_llama_cpp_is_reported_not_raised(self) -> None:
        log = _Log()
        with self._fake_make() as run, mock.patch.object(mb, "_nvcc_path", return_value=None):
            _run({"llama_model": self.model}, log)
        (self.llama / "build" / "bin" / "libllama.so").unlink()
        with self._fake_make() as run2:
            _run({"llama_model": self.model}, log)
        run.assert_not_called()
        run2.assert_not_called()
        self.assertIn("CUDA toolkit", log.lines[0])
        self.assertIn("llama.cpp", log.lines[1])

    def test_failed_build_falls_back_without_raising(self) -> None:
        log = _Log()
        setup = {"llama_model": self.model}
        with self._fake_make(rc=2):
            self.assertEqual(_run(setup, log), setup)
        self.assertTrue(any("build failed" in line for line in log.lines))
        self.assertFalse((self.root / ".openjet-build.json").exists())

    def test_current_build_is_not_rebuilt_but_a_new_gpu_arch_is(self) -> None:
        with self._fake_make() as run, mock.patch.object(mb, "_megakernel_dir", return_value=self.root):
            _run({"llama_model": self.model}, _Log())
            _run({"llama_model": self.model}, _Log())
            self.assertEqual(run.call_count, 1)
            with mock.patch.object(mb, "_megakernel_gpu_info", return_value=("0", "12.0")):
                _run({"llama_model": self.model}, _Log())
            self.assertEqual(run.call_count, 2)
            self.assertIn("ARCH=sm_120", run.call_args.args)

    def test_turing_compatibility_header_change_invalidates_the_build(self) -> None:
        with self._fake_make(), mock.patch.object(mb, "_megakernel_dir", return_value=self.root):
            _run({"llama_model": self.model}, _Log())
            header = self.root / "cuda_compat.cuh"
            header.write_text("")
            changed = (self.root / "libqwen38mk.so").stat().st_mtime + 1
            os.utime(header, (changed, changed))
            self.assertFalse(mb._build_is_current(self.root, "sm_89"))


class PrebuiltMegakernelTests(unittest.TestCase):
    """an installed wheel downloads the engine for its GPU instead of compiling it"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.model = str(base / Path(MODEL).name)
        types = {"token_embd.weight": 12, "output.weight": 14}
        with open(self.model, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, len(types), 0))
            for name, kind in types.items():
                name = name.encode()
                f.write(struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 256, kind, 0))
        self.root = base / "site-packages" / "megakernel"
        (self.root / "server").mkdir(parents=True)
        for name in ("Makefile", "draft_vocab_32k.bin", "server/mk_server.py"):
            (self.root / name).write_text("")
        self.archive = base / "engine.tar.gz"
        staged = base / "staged"
        staged.mkdir()
        for name in ("libqwen38mk.so", "libqwen38mk_mixed.so", "libllama.so.0.0.1"):
            (staged / name).write_text(name)
        os.symlink("libllama.so.0.0.1", staged / "libllama.so.0")
        with tarfile.open(self.archive, "w:gz") as tf:
            for entry in staged.iterdir():
                tf.add(entry, arcname=entry.name)
        self.urls: list[str] = []
        self.patches = [
            mock.patch.object(mb.sys, "platform", "linux"),
            mock.patch.object(mb.platform, "machine", return_value="x86_64"),
            mock.patch.object(mb, "_openjet_version", return_value="0.6.0"),
            mock.patch.object(mb, "_megakernel_root", return_value=self.root),
            mock.patch("src.llama_server._megakernel_root", return_value=self.root),
            mock.patch.object(mb, "_megakernel_gpu_info", return_value=("0", "8.6")),
            mock.patch.object(mb, "_nvcc_path", return_value=None),
            mock.patch.object(mb, "_download_to_path", side_effect=self._download),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    async def _download(self, url, target, **_kwargs):
        self.urls.append(url)
        if self.archive is None:
            raise RuntimeError("Client error '404 Not Found'")
        target.write_bytes(self.archive.read_bytes())

    def test_downloads_this_releases_engine_without_the_cuda_toolkit(self) -> None:
        log = _Log()
        with mock.patch.object(mb, "_run_exec") as run:
            _run({"llama_model": self.model}, log)
        run.assert_not_called()
        self.assertEqual(self.urls, ["https://github.com/L-Forster/open-jet/releases/download/v0.6.0/"
                                     "openjet-engine-linux-x64-sm86.tar.gz"])
        self.assertTrue((self.root / "libqwen38mk.so").is_file())
        self.assertEqual(os.readlink(self.root / "libllama.so.0"), "libllama.so.0.0.1")
        for tag in (".openjet-build.json", ".openjet-build-mixed.json"):
            self.assertEqual(json.loads((self.root / tag).read_text()), {"arch": "sm_86", "prebuilt": "0.6.0"})
        self.assertIn("OpenJet engine installed", log.lines[-1])
        # current until the next release, whatever the source mtimes
        self.assertTrue(mb._build_is_current(self.root, "sm_86", self.model))
        with mock.patch.object(mb, "_openjet_version", return_value="0.6.1"):
            self.assertFalse(mb._build_is_current(self.root, "sm_86", self.model))

    def test_a_release_without_the_engine_falls_back_to_compiling(self) -> None:
        self.archive = None
        log = _Log()
        _run({"llama_model": self.model}, log)
        self.assertEqual(len(self.urls), 1)
        self.assertIn("No prebuilt engine for sm_86 in v0.6.0", log.lines[0])
        self.assertIn("CUDA toolkit not found", log.lines[-1])

    def test_a_source_checkout_compiles_its_own_engine(self) -> None:
        (self.root.parent / ".git").mkdir()
        _run({"llama_model": self.model}, _Log())
        self.assertEqual(self.urls, [])

    def test_windows_uses_the_prebuilt_dll_and_never_compiles(self) -> None:
        self.archive = None
        log = _Log()
        with mock.patch.object(mb.sys, "platform", "win32"), mock.patch.object(mb, "_run_exec") as run:
            _run({"llama_model": self.model}, log)
        run.assert_not_called()
        self.assertTrue(self.urls[0].endswith("/openjet-engine-windows-x64-sm86.zip"))
        self.assertIn("no prebuilt engine for sm_86", log.lines[-1])


class SourceListTests(unittest.TestCase):
    def test_every_engine_dependency_invalidates_the_build(self) -> None:
        makefile = (Path(mb.__file__).resolve().parents[1] / "megakernel" / "Makefile").read_text()
        mixed = {"qwen38_mk.cu": "qwen38_mixed.cu", "prefill_gemm.cuh": "prefill_gemm_mixed.cuh"}
        for target, sources in (("qwen38_mk.o:", set(mb._SOURCES)), ("qwen38_mixed.o:", {mixed.get(s, s) for s in mb._SOURCES})):
            line = next(l for l in makefile.splitlines() if l.startswith(target))
            self.assertEqual(sorted(set(line.split(":", 1)[1].split()) - sources), [], target)


if __name__ == "__main__":
    unittest.main()
