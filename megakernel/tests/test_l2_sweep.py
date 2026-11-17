"""Tests of scripts/l2_sweep.py without a GPU: a stub program stands in for qwen38_mk."""
import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "l2_sweep.py")
spec = importlib.util.spec_from_file_location("l2_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

# qwen38_mk --batch: the MK_L2_TUNE line, then per job a rate that depends on MK_L2_DN (STUB_FAST) and fixed token ids;
# STUB_NO_TUNE: a build without the settings, STUB_BREAK: another MK_L2_DN value changes the ids
STUB = r'''
import os, shlex, sys
args = sys.argv[1:]
batch = args[args.index("--batch") + 1]
names = ["MK_L2_DN", "MK_L2_ATTN", "MK_L2_ATTN_CTX", "MK_L2_TOP", "MK_L2_CK", "MK_L2_MTP", "MK_L2_MTP_R", "MK_L2_MTP_KEEP"]
if not os.environ.get("STUB_NO_TUNE"):
    print("[mk] MK_L2_TUNE: " + " ".join(n + "=" + os.environ.get(n, "7") for n in names), file=sys.stderr)
dn = os.environ.get("MK_L2_DN", "")
fast = dn == os.environ.get("STUB_FAST")
for line in open(batch):
    v = shlex.split(line)
    pfx = v[v.index("-o") + 1]
    rate = 110.0 if fast else 100.0
    ids = "1 2 3" + (" 4" if dn and dn == os.environ.get("STUB_BREAK") else "")
    open(pfx + ".out", "w").write("IN_IDS: 9\nIDS: " + ids + "\n")
    open(pfx + ".err", "w").write("\n[mk] prompt 40000 tok in 20.000 s (2000.0 tok/s)\n"
                                  "[mk] decode 511 tok in 5.000 s = %.2f tok/s | cycles 100, accepted 400/500 drafts (80.0%%), 5.11 tok/cycle\n" % rate)
'''


class Units(unittest.TestCase):
    def test_a_typo_is_refused(self):
        self.assertEqual(sweep.parse_setting("MK_L2_DN=8192,MK_L2_TOP=2"), {"MK_L2_DN": "8192", "MK_L2_TOP": "2"})
        for bad in ("MK_L2_DNN=1", "MK_L2_DN", "MK_L2_DN=x", "MK_L2_DN=-1"):
            with self.assertRaises(ValueError):
                sweep.parse_setting(bad)

    def test_grids_parse(self):
        for s in sweep.GRID_QUICK + sweep.GRID_FULL:
            sweep.parse_setting(s)

    def test_schedule_alternates_with_the_defaults(self):
        order = sweep.schedule(["a", "b", "c"], 2)
        self.assertEqual(order, [(0, None), (0, 0), (0, 1), (0, None), (0, 2), (0, None),
                                 (1, None), (1, 2), (1, 1), (1, None), (1, 0), (1, None)])

    def test_compared_with_the_defaults_around_it(self):
        runs = [{"setting": None, "short": 100.0, "ids": []}, {"setting": 0, "short": 110.0, "ids": []},
                {"setting": None, "short": 120.0, "ids": []}]
        self.assertAlmostEqual(sweep.base_around(runs, 1, "short"), 110.0)
        self.assertAlmostEqual(sweep.noise_of(runs, "short"), 0.2)
        rows, noise = sweep.analyse(runs, ["MK_L2_DN=1"])
        self.assertAlmostEqual(rows[0]["metrics"]["short"]["delta"], 0.0)
        self.assertTrue(rows[0]["ids_same"])
        self.assertEqual(sweep.fmt_delta(rows[0]["metrics"]["short"], noise["short"]), "~+0.0%")


class BuildInfoBom(unittest.TestCase):
    def test_the_bom_of_a_package_does_not_reach_the_output(self):
        # scripts/package_release.ps1 writes BUILD_INFO.txt as PowerShell's utf8, with a BOM: exactness.py printed it last
        # and died on a cp1252 console
        spec2 = importlib.util.spec_from_file_location("exactness", os.path.join(ROOT, "scripts", "exactness.py"))
        ex = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(ex)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "BUILD_INFO.txt"), "wb") as f:
                f.write(b"\xef\xbb\xbfqwen38-megakernel v1, Windows x64\r\ncommit x\r\n")
            ex.HERE = d
            line = ex.build_info()
        self.assertEqual(line, "qwen38-megakernel v1, Windows x64")
        line.encode("cp1252")


class ServerStop(unittest.TestCase):
    def test_stop_waits_for_the_port(self):
        # a Windows venv's python.exe is a launcher: the process we wait for is gone while the server still holds the port
        spec2 = importlib.util.spec_from_file_location("exactness", os.path.join(ROOT, "scripts", "exactness.py"))
        ex = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(ex)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        threading.Timer(1.0, listener.close).start()
        srv = ex.Server.__new__(ex.Server)
        srv.chk = types.SimpleNamespace(a=types.SimpleNamespace(port=port))
        srv.p = types.SimpleNamespace(poll=lambda: 0)
        srv.logf = tempfile.TemporaryFile()
        self.assertTrue(ex.port_open(port))
        srv.stop()
        self.assertFalse(ex.port_open(port))


@unittest.skipIf(os.name == "nt", "the stub is a POSIX script")
class Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exe = os.path.join(self.tmp.name, "qwen38_mk")
        with open(self.exe, "w") as f:
            f.write(f"#!{sys.executable}\n" + STUB)
        os.chmod(self.exe, os.stat(self.exe).st_mode | stat.S_IXUSR)
        self.model = os.path.join(self.tmp.name, "m.gguf")
        open(self.model, "w").close()
        self.out = os.path.join(self.tmp.name, "out")

    def run_sweep(self, *args, **env):
        return subprocess.run([sys.executable, SCRIPT, "-m", self.model, "--exe", self.exe, "--out", self.out, "--rounds", "1", *args],
                              capture_output=True, text=True, env=dict(os.environ, **env), timeout=120)

    def test_a_faster_setting_shows(self):
        r = self.run_sweep("--set", "MK_L2_DN=8192", "--set", "MK_L2_TOP=2", STUB_FAST="8192")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("+10.0%", r.stdout)
        self.assertIn("~+0.0%", r.stdout)
        with open(os.path.join(self.out, "results.json")) as f:
            self.assertEqual(json.load(f)["grid"], ["MK_L2_DN=8192", "MK_L2_TOP=2"])

    def test_changed_tokens_are_reported(self):
        r = self.run_sweep("--set", "MK_L2_DN=8192", STUB_BREAK="8192")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("OUTPUT CHANGED", r.stdout)

    def test_a_build_without_the_settings_stops(self):
        r = self.run_sweep("--set", "MK_L2_DN=8192", STUB_NO_TUNE="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("-DMK_L2_TUNE=1", r.stderr)

    def test_a_typo_stops_before_any_run(self):
        r = self.run_sweep("--set", "MK_L2_TOPP=2")
        self.assertEqual(r.returncode, 2)
        self.assertFalse(os.path.exists(self.out))

    def test_no_long_prompt(self):
        r = self.run_sweep("--no-long", "--set", "MK_L2_DN=8192")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("after the long prompt", r.stdout)


if __name__ == "__main__":
    unittest.main()
