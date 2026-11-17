"""The server's module layout: what splitting it across files can break while every other test still passes."""
import subprocess
import sys
import tempfile
import unittest

from server_modules import SERVER, load

SET_BY_MAIN = ("MAX_BODY_MB", "MAX_INFLIGHT_MB", "MAX_CONNECTIONS", "REQUEST_TIMEOUT_S", "MAX_IMAGES", "MAX_VIDEO_FRAMES",
               "IMAGE_MAX_PIXELS", "REASONING_MAX_BYTES")


class ServerLayoutTests(unittest.TestCase):
    def test_limits_set_by_main_are_only_in_mk_limits(self):
        # a by-name import would keep the default in production; a patch replaces every copy, so no other test would see it
        mk = load()
        for name in SET_BY_MAIN:
            self.assertEqual([m.__name__ for m in mk._modules if name in vars(m)], ["mk_limits"], name)

    def test_the_loader_gets_every_server_module(self):
        self.assertEqual({m.__name__ for m in load()._modules}, {p.stem for p in SERVER.glob("*.py")})

    def test_runs_as_a_script_from_another_directory(self):
        # (the tests import the modules with server/ put on sys.path by hand; the launchers run the file)
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run([sys.executable, "-B", str(SERVER / "mk_server.py"), "--help"], cwd=cwd,
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--model", r.stdout)
