from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.self_update import available_release_update, update_from_latest_release


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class SelfUpdateTests(unittest.TestCase):
    def test_available_release_update_skips_unknown_installed_version(self) -> None:
        with patch("src.self_update.urlopen") as urlopen_mock:
            release = available_release_update(current_version="unknown")

        self.assertIsNone(release)
        urlopen_mock.assert_not_called()

    def test_update_from_latest_release_merges_new_release_config_with_user_settings(self) -> None:
        release_payload = json.dumps(
            {
                "tag_name": "v0.4.0",
                "tarball_url": "https://example.invalid/open-jet-v0.4.0.tar.gz",
            }
        ).encode("utf-8")
        release_response = _FakeResponse(release_payload)
        archive_response = _FakeResponse(b"tarball-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("runtime: llama_cpp\n")

            def fake_urlopen(request, timeout=None):
                del timeout
                url = request.full_url
                if url.endswith("/releases/latest"):
                    return release_response
                if url == "https://example.invalid/open-jet-v0.4.0.tar.gz":
                    return archive_response
                raise AssertionError(url)

            with patch("src.self_update.urlopen", side_effect=fake_urlopen), patch(
                "src.self_update.CONFIG_PATH",
                config_path,
            ), patch(
                "src.self_update.subprocess.run",
                side_effect=lambda args, check: config_path.write_text(
                    "runtime: openai_compatible\ntelemetry:\n  enabled: true\nnew_feature:\n  enabled: true\n"
                ),
            ) as run_mock:
                previous_cwd = Path.cwd()
                os.chdir(root)
                try:
                    message = update_from_latest_release(current_version="0.3.0")
                    merged_config = config_path.read_text()
                finally:
                    os.chdir(previous_cwd)

            self.assertEqual(message, "Updated open-jet from 0.3.0 to 0.4.0.")
            self.assertIn("runtime: llama_cpp", merged_config)
            self.assertIn("new_feature:", merged_config)
            self.assertIn("telemetry:", merged_config)
            self.assertTrue((root / "config.yaml.bak.pre-update-0.4.0").is_file())
            run_mock.assert_called_once()
            self.assertEqual(run_mock.call_args.args[0][:4], [sys.executable, "-m", "pip", "install"])

    def test_update_from_latest_release_skips_install_when_already_current(self) -> None:
        release_payload = json.dumps(
            {
                "tag_name": "v0.3.0",
                "tarball_url": "https://example.invalid/open-jet-v0.3.0.tar.gz",
            }
        ).encode("utf-8")

        with patch("src.self_update.urlopen", return_value=_FakeResponse(release_payload)), patch(
            "src.self_update.subprocess.run"
        ) as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(message, "open-jet 0.3.0 is already up to date.")
        run_mock.assert_not_called()

    def test_update_from_latest_release_raises_clear_error_when_install_fails(self) -> None:
        release_payload = json.dumps(
            {
                "tag_name": "v0.4.0",
                "tarball_url": "https://example.invalid/open-jet-v0.4.0.tar.gz",
            }
        ).encode("utf-8")
        release_response = _FakeResponse(release_payload)
        archive_response = _FakeResponse(b"tarball-bytes")

        def fake_urlopen(request, timeout=None):
            del timeout
            url = request.full_url
            if url.endswith("/releases/latest"):
                return release_response
            if url == "https://example.invalid/open-jet-v0.4.0.tar.gz":
                return archive_response
            raise AssertionError(url)

        with patch("src.self_update.urlopen", side_effect=fake_urlopen), patch(
            "src.self_update.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["pip"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to install open-jet 0.4.0."):
                update_from_latest_release(current_version="0.3.0")
