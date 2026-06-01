from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.self_update import (
    RepoUpdateInfo,
    _has_tracked_changes,
    _install_command,
    _migrate_config_after_update,
    _sync_managed_llama_cpp_after_update,
    available_update,
    update_from_latest_release,
)
from src.config import MANAGED_MODELS_DIR


class SelfUpdateTests(unittest.TestCase):
    def test_has_tracked_changes_ignores_untracked_files(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain", "--untracked-files=no"],
            returncode=0,
            stdout="",
            stderr="",
        )

        with patch("src.self_update.subprocess.run", return_value=completed):
            self.assertFalse(_has_tracked_changes(update_from_latest_release.__globals__["_REPO_ROOT"]))

    def test_available_update_prefers_current_upstream_branch_over_origin_head(self) -> None:
        def fake_git_output(*args: str) -> str | None:
            if args == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"):
                return "origin/main"
            if args == ("rev-parse", "HEAD"):
                return "1111111222222333333444444555555666666777"
            if args == ("rev-parse", "FETCH_HEAD"):
                return "aaaaaaa222222333333444444555556666666777"
            raise AssertionError(args)

        with patch("src.self_update._git_output", side_effect=fake_git_output), patch(
            "src.self_update._git_capture",
            return_value="",
        ), patch(
            "src.self_update._git_ok",
            return_value=True,
        ):
            update = available_update()

        self.assertEqual(
            update,
            RepoUpdateInfo(
                remote="origin",
                branch="main",
                local_commit="1111111222222333333444444555555666666777",
                remote_commit="aaaaaaa222222333333444444555556666666777",
            ),
        )

    def test_available_update_skips_when_remote_matches_head(self) -> None:
        def fake_git_output(*args: str) -> str | None:
            if args == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"):
                return None
            if args == ("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"):
                return "refs/remotes/origin/master"
            if args == ("rev-parse", "HEAD"):
                return "1111111222222333333444444555555666666777"
            if args == ("rev-parse", "FETCH_HEAD"):
                return "1111111222222333333444444555555666666777"
            raise AssertionError(args)

        with patch("src.self_update._git_output", side_effect=fake_git_output), patch(
            "src.self_update._git_capture",
            return_value="",
        ):
            self.assertIsNone(available_update())

    def test_available_update_returns_remote_commit_when_head_is_behind(self) -> None:
        def fake_git_output(*args: str) -> str | None:
            if args == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"):
                return None
            if args == ("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"):
                return "refs/remotes/origin/master"
            if args == ("rev-parse", "HEAD"):
                return "1111111222222333333444444555555666666777"
            if args == ("rev-parse", "FETCH_HEAD"):
                return "aaaaaaa222222333333444444555556666666777"
            raise AssertionError(args)

        with patch("src.self_update._git_output", side_effect=fake_git_output), patch(
            "src.self_update._git_capture",
            return_value="",
        ), patch(
            "src.self_update._git_ok",
            return_value=True,
        ):
            update = available_update()

        self.assertEqual(
            update,
            RepoUpdateInfo(
                remote="origin",
                branch="master",
                local_commit="1111111222222333333444444555555666666777",
                remote_commit="aaaaaaa222222333333444444555556666666777",
            ),
        )

    def test_update_from_latest_release_uses_repo_head_when_checkout_is_behind_remote(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: "src/app.py\n" if args == ("diff", "--name-only", update.local_commit, update.remote_commit) else None,
        ), patch("src.self_update._sync_managed_llama_cpp_after_update", return_value=None), patch(
            "src.self_update._migrate_config_after_update",
            return_value=False,
        ), patch(
            "src.self_update.subprocess.run",
        ) as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(message, "Updated open-jet repo from 1111111 to aaaaaaa.")
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[0].args[0][:4], ["git", "pull", "--ff-only", "origin"])
        self.assertEqual(run_mock.call_args_list[1].args[0][:2], ["bash", str(update_from_latest_release.__globals__["_INSTALL_SCRIPT"])])
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"]["OPENJET_INSTALL_MODE"], "update")
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"]["OPENJET_UPDATE_REINSTALL"], "0")

    def test_update_from_latest_release_reinstalls_when_update_changes_install_requirements(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: "pyproject.toml\n" if args == ("diff", "--name-only", update.local_commit, update.remote_commit) else None,
        ), patch("src.self_update._sync_managed_llama_cpp_after_update", return_value=None), patch(
            "src.self_update._migrate_config_after_update",
            return_value=False,
        ), patch(
            "src.self_update.subprocess.run",
        ) as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(message, "Updated open-jet repo from 1111111 to aaaaaaa.")
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"]["OPENJET_UPDATE_REINSTALL"], "1")

    def test_update_from_latest_release_reinstalls_when_installer_changes(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: "install.bat\n" if args == ("diff", "--name-only", update.local_commit, update.remote_commit) else None,
        ), patch("src.self_update._sync_managed_llama_cpp_after_update", return_value=None), patch(
            "src.self_update._migrate_config_after_update",
            return_value=False,
        ), patch(
            "src.self_update.subprocess.run",
        ) as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(message, "Updated open-jet repo from 1111111 to aaaaaaa.")
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"]["OPENJET_UPDATE_REINSTALL"], "1")

    def test_install_command_uses_batch_installer_on_windows(self) -> None:
        with patch("src.self_update.sys.platform", "win32"):
            command = _install_command()

        self.assertEqual(command[:2], ["cmd", "/c"])
        self.assertTrue(command[2].endswith("install.bat"))

    def test_update_from_latest_release_refuses_repo_update_when_checkout_is_dirty(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot update repo checkout with local changes. Commit or stash them first.",
            ):
                update_from_latest_release(current_version="0.3.0")

    def test_update_from_latest_release_raises_clear_error_when_pull_fails(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._sync_managed_llama_cpp_after_update",
            return_value=None,
        ), patch(
            "src.self_update._migrate_config_after_update",
            return_value=False,
        ), patch(
            "src.self_update.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["git"]),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to update repo checkout from 1111111 to aaaaaaa.",
            ):
                update_from_latest_release(current_version="0.3.0")

    def test_update_from_latest_release_does_not_source_build_managed_llama_cpp(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: "src/app.py\n" if args == ("diff", "--name-only", update.local_commit, update.remote_commit) else None,
        ), patch(
            "src.self_update._sync_managed_llama_cpp_after_update",
            return_value=None,
        ) as sync_llama, patch(
            "src.self_update._migrate_config_after_update",
            return_value=False,
        ), patch("src.self_update.subprocess.run") as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(
            message,
            "Updated open-jet repo from 1111111 to aaaaaaa.",
        )
        sync_llama.assert_called_once_with()
        self.assertEqual(run_mock.call_count, 2)

    def test_sync_managed_llama_cpp_after_update_does_not_run_source_build(self) -> None:
        with patch("src.self_update.subprocess.run") as run_mock:
            synced = _sync_managed_llama_cpp_after_update()

        self.assertIsNone(synced)
        run_mock.assert_not_called()

    def test_update_from_latest_release_migrates_model_config_after_repo_update(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._has_tracked_changes",
            return_value=False,
        ), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: "src/config.py\n" if args == ("diff", "--name-only", update.local_commit, update.remote_commit) else None,
        ), patch("src.self_update._sync_managed_llama_cpp_after_update", return_value=None), patch(
            "src.self_update._migrate_config_after_update",
            return_value=True,
        ) as migrate_config, patch("src.self_update.subprocess.run"):
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(
            message,
            "Updated open-jet repo from 1111111 to aaaaaaa. Updated model config for Qwen3.6 MTP.",
        )
        migrate_config.assert_called_once_with()

    def test_update_from_latest_release_runs_model_config_migration_when_repo_is_current(self) -> None:
        with patch("src.self_update.available_update", return_value=None), patch(
            "src.self_update._migrate_config_after_update",
            return_value=True,
        ):
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(
            message,
            "open-jet repo is already up to date. Updated model config for Qwen3.6 MTP.",
        )

    def test_migrate_config_after_update_persists_legacy_qwen_mtp_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            legacy_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-mtp.gguf")
            migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-MTP.gguf")
            config_path.write_text(
                "\n".join(
                    [
                        "llama_cpp_ref: pull/22673/head",
                        f"llama_model: {legacy_model}",
                        "model_profiles:",
                        "- name: mtp",
                        f"  llama_model: {legacy_model}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("src.self_update._REPO_ROOT", root):
                changed = _migrate_config_after_update()

            text = config_path.read_text(encoding="utf-8")

        self.assertTrue(changed)
        self.assertIn(f"llama_model: {migrated_model}", text)
        self.assertIn(f"model_download_path: {migrated_model}", text)
        self.assertIn("model_source: direct", text)
        self.assertIn("setup_missing_model: true", text)
        self.assertNotIn("setup_update_model:", text)
        self.assertNotIn("model_update_target:", text)
        self.assertIn("https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true", text)
        self.assertIn("llama_mtp: true", text)
        self.assertNotIn("llama_cpp_ref:", text)
