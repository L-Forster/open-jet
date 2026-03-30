from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from src.self_update import RepoUpdateInfo, available_update, update_from_latest_release


class SelfUpdateTests(unittest.TestCase):
    def test_available_update_skips_when_remote_matches_head(self) -> None:
        def fake_git_output(*args: str) -> str | None:
            if args == ("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"):
                return "refs/remotes/origin/master"
            if args == ("rev-parse", "HEAD"):
                return "1111111222222333333444444555555666666777"
            if args == ("ls-remote", "--heads", "origin", "master"):
                return "1111111222222333333444444555555666666777\trefs/heads/master"
            raise AssertionError(args)

        with patch("src.self_update._git_output", side_effect=fake_git_output):
            self.assertIsNone(available_update())

    def test_available_update_returns_remote_commit_when_head_is_behind(self) -> None:
        def fake_git_output(*args: str) -> str | None:
            if args == ("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"):
                return "refs/remotes/origin/master"
            if args == ("rev-parse", "HEAD"):
                return "1111111222222333333444444555555666666777"
            if args == ("ls-remote", "--heads", "origin", "master"):
                return "aaaaaaa222222333333444444555556666666777\trefs/heads/master"
            raise AssertionError(args)

        with patch("src.self_update._git_output", side_effect=fake_git_output), patch(
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
            "src.self_update._git_output",
            side_effect=lambda *args: (
                ""
                if args == ("status", "--porcelain")
                else "src/app.py\n"
                if args == ("diff", "--name-only", update.local_commit, update.remote_commit)
                else None
            ),
        ), patch("src.self_update.subprocess.run") as run_mock:
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
            "src.self_update._git_output",
            side_effect=lambda *args: (
                ""
                if args == ("status", "--porcelain")
                else "pyproject.toml\n"
                if args == ("diff", "--name-only", update.local_commit, update.remote_commit)
                else None
            ),
        ), patch("src.self_update.subprocess.run") as run_mock:
            message = update_from_latest_release(current_version="0.3.0")

        self.assertEqual(message, "Updated open-jet repo from 1111111 to aaaaaaa.")
        self.assertEqual(run_mock.call_args_list[1].kwargs["env"]["OPENJET_UPDATE_REINSTALL"], "1")

    def test_update_from_latest_release_refuses_repo_update_when_checkout_is_dirty(self) -> None:
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )
        with patch("src.self_update.available_update", return_value=update), patch(
            "src.self_update._git_output",
            side_effect=lambda *args: " M src/app.py" if args == ("status", "--porcelain") else None,
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
            "src.self_update._git_output",
            return_value="",
        ), patch(
            "src.self_update.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["git"]),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to update repo checkout from 1111111 to aaaaaaa.",
            ):
                update_from_latest_release(current_version="0.3.0")
