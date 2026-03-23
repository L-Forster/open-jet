from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main as cli_main
from src.device_sources import DeviceSource
from src.peripherals import PeripheralDevice, PeripheralKind, PeripheralTransport


class DeviceCliTests(unittest.TestCase):
    def test_device_list_cli_prints_current_ids(self) -> None:
        stdout = io.StringIO()
        source = DeviceSource(
            primary_ref="camera0",
            refs=("camera0", "camera:/dev/video0", "video0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
        )

        with patch("src.cli.load_config", return_value={}), patch(
            "src.cli.list_device_sources",
            return_value=[source],
        ), patch(
            "src.cli.sync_devices_registry",
            return_value=Path("/tmp/devices.md"),
        ), patch("sys.stdout", stdout):
            cli_main(["device", "list"])

        rendered = stdout.getvalue()
        self.assertIn("Device registry: /tmp/devices.md", rendered)
        self.assertIn("- camera0: Front Camera", rendered)
        self.assertIn("open-jet device add <existing_id> <new_id>", rendered)

    def test_device_add_cli_saves_alias(self) -> None:
        stdout = io.StringIO()
        cfg: dict[str, object] = {}
        source = DeviceSource(
            primary_ref="deskcam",
            refs=("deskcam", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
        )

        with patch("src.cli.load_config", return_value=cfg), patch(
            "src.cli.assign_device_alias",
            return_value=source,
        ) as assign_alias, patch(
            "src.cli.sync_devices_registry",
            return_value=Path("/tmp/devices.md"),
        ), patch("src.cli.save_config") as save_config, patch("sys.stdout", stdout):
            cli_main(["device", "add", "camera0", "deskcam"])

        assign_alias.assert_called_once_with(cfg, reference="camera0", alias="deskcam")
        save_config.assert_called_once_with(cfg)
        rendered = stdout.getvalue()
        self.assertIn("Saved device id deskcam for Front Camera.", rendered)
        self.assertIn("Use @deskcam in chat", rendered)


if __name__ == "__main__":
    unittest.main()
