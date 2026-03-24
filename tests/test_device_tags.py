from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src.app import OpenJetApp
from src.device_sources import DeviceSource, assign_device_alias, capture_gpio_text, list_device_sources, render_devices_markdown, resolve_device_source, set_device_enabled, write_devices_markdown
from src.observation import ObservationStore
from src.peripherals import Observation, ObservationModality, PeripheralDevice, PeripheralKind, PeripheralTransport


class DeviceSourceListingTests(unittest.TestCase):
    def test_list_device_sources_includes_default_and_custom_aliases(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Front Camera",
            path="/dev/video0",
        )
        with patch("src.device_sources.discover_peripherals", return_value=[device]):
            sources = list_device_sources({"device_aliases": {"front": device.id}})

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].primary_ref, "front")
        self.assertIn("camera0", sources[0].refs)
        self.assertIn("video0", sources[0].refs)

    def test_assign_device_alias_updates_config_and_resolves_by_alias(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Front Camera",
            path="/dev/video0",
        )
        cfg: dict[str, object] = {}
        with patch("src.device_sources.discover_peripherals", return_value=[device]):
            source = assign_device_alias(cfg, reference="camera0", alias="front")
            resolved = resolve_device_source("front", cfg)

        self.assertEqual(source.primary_ref, "front")
        self.assertIsNotNone(resolved)
        self.assertEqual(cfg["device_aliases"], {"front": device.id})

    def test_set_device_enabled_persists_disabled_state(self) -> None:
        device = PeripheralDevice(
            id="microphone:hw:2,0",
            kind=PeripheralKind.MICROPHONE,
            transport=PeripheralTransport.ALSA,
            label="Room Mic",
            path="hw:2,0",
        )
        cfg: dict[str, object] = {}
        with patch("src.device_sources.discover_peripherals", return_value=[device]):
            disabled = set_device_enabled(cfg, reference="mic0", enabled=False)
            resolved = resolve_device_source("mic0", cfg)

        self.assertFalse(disabled.enabled)
        self.assertIsNotNone(resolved)
        self.assertFalse(resolved.enabled)
        self.assertEqual(cfg["disabled_device_ids"], [device.id])

    def test_list_device_sources_includes_configured_gpio_bindings(self) -> None:
        chip = PeripheralDevice(
            id="sensor:/dev/gpiochip0",
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.GPIO,
            label="GPIO chip /dev/gpiochip0",
            path="/dev/gpiochip0",
        )
        cfg = {
            "gpio_bindings": [
                {
                    "name": "door",
                    "chip": "gpio0",
                    "lines": [17],
                    "label": "Door Sensor",
                },
                {
                    "name": "relay-bank",
                    "chip": "/dev/gpiochip0",
                    "lines": [22, 23],
                    "label": "Relay Bank",
                },
            ]
        }
        with patch("src.device_sources.discover_peripherals", return_value=[chip]):
            sources = list_device_sources(cfg)

        refs = {source.primary_ref: source for source in sources}
        self.assertIn("door", refs)
        self.assertIn("relay-bank", refs)
        self.assertEqual(refs["door"].device.metadata["gpio_lines"], [17])
        self.assertEqual(refs["relay-bank"].device.metadata["gpio_lines"], [22, 23])


class GpioCaptureTests(unittest.TestCase):
    def test_capture_gpio_text_persists_text_buffer(self) -> None:
        device = PeripheralDevice(
            id="sensor:/dev/gpiochip0",
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.GPIO,
            label="GPIO chip /dev/gpiochip0",
            path="/dev/gpiochip0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            from src.observation import ObservationStore

            store = ObservationStore(Path(tmp) / "state")
            with patch("src.device_sources.resolve_binary", return_value="/usr/bin/gpioinfo"), patch(
                "src.device_sources.run_command",
                return_value=SimpleNamespace(ok=True, stdout="gpiochip0 - 8 lines:\nline 0: unnamed unused input active-high\n", stderr=""),
            ):
                observation = capture_gpio_text(device, store=store)
                self.assertEqual(observation.modality, ObservationModality.TEXT)
                self.assertTrue(Path(observation.payload_ref or "").is_file())
                self.assertIn("GPIO snapshot", observation.summary)

    def test_capture_gpio_text_filters_to_configured_binding_lines(self) -> None:
        device = PeripheralDevice(
            id="sensor:gpio-binding:door",
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.GPIO,
            label="Door Sensor",
            path="/dev/gpiochip0",
            metadata={"gpio_chip_path": "/dev/gpiochip0", "gpio_lines": [17]},
        )
        gpio_text = (
            "gpiochip0 - 32 lines:\n"
            "line 16: unnamed unused input active-high\n"
            "line 17: unnamed used input active-high\n"
            "line 18: unnamed unused input active-high\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            from src.observation import ObservationStore

            store = ObservationStore(Path(tmp) / "state")
            with patch("src.device_sources.resolve_binary", return_value="/usr/bin/gpioinfo"), patch(
                "src.device_sources.run_command",
                return_value=SimpleNamespace(ok=True, stdout=gpio_text, stderr=""),
            ):
                observation = capture_gpio_text(device, store=store)
            payload = Path(observation.payload_ref or "").read_text(encoding="utf-8")
            self.assertIn("line 17:", payload)
            self.assertNotIn("line 16:", payload)
            self.assertNotIn("line 18:", payload)


class DevicesMarkdownTests(unittest.TestCase):
    def test_render_devices_markdown_lists_ids_without_loading_logs(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Front Camera",
            path="/dev/video0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            from src.observation import ObservationStore

            store = ObservationStore(Path(tmp) / "state")
            with patch("src.device_sources.discover_peripherals", return_value=[device]):
                rendered = render_devices_markdown({"device_aliases": {"front": device.id}}, store=store)

        self.assertIn("# Devices", rendered)
        self.assertIn("## front", rendered)
        self.assertIn("chat_tag: `@front`", rendered)
        self.assertIn("latest_payload_file: `none`", rendered)
        self.assertIn("not preloaded", rendered)

    def test_write_devices_markdown_returns_absolute_registry_path(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Front Camera",
            path="/dev/video0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            from src.observation import ObservationStore

            store = ObservationStore(Path(tmp) / "state")
            with patch("src.device_sources.discover_peripherals", return_value=[device]):
                registry_path = write_devices_markdown({"device_aliases": {"front": device.id}}, store=store)

        self.assertTrue(registry_path.is_absolute())



class AppDeviceTagTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_text_with_device_tag_points_to_registry_file(self) -> None:
        app = OpenJetApp()
        app.agent = SimpleNamespace(
            messages=[],
            set_turn_context=Mock(),
            persistent_context_tokens=Mock(return_value=0),
            runtime_overhead_tokens=Mock(return_value=0),
            _messages_for_runtime=Mock(return_value=[]),
        )
        app.client = SimpleNamespace(context_window_tokens=2048)
        source = DeviceSource(
            primary_ref="front",
            refs=("front", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "devices.md"
            registry.write_text("# Devices\n\n## front\n- latest_payload_file: `none`\n", encoding="utf-8")

            with patch.object(app.commands, "maybe_handle", AsyncMock(return_value=False)), patch.object(
                app, "resolve_device_source", side_effect=lambda ref: source if ref == "front" else None
            ), patch.object(
                app, "write_devices_registry", return_value=registry
            ), patch.object(
                app, "_load_mentioned_files_into_context", AsyncMock()
            ), patch.object(
                app, "_start_agent_turn"
            ), patch.object(
                app, "persist_session_state"
            ), patch.object(
                app, "persist_harness_state"
            ), patch.object(
                app, "_render_token_counter"
            ), patch.object(
                app, "_begin_turn_trace"
            ):
                await app.submit_text("@front what is on the desk?")
                app._prepare_turn_context()

        content = app.agent.messages[-1]["content"]
        self.assertIsInstance(content, str)
        self.assertEqual(content, "what is on the desk?")
        self.assertIn("what is on the desk?", content)
        self.assertNotIn("@front what is on the desk?", content)
        self.assertEqual(app._active_turn_device_refs, ("front",))
        turn_context = app.agent.set_turn_context.call_args.args[0]
        joined_context = "\n\n".join(message["content"] for message in turn_context)
        self.assertIn("IO device registry located in", joined_context)
        self.assertIn(str(registry), joined_context)
        self.assertIn("Referenced device ids for this turn: front.", joined_context)

    async def test_submit_text_with_spoofed_device_writes_real_registry_file(self) -> None:
        app = OpenJetApp()
        app.cfg = {}
        app.agent = SimpleNamespace(
            messages=[],
            set_turn_context=Mock(),
            persistent_context_tokens=Mock(return_value=0),
            runtime_overhead_tokens=Mock(return_value=0),
            _messages_for_runtime=Mock(return_value=[]),
        )
        app.client = SimpleNamespace(context_window_tokens=2048)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.observation_store = ObservationStore(root / ".openjet" / "state" / "observations")
            device = PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            )
            with patch("src.device_sources.discover_peripherals", return_value=[device]), patch.object(
                app.commands,
                "maybe_handle",
                AsyncMock(return_value=False),
            ), patch.object(
                app,
                "_load_mentioned_files_into_context",
                AsyncMock(),
            ), patch.object(
                app,
                "_start_agent_turn",
            ), patch.object(
                app,
                "persist_session_state",
            ), patch.object(
                app,
                "persist_harness_state",
            ), patch.object(
                app,
                "_render_token_counter",
            ), patch.object(
                app,
                "_begin_turn_trace",
            ):
                await app.submit_text("@camera0 what is on the desk?")
                app._prepare_turn_context()

            registry = root / ".openjet" / "state" / "devices.md"
            registry_exists = registry.is_file()
            registry_text = registry.read_text(encoding="utf-8") if registry_exists else ""
            content = app.agent.messages[-1]["content"]
            turn_context = app.agent.set_turn_context.call_args.args[0]
            joined_context = "\n\n".join(message["content"] for message in turn_context)

        self.assertTrue(registry_exists)
        self.assertEqual(content, "what is on the desk?")
        self.assertIn(str(registry), joined_context)
        self.assertIn("Referenced device ids for this turn: camera0.", joined_context)
        self.assertIn("## camera0", registry_text)


class DeviceCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_help_shows_device_namespace_and_hides_compat_aliases(self) -> None:
        app = OpenJetApp()

        handled = await app.commands.maybe_handle("/help")

        self.assertTrue(handled)
        rendered = "\n".join(str(entry) for entry in app.query_one("#chat-log")._entries)
        self.assertIn("/device", rendered)
        self.assertNotIn("/device-add", rendered)
        self.assertNotIn("/device-on", rendered)
        self.assertNotIn("/device-off", rendered)

    async def test_devices_command_lists_device_refs(self) -> None:
        app = OpenJetApp()
        source = DeviceSource(
            primary_ref="front",
            refs=("front", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
        )
        with patch.object(app, "list_device_sources", return_value=[source]):
            handled = await app.commands.maybe_handle("/devices")

        self.assertTrue(handled)
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("tag=@front" in str(entry) for entry in entries))

    async def test_device_list_namespace_lists_device_refs(self) -> None:
        app = OpenJetApp()
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
        with patch.object(app, "write_devices_registry", return_value=Path("/tmp/devices.md")), patch.object(
            app,
            "list_device_sources",
            return_value=[source],
        ):
            handled = await app.commands.maybe_handle("/device list")

        self.assertTrue(handled)
        rendered = "\n".join(str(entry) for entry in app.query_one("#chat-log")._entries)
        self.assertIn("Device registry:", rendered)
        self.assertIn("open-jet device add <existing_id> <new_id>", rendered)

    async def test_devices_command_writes_registry_for_spoofed_devices(self) -> None:
        app = OpenJetApp()
        app.cfg = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.observation_store = ObservationStore(root / ".openjet" / "state" / "observations")
            devices = [
                PeripheralDevice(
                    id="camera:/dev/video0",
                    kind=PeripheralKind.CAMERA,
                    transport=PeripheralTransport.V4L2,
                    label="Front Camera",
                    path="/dev/video0",
                ),
                PeripheralDevice(
                    id="microphone:hw:2,0",
                    kind=PeripheralKind.MICROPHONE,
                    transport=PeripheralTransport.ALSA,
                    label="Room Mic",
                    path="hw:2,0",
                ),
            ]
            with patch("src.device_sources.discover_peripherals", return_value=devices):
                handled = await app.commands.maybe_handle("/devices")
            registry = root / ".openjet" / "state" / "devices.md"
            registry_exists = registry.is_file()
            rendered = registry.read_text(encoding="utf-8") if registry_exists else ""

        self.assertTrue(handled)
        self.assertTrue(registry_exists)
        self.assertIn("## camera0", rendered)
        self.assertIn("## mic0", rendered)
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any(str(registry) in str(entry) for entry in entries))

    async def test_device_add_command_persists_id(self) -> None:
        app = OpenJetApp()
        source = DeviceSource(
            primary_ref="front",
            refs=("front", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
        )
        with patch.object(app, "assign_device_alias", return_value=source) as assign_alias, patch.object(
            app, "write_devices_registry", return_value=Path("/tmp/devices.md")
        ) as write_devices_registry, patch(
            "src.commands.save_config"
        ) as save_config:
            handled = await app.commands.maybe_handle("/device add camera0 front")

        self.assertTrue(handled)
        assign_alias.assert_called_once_with("camera0", "front")
        write_devices_registry.assert_called_once()
        save_config.assert_called_once()

    async def test_device_add_command_rewrites_registry_with_alias(self) -> None:
        app = OpenJetApp()
        app.cfg = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.observation_store = ObservationStore(root / ".openjet" / "state" / "observations")
            device = PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            )
            with patch("src.device_sources.discover_peripherals", return_value=[device]), patch(
                "src.commands.save_config"
            ):
                handled = await app.commands.maybe_handle("/device add camera0 deskcam")
            registry = root / ".openjet" / "state" / "devices.md"
            rendered = registry.read_text(encoding="utf-8")

        self.assertTrue(handled)
        self.assertEqual(app.cfg["device_aliases"], {"deskcam": device.id})
        self.assertIn("## deskcam", rendered)
        self.assertIn("chat_tag: `@deskcam`", rendered)

    async def test_device_off_command_persists_disabled_state(self) -> None:
        app = OpenJetApp()
        source = DeviceSource(
            primary_ref="room-mic",
            refs=("room-mic", "mic0"),
            device=PeripheralDevice(
                id="microphone:hw:2,0",
                kind=PeripheralKind.MICROPHONE,
                transport=PeripheralTransport.ALSA,
                label="Room Mic",
                path="hw:2,0",
            ),
            enabled=False,
        )
        with patch.object(app, "set_device_enabled", return_value=source) as set_enabled, patch.object(
            app, "write_devices_registry", return_value=Path("/tmp/devices.md")
        ) as write_devices_registry, patch(
            "src.commands.save_config"
        ) as save_config:
            handled = await app.commands.maybe_handle("/device off room-mic")

        self.assertTrue(handled)
        set_enabled.assert_called_once_with("room-mic", False)
        write_devices_registry.assert_called_once()
        save_config.assert_called_once()

    async def test_device_toggle_rewrites_registry_state_with_spoofed_device(self) -> None:
        app = OpenJetApp()
        app.cfg = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app.observation_store = ObservationStore(root / ".openjet" / "state" / "observations")
            device = PeripheralDevice(
                id="microphone:hw:2,0",
                kind=PeripheralKind.MICROPHONE,
                transport=PeripheralTransport.ALSA,
                label="Room Mic",
                path="hw:2,0",
            )
            with patch("src.device_sources.discover_peripherals", return_value=[device]), patch(
                "src.commands.save_config"
            ):
                handled_off = await app.commands.maybe_handle("/device off mic0")
                registry = root / ".openjet" / "state" / "devices.md"
                off_rendered = registry.read_text(encoding="utf-8")
                handled_on = await app.commands.maybe_handle("/device on mic0")
                on_rendered = registry.read_text(encoding="utf-8")

        self.assertTrue(handled_off)
        self.assertTrue(handled_on)
        self.assertIn("- state: `disabled`", off_rendered)
        self.assertIn("- state: `enabled`", on_rendered)
