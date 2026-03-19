import io
from urllib.error import HTTPError, URLError

import pytest

from autoscene.emulator.base import EmulatorAdapter
from autoscene.emulator.network_device import NetworkDeviceEmulatorAdapter
from autoscene.emulator.null import NullEmulatorAdapter
from autoscene.emulator.qt_drive_cluster import QtDriveClusterEmulatorAdapter
from autoscene.emulator.registry import (
    build_emulator_registry,
    create_emulator,
    register_emulator,
)


class RecordingEmulator(EmulatorAdapter):
    def __init__(self):
        self.commands = []

    def execute(self, command: str) -> str:
        self.commands.append(command)
        return f"ok:{command}"


def test_base_send_forwards_to_execute() -> None:
    emulator = RecordingEmulator()
    assert emulator.send("abc") == "ok:abc"
    assert emulator.send({"x": 1}) == "ok:{'x': 1}"
    assert emulator.commands == ["abc", "{'x': 1}"]


def test_null_emulator_raises() -> None:
    emulator = NullEmulatorAdapter()
    with pytest.raises(RuntimeError, match="No emulator configured"):
        emulator.execute("x")
    with pytest.raises(RuntimeError, match="No emulator configured"):
        emulator.send({"k": "v"})


def test_network_execute_uses_send(monkeypatch: pytest.MonkeyPatch) -> None:
    emulator = NetworkDeviceEmulatorAdapter(base_url="http://localhost:9999")
    calls = []
    monkeypatch.setattr(
        emulator,
        "send",
        lambda payload, endpoint=None, method=None, headers=None: calls.append(payload)
        or "ok",
    )
    assert emulator.execute("reboot") == "ok"
    assert calls == [{"command": "reboot"}]


def test_network_send_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b"done"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "autoscene.emulator.network_device.urlopen",
        fake_urlopen,
    )
    emulator = NetworkDeviceEmulatorAdapter(
        base_url="http://127.0.0.1:8080",
        default_endpoint="/api/device/events",
        default_headers={"X-Test": "1"},
        timeout_seconds=3.0,
    )
    out = emulator.send({"device_id": "d1", "value": 7})
    assert out == "done"
    assert captured["url"] == "http://127.0.0.1:8080/api/device/events"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 3.0
    assert b'"device_id": "d1"' in captured["body"]
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["content-type"] == "application/json"
    assert headers["x-test"] == "1"


def test_network_send_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("autoscene.emulator.network_device.urlopen", fake_urlopen)
    emulator = NetworkDeviceEmulatorAdapter(base_url="http://localhost:8080")
    emulator.send("plain text", endpoint="/x", method="PUT")
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["content-type"].startswith("text/plain")
    assert captured["body"] == b"plain text"


def test_network_send_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):
        raise HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(b"oops"))

    monkeypatch.setattr("autoscene.emulator.network_device.urlopen", fake_urlopen)
    emulator = NetworkDeviceEmulatorAdapter(base_url="http://localhost:8080")
    with pytest.raises(RuntimeError, match="HTTP error while sending device event"):
        emulator.send({"a": 1})


def test_network_send_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout):
        raise URLError("no route")

    monkeypatch.setattr("autoscene.emulator.network_device.urlopen", fake_urlopen)
    emulator = NetworkDeviceEmulatorAdapter(base_url="http://localhost:8080")
    with pytest.raises(RuntimeError, match="Network error while sending device event"):
        emulator.send({"a": 1})


def test_registry_defaults_and_custom_registration() -> None:
    assert isinstance(create_emulator({}), NullEmulatorAdapter)
    assert isinstance(create_emulator({"type": "network_device"}), NetworkDeviceEmulatorAdapter)
    assert isinstance(create_emulator({"type": "external_device"}), NetworkDeviceEmulatorAdapter)
    assert isinstance(create_emulator({"type": "qt_drive_cluster"}), QtDriveClusterEmulatorAdapter)
    assert isinstance(create_emulator({"type": "qt_cluster"}), QtDriveClusterEmulatorAdapter)

    class CustomEmulator(NullEmulatorAdapter):
        pass

    register_emulator("custom", CustomEmulator)
    assert isinstance(create_emulator({"type": "custom"}), CustomEmulator)


def test_registry_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown emulator type"):
        create_emulator({"type": "missing"})


def test_emulator_registry_supports_namespaced_plugins() -> None:
    class PluginEmulator(NullEmulatorAdapter):
        pass

    class SamplePlugin:
        namespace = "sample"
        override = False

        @staticmethod
        def register_emulators(registry) -> None:
            registry.register("device", PluginEmulator)

    registry = build_emulator_registry(plugins=(SamplePlugin(),))

    assert isinstance(registry.create({"type": "sample.device"}), PluginEmulator)


def test_emulator_registry_supports_plugin_override() -> None:
    class OverrideNullEmulator(NullEmulatorAdapter):
        pass

    class OverridePlugin:
        override = True

        @staticmethod
        def register_emulators(registry) -> None:
            registry.register("none", OverrideNullEmulator)

    emulator = create_emulator({"type": "none"}, plugins=(OverridePlugin(),))

    assert isinstance(emulator, OverrideNullEmulator)


def test_emulator_registry_rejects_collisions_without_override() -> None:
    class ConflictingPlugin:
        override = False

        @staticmethod
        def register_emulators(registry) -> None:
            registry.register("none", NullEmulatorAdapter)

    with pytest.raises(ValueError, match="Emulator already registered: none"):
        build_emulator_registry(plugins=(ConflictingPlugin(),))


def test_qt_cluster_execute_routes_to_known_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    emulator = QtDriveClusterEmulatorAdapter()
    calls = []
    monkeypatch.setattr(
        NetworkDeviceEmulatorAdapter,
        "send",
        lambda self, payload, endpoint=None, method=None, headers=None: calls.append(
            (payload, endpoint, method, headers)
        )
        or "ok",
    )
    assert emulator.execute("state") == "ok"
    assert emulator.execute("demo_start") == "ok"
    assert emulator.execute("demo_stop") == "ok"
    assert calls == [
        (None, "/state", "GET", None),
        (None, "/demo/start", "POST", None),
        (None, "/demo/stop", "POST", None),
    ]


def test_qt_cluster_send_normalizes_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return b"done"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "autoscene.emulator.network_device.urlopen",
        fake_urlopen,
    )
    emulator = QtDriveClusterEmulatorAdapter(
        state_defaults={"demo_mode": False, "mode": "sport"},
        base_url="http://127.0.0.1:8765",
    )
    out = emulator.send({"speed": 126, "rpm": 4200, "gear": "m2", "temp": 95})
    assert out == "done"
    assert captured["url"] == "http://127.0.0.1:8765/state"
    assert captured["method"] == "POST"
    assert b'"speed": 126' in captured["body"]
    assert b'"rpm": 4200' in captured["body"]
    assert b'"gear": "M2"' in captured["body"]
    assert b'"coolant_temp": 95' in captured["body"]
    assert b'"demo_mode": false' in captured["body"]
    assert b'"mode": "SPORT"' in captured["body"]
    assert b'"mode_index": 1' in captured["body"]


def test_qt_cluster_send_command_payload_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    emulator = QtDriveClusterEmulatorAdapter()
    calls = []
    monkeypatch.setattr(
        NetworkDeviceEmulatorAdapter,
        "send",
        lambda self, payload, endpoint=None, method=None, headers=None: calls.append(
            (payload, endpoint, method, headers)
        )
        or "ok",
    )
    assert emulator.send({"command": "demo_stop"}) == "ok"
    assert calls == [
        (None, "/demo/stop", "POST", None),
    ]


def test_qt_cluster_rejects_removed_command_aliases() -> None:
    emulator = QtDriveClusterEmulatorAdapter()

    with pytest.raises(ValueError, match="Unsupported qt_drive_cluster command"):
        emulator.execute("get_state")

    with pytest.raises(ValueError, match="Unsupported qt_drive_cluster command"):
        emulator.execute("start_demo")

    with pytest.raises(ValueError, match="Unsupported qt_drive_cluster command"):
        emulator.execute("stop_demo")


def test_qt_cluster_rejects_removed_action_payload_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    monkeypatch.setattr(
        NetworkDeviceEmulatorAdapter,
        "send",
        lambda self, payload, endpoint=None, method=None, headers=None: captured.update(
            {
                "payload": payload,
                "endpoint": endpoint,
                "method": method,
                "headers": headers,
            }
        )
        or "ok",
    )

    emulator = QtDriveClusterEmulatorAdapter()
    emulator.send({"action": "state"})

    assert captured["payload"] == {"action": "state"}
    assert captured["endpoint"] is None
    assert captured["method"] is None
