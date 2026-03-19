from __future__ import annotations

import autoscene.runner.runtime_profile_resolver as resolver_mod
from autoscene.runner.runtime_models import RuntimeProfile


def test_runtime_profile_resolver_binds_plugin_aware_factories() -> None:
    calls: dict[str, object] = {}
    plugins = (object(), object())

    def emulator_factory(config, plugins):
        calls["emulator"] = (dict(config), plugins)
        return "emulator"

    def detector_factory(config, **kwargs):
        calls["detector"] = (dict(config), dict(kwargs))
        return "detector"

    def reader_factory(config):
        calls["reader"] = dict(config)
        return "reader"

    def log_source_factory(config):
        calls["log_source"] = dict(config)
        return "log-source"

    def ocr_engine_factory(config):
        calls["ocr"] = dict(config)
        return "ocr"

    resolver = resolver_mod.RuntimeProfileResolver(
        emulator_factory=emulator_factory,
        detector_factory=detector_factory,
        reader_factory=reader_factory,
        log_source_factory=log_source_factory,
        ocr_engine_factory=ocr_engine_factory,
        capture_factory=lambda config: ("capture", dict(config)),
        actions_factory=lambda **kwargs: ("actions", dict(kwargs)),
        action_dispatcher_factory=lambda **kwargs: ("action_dispatcher", dict(kwargs)),
        check_dispatcher_factory=lambda **kwargs: ("check_dispatcher", dict(kwargs)),
        hook_bus_factory=lambda: "hook-bus",
    )

    profile = resolver.resolve(RuntimeProfile(plugins=plugins))

    assert profile.plugins == plugins
    assert profile.emulator_factory({"type": "mock"}) == "emulator"
    assert profile.detector_factory({"type": "vision"}) == "detector"
    assert profile.reader_factory({"type": "reader"}) == "reader"
    assert profile.log_source_factory({"path": "demo.log"}) == "log-source"
    assert profile.ocr_engine_factory({"type": "ocr"}) == "ocr"

    assert calls["emulator"] == ({"type": "mock"}, plugins)
    assert calls["detector"] == ({"type": "vision"}, {"plugins": plugins})
    assert calls["reader"] == {"type": "reader"}
    assert calls["log_source"] == {"path": "demo.log"}
    assert calls["ocr"] == {"type": "ocr"}


def test_runtime_profile_resolver_preserves_explicit_profile_overrides() -> None:
    explicit_emulator_factory = lambda config: ("emulator", config)
    explicit_capture_factory = lambda config: ("capture", config)
    explicit_hook_bus_factory = lambda: "hook-bus"
    explicit_action_dispatcher_factory = lambda **kwargs: ("action_dispatcher", kwargs)
    explicit_pipeline_factory = lambda: "pipeline"

    override = RuntimeProfile(
        name="custom",
        environment="staging",
        plugins=(object(),),
        emulator_factory=explicit_emulator_factory,
        capture_factory=explicit_capture_factory,
        hook_bus_factory=explicit_hook_bus_factory,
        action_dispatcher_factory=explicit_action_dispatcher_factory,
        pipeline_factory=explicit_pipeline_factory,
    )

    resolved = resolver_mod.RuntimeProfileResolver().resolve(override)

    assert resolved.name == "custom"
    assert resolved.environment == "staging"
    assert resolved.emulator_factory is explicit_emulator_factory
    assert resolved.capture_factory is explicit_capture_factory
    assert resolved.hook_bus_factory is explicit_hook_bus_factory
    assert resolved.action_dispatcher_factory is explicit_action_dispatcher_factory
    assert resolved.pipeline_factory is explicit_pipeline_factory


def test_bind_plugin_aware_factory_inspects_signature_only_once(monkeypatch) -> None:
    plugins = (object(),)
    signature_calls = 0
    factory_calls: list[tuple[dict[str, object], tuple[object, ...]]] = []

    def plugin_aware_factory(config, plugins):
        factory_calls.append((dict(config), plugins))
        return "ok"

    original_signature = resolver_mod.inspect.signature

    def counting_signature(factory):
        nonlocal signature_calls
        signature_calls += 1
        return original_signature(factory)

    monkeypatch.setattr(resolver_mod.inspect, "signature", counting_signature)

    bound = resolver_mod.RuntimeProfileResolver._bind_plugin_aware_factory(
        plugin_aware_factory,
        plugins=plugins,
    )

    assert bound({"type": "first"}) == "ok"
    assert bound({"type": "second"}) == "ok"
    assert signature_calls == 1
    assert factory_calls == [
        ({"type": "first"}, plugins),
        ({"type": "second"}, plugins),
    ]


def test_default_capture_factory_selects_video_static_or_window_capture(monkeypatch) -> None:
    created: dict[str, object] = {}

    def fake_create_video_stream_capture(config):
        created["video"] = dict(config)
        return "video-capture"

    def fake_create_static_image_capture(config):
        created["static"] = dict(config)
        return "static-capture"

    def fake_window_capture(**kwargs):
        created["window"] = dict(kwargs)
        return "window-capture"

    monkeypatch.setattr(
        resolver_mod,
        "create_video_stream_capture",
        fake_create_video_stream_capture,
    )
    monkeypatch.setattr(
        resolver_mod,
        "create_static_image_capture",
        fake_create_static_image_capture,
    )
    monkeypatch.setattr(resolver_mod, "WindowCapture", fake_window_capture)

    video_capture = resolver_mod.default_capture_factory(
        {"type": "video_stream", "source": "demo"}
    )
    static_capture = resolver_mod.default_capture_factory(
        {"type": "static_image", "path": "demo.png"}
    )
    window_capture = resolver_mod.default_capture_factory(
        {"window_title": "Demo", "region": {"left": 1, "top": 2, "width": 3, "height": 4}}
    )

    assert video_capture == "video-capture"
    assert static_capture == "static-capture"
    assert window_capture == "window-capture"
    assert created["video"] == {"type": "video_stream", "source": "demo"}
    assert created["static"] == {"type": "static_image", "path": "demo.png"}
    assert created["window"] == {
        "default_window_title": "Demo",
        "default_region": {"left": 1, "top": 2, "width": 3, "height": 4},
    }
