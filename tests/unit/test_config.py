"""Unit tests for configuration loading."""

from pathlib import Path

from core.config import Settings, get_settings


def test_settings_defaults():
    s = Settings()
    assert s.app_name == "license-plate-detector"
    assert s.api_port == 8000
    assert 0.0 <= s.conf_thres <= 1.0
    assert 0.0 <= s.iou_thres <= 1.0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("CONF_THRES", "0.5")
    # Clear cache so get_settings re-reads env
    get_settings.cache_clear()
    s = get_settings()
    assert s.api_port == 9000
    assert s.conf_thres == 0.5


def test_model_weights_is_path():
    s = Settings()
    assert isinstance(s.model_weights, Path)
