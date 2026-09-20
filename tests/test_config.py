"""Tests for config._get_env_int bounds handling."""

import config

_KEY = "OMNISCIENCE_TEST_INT"


def test_reads_valid_value(monkeypatch):
    monkeypatch.setenv(_KEY, "42")
    assert config._get_env_int(_KEY, 10, 1, 100) == 42


def test_clamps_below_minimum(monkeypatch):
    monkeypatch.setenv(_KEY, "0")
    assert config._get_env_int(_KEY, 10, 5, 100) == 5


def test_clamps_above_maximum(monkeypatch):
    monkeypatch.setenv(_KEY, "9999")
    assert config._get_env_int(_KEY, 10, 1, 100) == 100


def test_non_integer_returns_default(monkeypatch):
    monkeypatch.setenv(_KEY, "not-an-int")
    assert config._get_env_int(_KEY, 10, 1, 100) == 10


def test_missing_returns_default(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert config._get_env_int(_KEY, 7, 1, 100) == 7
