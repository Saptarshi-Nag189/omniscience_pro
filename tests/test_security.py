from pathlib import Path

import pytest

import security
from security import (
    check_rate_limit,
    sanitize_filename,
    sanitize_session_id,
    validate_path_within_directory,
)

# ── sanitize_session_id ──────────────────────────────────────────────────────

def test_sanitize_session_id_valid():
    sid = "chat_20240101_120000_abcd1234"
    assert sanitize_session_id(sid) == sid


def test_sanitize_session_id_rejects_path_traversal():
    with pytest.raises(ValueError):
        sanitize_session_id("../../etc/passwd")


def test_sanitize_session_id_rejects_slash():
    with pytest.raises(ValueError):
        sanitize_session_id("chat/evil")


def test_sanitize_session_id_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_session_id("")


def test_sanitize_session_id_rejects_special_chars():
    with pytest.raises(ValueError):
        sanitize_session_id("chat;rm -rf /")


# ── sanitize_filename ────────────────────────────────────────────────────────

def test_sanitize_filename_strips_null_bytes():
    result = sanitize_filename("file\x00.txt")
    assert "\x00" not in result


def test_sanitize_filename_strips_path_separators():
    result = sanitize_filename("../../evil.py")
    assert ".." not in result
    assert "/" not in result


def test_sanitize_filename_keeps_extension():
    result = sanitize_filename("mycode.py")
    assert result.endswith(".py")


def test_sanitize_filename_handles_unicode():
    result = sanitize_filename("résuméé.txt")
    assert isinstance(result, str)
    assert len(result) > 0


# ── validate_path_within_directory ──────────────────────────────────────────

def test_validate_path_accepts_child(tmp_path):
    child = tmp_path / "sub" / "file.txt"
    assert validate_path_within_directory(child, tmp_path) is True


def test_validate_path_rejects_outside(tmp_path):
    assert validate_path_within_directory(Path("/etc/passwd"), tmp_path) is False


def test_validate_path_rejects_parent_escape(tmp_path):
    escaped = tmp_path / ".." / "outside.txt"
    assert validate_path_within_directory(escaped, tmp_path) is False


# ── check_rate_limit (SQLite-backed) ────────────────────────────────────────

def test_rate_limit_allows_requests_under_threshold(tmp_rl_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 3)
    results = [check_rate_limit("tester") for _ in range(3)]
    assert all(results)


def test_rate_limit_blocks_after_threshold(tmp_rl_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 2)
    check_rate_limit("u")
    check_rate_limit("u")
    assert check_rate_limit("u") is False


def test_rate_limit_isolates_users(tmp_rl_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1)
    check_rate_limit("alice")
    # alice is blocked, bob should still be allowed
    assert check_rate_limit("bob") is True


def test_rate_limit_window_expiry(tmp_rl_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(security, "RATE_LIMIT_WINDOW_SECONDS", 1)
    check_rate_limit("expiry_user")
    # Exhaust the quota
    assert check_rate_limit("expiry_user") is False
    # After the window expires the slot should free up
    import time
    time.sleep(1.1)
    assert check_rate_limit("expiry_user") is True
