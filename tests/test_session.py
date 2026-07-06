import json
import os
import time
from datetime import datetime, timedelta

import session
from session import cleanup_expired_sessions, save_session


def _write_session(chats_dir, name, age_seconds):
    """Write an empty session file and backdate its mtime."""
    path = chats_dir / name
    path.write_text(json.dumps([]))
    past = time.time() - age_seconds
    os.utime(path, (past, past))
    return path


# ── cleanup_expired_sessions ─────────────────────────────────────────────────

def test_cleanup_removes_expired_sessions(tmp_chats, monkeypatch):
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 1)
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 999)
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)

    old = _write_session(tmp_chats, "old.json", age_seconds=7200)  # 2 h old
    fresh = _write_session(tmp_chats, "fresh.json", age_seconds=60)  # 1 min old

    removed = cleanup_expired_sessions()
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_removes_idle_sessions(tmp_chats, monkeypatch):
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 168)   # 1 week — won't trigger
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 4)
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)

    idle = _write_session(tmp_chats, "idle.json", age_seconds=5 * 3600)  # 5 h idle
    active = _write_session(tmp_chats, "active.json", age_seconds=60)

    removed = cleanup_expired_sessions()
    assert removed == 1
    assert not idle.exists()
    assert active.exists()


def test_cleanup_enforces_max_sessions(tmp_chats, monkeypatch):
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 168)
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 168)
    monkeypatch.setattr(session, "MAX_SESSIONS", 2)

    for i in range(4):
        _write_session(tmp_chats, f"s{i}.json", age_seconds=i * 10)

    cleanup_expired_sessions()
    remaining = list(tmp_chats.glob("*.json"))
    assert len(remaining) == 2


def test_cleanup_ignores_non_json_files(tmp_chats, monkeypatch):
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 1)
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 999)
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)

    txt = tmp_chats / "notes.txt"
    txt.write_text("hello")
    past = time.time() - 9999
    os.utime(txt, (past, past))

    removed = cleanup_expired_sessions()
    assert removed == 0
    assert txt.exists()


def test_cleanup_returns_zero_when_empty(tmp_chats, monkeypatch):
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 1)
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 999)
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)
    assert cleanup_expired_sessions() == 0


# ── API keys must never be persisted to session files ────────────────────────

def test_save_session_atomic_with_restricted_perms(tmp_chats):
    """Session files are written via temp+rename, land with 0o600, no .tmp left."""
    sid = "chat_20240101_130000_efgh5678"
    save_session(sid, [{"role": "user", "content": "hi"}])

    path = tmp_chats / f"{sid}.json"
    assert path.exists()
    assert (path.stat().st_mode & 0o777) == 0o600
    data = json.loads(path.read_text())
    assert data["messages"][0]["content"] == "hi"
    assert not list(tmp_chats.glob("*.tmp"))


def test_saved_session_contains_no_api_key(tmp_chats):
    """Only message role/content is persisted — provider keys live in memory only."""
    sid = "chat_20240101_120000_abcd1234"
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there", "sources": ["a.py"]},
    ]
    save_session(sid, messages)

    raw = (tmp_chats / f"{sid}.json").read_text()
    assert "api_key" not in raw
    assert "sk-" not in raw


# ── expiry (creation time) and idle (mtime) are independent signals ──────────

def test_expiry_uses_filename_creation_time_not_mtime(tmp_chats, monkeypatch):
    """A session created long ago expires even if it was just written (fresh mtime)."""
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 24)
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 999)  # idle won't fire
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)

    old_ts = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d_%H%M%S")
    path = tmp_chats / f"chat_{old_ts}_abcd1234.json"
    path.write_text(json.dumps([]))  # mtime = now (fresh)

    removed = cleanup_expired_sessions()
    assert removed == 1
    assert not path.exists()


def test_idle_uses_mtime_not_creation_time(tmp_chats, monkeypatch):
    """A recently-created session still gets pruned once it goes idle (old mtime)."""
    monkeypatch.setattr(session, "SESSION_EXPIRY_HOURS", 168)  # expiry won't fire
    monkeypatch.setattr(session, "SESSION_IDLE_TIMEOUT_HOURS", 4)
    monkeypatch.setattr(session, "MAX_SESSIONS", 1000)

    fresh_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = tmp_chats / f"chat_{fresh_ts}_abcd1234.json"
    path.write_text(json.dumps([]))
    past = time.time() - 5 * 3600  # last written 5 h ago
    os.utime(path, (past, past))

    removed = cleanup_expired_sessions()
    assert removed == 1
    assert not path.exists()
