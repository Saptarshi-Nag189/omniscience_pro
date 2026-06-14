import os
import json
import time
import sqlite3
import pytest


@pytest.fixture
def tmp_chats(tmp_path, monkeypatch):
    """A temporary CHATS_DIR with the module-level constant patched."""
    chats = tmp_path / "chats"
    chats.mkdir()
    import session
    monkeypatch.setattr(session, "CHATS_DIR", str(chats))
    return chats


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary SQLite database with a 'users' table for SQL mode tests."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"
    )
    conn.executemany(
        "INSERT INTO users VALUES (?,?,?)",
        [(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com")],
    )
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def tmp_rl_db(tmp_path, monkeypatch):
    """Patch _RL_DB to a temp path so rate-limit tests don't touch the real DB."""
    rl = str(tmp_path / "rate_limits.db")
    import security
    monkeypatch.setattr(security, "_RL_DB", rl)
    return rl
