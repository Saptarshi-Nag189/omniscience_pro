
import pytest

import security
from sql_mode import query_sqlite_db


@pytest.fixture(autouse=True)
def _isolate_rate_limit(tmp_rl_db, monkeypatch):
    """Isolate every SQL test from the shared rate-limit DB so repeated suite
    runs within the 60s window don't exhaust the quota and turn the blocking
    assertions flaky."""
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)


class FakeLLM:
    """Minimal LLM stub that returns a fixed SQL string."""
    def __init__(self, sql: str):
        self._sql = sql

    def invoke(self, _prompt):
        return self._sql


# ── Keyword blocking ─────────────────────────────────────────────────────────

def test_blocks_non_select_query(tmp_db):
    result = query_sqlite_db(tmp_db, "drop everything", FakeLLM("DROP TABLE users"))
    assert "Only SELECT" in result


def test_blocks_delete(tmp_db):
    result = query_sqlite_db(tmp_db, "remove bob", FakeLLM("DELETE FROM users WHERE id=2"))
    assert "prohibited" in result.lower() or "Only SELECT" in result


def test_blocks_union(tmp_db):
    result = query_sqlite_db(tmp_db, "union query", FakeLLM("SELECT 1 UNION SELECT 2"))
    assert "prohibited" in result.lower()


def test_blocks_pragma(tmp_db):
    result = query_sqlite_db(tmp_db, "pragma test", FakeLLM("PRAGMA user_version"))
    assert "prohibited" in result.lower() or "Only SELECT" in result


def test_blocks_comment_injection(tmp_db):
    result = query_sqlite_db(tmp_db, "comment", FakeLLM("SELECT * FROM users -- comment"))
    assert "prohibited" in result.lower()


def test_blocks_second_statement_single_semicolon(tmp_db):
    result = query_sqlite_db(tmp_db, "two", FakeLLM("SELECT 1; SELECT 2"))
    assert "Multiple SQL statements" in result


def test_allows_single_trailing_semicolon(tmp_db):
    result = query_sqlite_db(tmp_db, "list", FakeLLM("SELECT * FROM users;"))
    assert "Alice" in result


# ── Identifiers that merely contain a keyword substring must NOT be blocked ───

def test_allows_identifier_containing_keyword_substring(tmp_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)
    # "created_at" contains the substring CREATE — word-boundary matching must
    # let this through instead of falsely flagging a prohibited keyword.
    result = query_sqlite_db(
        tmp_db,
        "alias the name column",
        FakeLLM("SELECT name AS created_at FROM users"),
    )
    assert "Alice" in result
    assert "prohibited" not in result.lower()


# ── AND / OR are now allowed (fixed bug) ────────────────────────────────────

def test_allows_and_in_where_clause(tmp_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)
    result = query_sqlite_db(
        tmp_db,
        "find alice with id 1",
        FakeLLM("SELECT * FROM users WHERE id=1 AND name='Alice'"),
    )
    assert "Alice" in result


def test_allows_or_in_where_clause(tmp_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)
    result = query_sqlite_db(
        tmp_db,
        "find alice or bob",
        FakeLLM("SELECT * FROM users WHERE id=1 OR id=2"),
    )
    assert "Alice" in result or "Bob" in result


# ── Valid queries work end-to-end ────────────────────────────────────────────

def test_valid_select_all(tmp_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)
    result = query_sqlite_db(tmp_db, "list all users", FakeLLM("SELECT * FROM users"))
    assert "SQL:" in result
    assert "Alice" in result
    assert "Bob" in result


def test_valid_select_with_limit(tmp_db, monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", 1000)
    result = query_sqlite_db(tmp_db, "first user", FakeLLM("SELECT * FROM users LIMIT 1"))
    assert "SQL:" in result


# ── Runaway query is aborted by the progress-handler deadline ────────────────

def test_runaway_query_hits_timeout(tmp_path, monkeypatch):
    import sqlite3

    import sql_mode

    db = tmp_path / "big.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (n INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(1000)])
    conn.commit()
    conn.close()

    monkeypatch.setattr(sql_mode, "_QUERY_TIMEOUT", 0.2)
    # 1000^3 = 1e9 rows scanned — impossible within 0.2s, must be interrupted.
    result = query_sqlite_db(
        str(db), "explode", FakeLLM("SELECT count(*) FROM t a, t b, t c")
    )
    assert "timeout" in result.lower()


# ── Bad DB path ───────────────────────────────────────────────────────────────

def test_invalid_db_path():
    result = query_sqlite_db("/nonexistent/path/db.sqlite", "list all", FakeLLM("SELECT 1"))
    assert any(word in result.lower() for word in ["error", "failed", "invalid", "not found", "no such"])
