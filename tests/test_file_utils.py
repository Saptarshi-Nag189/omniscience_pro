"""Tests for directory-scanning security guards."""

from unittest.mock import MagicMock

import file_utils


class _FakeSplitter:
    def split_text(self, content):
        return [content]


class _FakeProgress:
    def progress(self, *_a):
        pass

    def empty(self):
        pass


def _capture_warnings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        file_utils.st, "warning", lambda msg: calls.append(str(msg)), raising=False
    )
    monkeypatch.setattr(
        file_utils.st, "progress", lambda *_a: _FakeProgress(), raising=False
    )
    return calls


def test_scan_blocks_filesystem_root(monkeypatch):
    """Scanning '/' would descend into every system directory — must be blocked."""
    calls = _capture_warnings(monkeypatch)
    assert file_utils.scan_directory("/") == []
    assert any("system directories" in m for m in calls)


def test_scan_blocks_sensitive_directory(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    assert file_utils.scan_directory("/etc") == []
    assert any("system directories" in m for m in calls)


def test_scan_blocks_subdir_of_sensitive(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    assert file_utils.scan_directory("/etc/ssl") == []
    assert any("system directories" in m for m in calls)


def test_scan_allows_normal_directory(tmp_path, monkeypatch):
    calls = _capture_warnings(monkeypatch)
    monkeypatch.setattr(file_utils, "get_text_splitter", lambda ext: _FakeSplitter())
    (tmp_path / "a.py").write_text("print('hi')")

    file_utils.scan_directory(str(tmp_path))
    assert not any("system directories" in m for m in calls)


def test_scan_empty_path_warns(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    assert file_utils.scan_directory("  ") == []
    assert calls  # warned about invalid path


# ── Upload retention cleanup ─────────────────────────────────────────────────

def _make_upload(dirpath, name, age_hours):
    import os
    import time
    p = dirpath / name
    p.write_text("data")
    past = time.time() - age_hours * 3600
    os.utime(p, (past, past))
    return p


def test_cleanup_removes_old_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(file_utils, "UPLOAD_RETENTION_HOURS", 24)

    old = _make_upload(tmp_path, "old.pdf", age_hours=48)
    fresh = _make_upload(tmp_path, "fresh.txt", age_hours=1)

    removed = file_utils.cleanup_old_uploads()
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_handles_missing_dir(monkeypatch):
    monkeypatch.setattr(file_utils, "UPLOAD_DIR", "/nonexistent/upload/dir")
    assert file_utils.cleanup_old_uploads() == 0


# ── read_file_content ─────────────────────────────────────────────────────────

def test_read_utf8_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    assert file_utils.read_file_content(p) == "hello world"


def test_read_latin1_fallback(tmp_path):
    # bytes that are invalid UTF-8 but valid latin-1 — must not be dropped.
    p = tmp_path / "b.txt"
    p.write_bytes("café".encode("latin-1"))
    assert file_utils.read_file_content(p) == "café"


def test_read_oversize_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils, "MAX_FILE_SIZE_MB", 0.000001)  # ~1 byte cap
    p = tmp_path / "big.txt"
    p.write_text("x" * 100)
    assert file_utils.read_file_content(p) == ""


def test_read_missing_file_returns_empty(tmp_path):
    assert file_utils.read_file_content(tmp_path / "nope.txt") == ""


# ── get_text_splitter ─────────────────────────────────────────────────────────

def test_get_text_splitter_uses_language_for_code(monkeypatch):
    splitter_cls = MagicMock()
    monkeypatch.setattr(file_utils, "RecursiveCharacterTextSplitter", splitter_cls)
    file_utils.get_text_splitter(".py")
    assert splitter_cls.from_language.called


def test_get_text_splitter_default_for_unknown(monkeypatch):
    splitter_cls = MagicMock()
    monkeypatch.setattr(file_utils, "RecursiveCharacterTextSplitter", splitter_cls)
    file_utils.get_text_splitter(".xyz")
    assert not splitter_cls.from_language.called
    assert splitter_cls.called  # fell back to the plain constructor
