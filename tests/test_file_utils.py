"""Tests for directory-scanning security guards."""

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
