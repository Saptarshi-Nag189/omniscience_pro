import io
import sys
from unittest.mock import MagicMock

from vision import process_vision_request


def _fake_image():
    return io.BytesIO(b"fake-image-bytes")


def test_vision_request_passes_timeout(monkeypatch):
    """The Ollama POST must include a timeout so a stalled model can't hang forever."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["timeout"] = timeout
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"response": "a cat"}
        return resp

    monkeypatch.setattr(sys.modules["requests"], "post", fake_post, raising=False)

    result = process_vision_request(_fake_image(), "what is this?")

    assert captured["timeout"] is not None
    assert captured["timeout"] > 0
    assert result == "a cat"


def test_vision_request_handles_connection_error(monkeypatch):
    """A network failure is caught and surfaced as a sanitized error string."""
    def boom(*_a, **_kw):
        raise OSError("connection refused")

    monkeypatch.setattr(sys.modules["requests"], "post", boom, raising=False)

    result = process_vision_request(_fake_image(), "what is this?")
    assert "Connection Error" in result
