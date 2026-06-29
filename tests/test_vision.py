import io
import sys
import types
from unittest.mock import MagicMock

from vision import _guess_mime, process_vision_request


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


# ── MIME detection ────────────────────────────────────────────────────────────

def test_guess_mime_png():
    assert _guess_mime(b"\x89PNG\r\n\x1a\n" + b"rest") == "image/png"


def test_guess_mime_jpeg():
    assert _guess_mime(b"\xff\xd8\xff\xe0" + b"rest") == "image/jpeg"


def test_guess_mime_defaults_png():
    assert _guess_mime(b"unknown-bytes") == "image/png"


# ── Cloud provider dispatch ───────────────────────────────────────────────────

def test_cloud_provider_requires_key():
    result = process_vision_request(
        _fake_image(), "describe", provider_type="openai", api_key=None,
    )
    assert "requires an API key" in result


def test_cloud_provider_invokes_chat_model(monkeypatch):
    """A cloud vision request builds a chat model and returns its content."""
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return types.SimpleNamespace(content="a sunset")

    openai_mod = types.ModuleType("langchain_openai")
    openai_mod.ChatOpenAI = FakeChat
    monkeypatch.setitem(sys.modules, "langchain_openai", openai_mod)

    msgs_mod = types.ModuleType("langchain_core.messages")
    msgs_mod.HumanMessage = lambda content: types.SimpleNamespace(content=content)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", msgs_mod)

    result = process_vision_request(
        _fake_image(), "describe", provider_type="openai",
        model_name="gpt-4o", api_key="sk-test",
    )
    assert result == "a sunset"
    assert captured["kwargs"]["model"] == "gpt-4o"
