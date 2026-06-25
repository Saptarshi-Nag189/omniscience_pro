import json
import sys
from unittest.mock import MagicMock

# Stub heavy deps before any import from rag_core
for mod in ["streamlit", "chromadb", "chromadb.config"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

for mod in [
    "langchain_huggingface",
    "langchain_chroma",
    "langchain_ollama",
    "langchain_core",
    "langchain_core.documents",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import rag_core  # noqa: E402
from rag_core import get_loaded_documents, list_ollama_models  # noqa: E402

# ── list_ollama_models ────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_list_ollama_models_returns_names(monkeypatch):
    payload = {"models": [{"name": "qwen3:4b"}, {"name": "llava:7b"}]}
    monkeypatch.setattr(
        rag_core.urllib.request, "urlopen",
        lambda req, timeout=2: _FakeResp(payload),
    )
    models = list_ollama_models()
    assert "qwen3:4b" in models
    assert "llava:7b" in models


def test_list_ollama_models_empty_on_error(monkeypatch):
    def _fail(*_a, **_kw):
        raise OSError("connection refused")
    monkeypatch.setattr(rag_core.urllib.request, "urlopen", _fail)
    assert list_ollama_models() == []


def test_list_ollama_models_empty_list_when_no_models(monkeypatch):
    monkeypatch.setattr(
        rag_core.urllib.request, "urlopen",
        lambda req, timeout=2: _FakeResp({"models": []}),
    )
    assert list_ollama_models() == []


# ── get_loaded_documents ──────────────────────────────────────────────────────

def _fake_vs(metadatas):
    vs = MagicMock()
    vs._collection.get.return_value = {"metadatas": metadatas}
    return vs


def test_get_loaded_documents_returns_sorted_unique_sources():
    vs = _fake_vs([
        {"filename": "b.py"},
        {"filename": "a.md"},
        {"filename": "b.py"},
    ])
    docs = get_loaded_documents(vs)
    assert docs == ["a.md", "b.py"]


def test_get_loaded_documents_falls_back_to_source_key():
    vs = _fake_vs([{"source": "/home/user/notes.txt"}])
    docs = get_loaded_documents(vs)
    assert docs == ["/home/user/notes.txt"]


def test_get_loaded_documents_returns_empty_on_error():
    vs = MagicMock()
    vs._collection.get.side_effect = Exception("boom")
    assert get_loaded_documents(vs) == []


# ── BytesWrapper (moved from omniscience_pro.py) ──────────────────────────────

def test_bytes_wrapper_roundtrip():
    from vision import BytesWrapper
    data = b"\x89PNG\r\n"
    wrapper = BytesWrapper(data)
    assert wrapper.getvalue() == data
