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
from rag_core import (  # noqa: E402
    fuzzy_match_filenames,
    get_loaded_documents,
    list_ollama_models,
    parse_file_mentions,
)

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
    # rag_core now uses the public Chroma API (vectorstore.get), not the
    # private _collection.get.
    vs.get.return_value = {"metadatas": metadatas}
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
    vs.get.side_effect = Exception("boom")
    assert get_loaded_documents(vs) == []


# ── parse_file_mentions ───────────────────────────────────────────────────────

def test_parse_plain_mention():
    mentions, clean = parse_file_mentions("explain @foo.py please")
    assert mentions == ["foo.py"]
    assert clean == "explain please"


def test_parse_quoted_mention_with_spaces():
    mentions, clean = parse_file_mentions('look at @"my file.py" now')
    assert mentions == ["my file.py"]
    assert clean == "look at now"


def test_parse_multiple_mentions():
    mentions, clean = parse_file_mentions("@a.py and @b.md")
    assert mentions == ["a.py", "b.md"]
    assert clean == "and"


def test_parse_no_mentions_returns_original_query():
    mentions, clean = parse_file_mentions("just a normal question")
    assert mentions == []
    assert clean == "just a normal question"


def test_parse_only_mention_falls_back_to_original_query():
    # When stripping mentions would leave an empty query, the original is kept
    # so the retriever still has something to search on.
    mentions, clean = parse_file_mentions("@foo.py")
    assert mentions == ["foo.py"]
    assert clean == "@foo.py"


def test_parse_ignores_email_addresses():
    # foo@bar.com must not be read as a @bar.com file mention.
    mentions, clean = parse_file_mentions("send it to foo@bar.com")
    assert mentions == []
    assert "foo@bar.com" in clean


# ── fuzzy_match_filenames ─────────────────────────────────────────────────────

def test_fuzzy_exact_basename_match():
    matched = fuzzy_match_filenames(["main.py"], ["/src/main.py", "/src/util.py"])
    assert matched == ["/src/main.py"]


def test_fuzzy_case_insensitive_and_partial():
    matched = fuzzy_match_filenames(["MAIN"], ["/src/main.py"])
    assert matched == ["/src/main.py"]


def test_fuzzy_match_without_extension():
    matched = fuzzy_match_filenames(["util"], ["/src/util.py"])
    assert matched == ["/src/util.py"]


def test_fuzzy_no_match_returns_empty():
    assert fuzzy_match_filenames(["missing.py"], ["/src/main.py"]) == []


def test_fuzzy_dedups_across_mentions():
    matched = fuzzy_match_filenames(["main", "main.py"], ["/src/main.py"])
    assert matched == ["/src/main.py"]


# ── BytesWrapper (moved from omniscience_pro.py) ──────────────────────────────

def test_bytes_wrapper_roundtrip():
    from vision import BytesWrapper
    data = b"\x89PNG\r\n"
    wrapper = BytesWrapper(data)
    assert wrapper.getvalue() == data
