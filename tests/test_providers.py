import sys
import types
from unittest.mock import MagicMock

from providers import (
    PROVIDERS,
    build_chat_llm,
    format_model_label,
    provider_available,
    supports_vision,
)

# ── Registry integrity ────────────────────────────────────────────────────────

def test_registry_has_required_fields():
    for name, conf in PROVIDERS.items():
        for key in ("type", "needs_key", "needs_base_url", "models", "vision_models"):
            assert key in conf, f"{name} missing {key}"


def test_ollama_needs_no_key_cloud_does():
    assert PROVIDERS["Ollama (Local)"]["needs_key"] is False
    assert PROVIDERS["OpenAI (ChatGPT)"]["needs_key"] is True
    assert PROVIDERS["Anthropic (Claude)"]["needs_key"] is True
    assert PROVIDERS["Google (Gemini)"]["needs_key"] is True


def test_custom_provider_needs_base_url():
    assert PROVIDERS["Custom (OpenAI-compatible)"]["needs_base_url"] is True


# ── Catalogue label rendering ─────────────────────────────────────────────────

def test_format_model_label_stars_and_tags():
    label = format_model_label({"id": "gpt-4o", "stars": 5, "tags": ["Flagship", "Vision"]})
    assert "gpt-4o" in label
    assert "★★★★★" in label
    assert "Flagship" in label and "Vision" in label


def test_format_model_label_partial_stars():
    label = format_model_label({"id": "x", "stars": 3, "tags": []})
    assert label.count("★") == 3
    assert label.count("☆") == 2


# ── Availability / capability helpers ─────────────────────────────────────────

def test_ollama_always_available():
    assert provider_available("Ollama (Local)") is True


def test_unknown_provider_not_available():
    assert provider_available("Nope") is False


def test_supports_vision():
    assert supports_vision("openai")
    assert supports_vision("ollama")
    assert supports_vision("anthropic")
    assert not supports_vision("totally-unknown")


# ── Factory dispatch ──────────────────────────────────────────────────────────

def test_build_ollama_returns_llm():
    # conftest stubs langchain_ollama.OllamaLLM as a MagicMock class
    llm = build_chat_llm("ollama", "qwen3:4b")
    assert llm is not None


def test_build_cloud_without_key_returns_none():
    assert build_chat_llm("openai", "gpt-4o", api_key=None) is None
    assert build_chat_llm("anthropic", "claude-3-5-sonnet-latest", api_key="") is None
    assert build_chat_llm("google", "gemini-2.0-flash", api_key=None) is None


def test_unknown_provider_returns_none():
    assert build_chat_llm("does-not-exist", "m", api_key="k") is None


def _install_fake_openai(monkeypatch, captured):
    """Inject a fake langchain_openai + StrOutputParser supporting the | pipe."""
    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __or__(self, other):
            return ("chain", self, other)

    mod = types.ModuleType("langchain_openai")
    mod.ChatOpenAI = FakeChat
    monkeypatch.setitem(sys.modules, "langchain_openai", mod)

    parser_mod = types.ModuleType("langchain_core.output_parsers")
    parser_mod.StrOutputParser = lambda: MagicMock(name="StrOutputParser")
    monkeypatch.setitem(sys.modules, "langchain_core.output_parsers", parser_mod)


def test_build_openai_passes_model_and_key(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)

    llm = build_chat_llm("openai", "gpt-4o", api_key="sk-test123")
    assert llm is not None
    assert captured["model"] == "gpt-4o"
    assert captured["api_key"] == "sk-test123"
    # streaming off when no callback supplied
    assert captured["streaming"] is False


def test_build_custom_openai_wires_base_url(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)

    llm = build_chat_llm(
        "custom_openai", "local-model", api_key="x",
        base_url="http://localhost:1234/v1",
    )
    assert llm is not None
    assert captured["base_url"] == "http://localhost:1234/v1"


def test_callback_enables_streaming(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, captured)

    build_chat_llm("openai", "gpt-4o", api_key="k", callback=MagicMock())
    assert captured["streaming"] is True
    assert len(captured["callbacks"]) == 1


def test_build_failure_is_swallowed(monkeypatch):
    """An exception while constructing the model yields None, not a crash."""
    def boom(**_kw):
        raise RuntimeError("bad config")

    mod = types.ModuleType("langchain_openai")
    mod.ChatOpenAI = boom
    monkeypatch.setitem(sys.modules, "langchain_openai", mod)

    assert build_chat_llm("openai", "gpt-4o", api_key="k") is None


# ── models.json catalogue overrides ───────────────────────────────────────────

def test_catalogue_override_replaces_models(monkeypatch):
    import providers
    monkeypatch.setitem(
        providers.PROVIDERS, "OpenAI (ChatGPT)",
        dict(providers.PROVIDERS["OpenAI (ChatGPT)"]),
    )
    providers.apply_catalogue_overrides({
        "OpenAI (ChatGPT)": {
            "models": [
                {"id": "gpt-5", "stars": 5, "tags": ["New"]},
                "gpt-4o-mini",  # plain string → defaults
            ],
        },
    })
    models = providers.PROVIDERS["OpenAI (ChatGPT)"]["models"]
    assert [m["id"] for m in models] == ["gpt-5", "gpt-4o-mini"]
    assert models[0]["stars"] == 5 and models[0]["tags"] == ["New"]
    assert models[1]["stars"] == 3 and models[1]["tags"] == []


def test_catalogue_override_skips_unknown_and_malformed(monkeypatch):
    import providers
    before = providers.PROVIDERS["Google (Gemini)"]["models"]
    providers.apply_catalogue_overrides({
        "Nonexistent Provider": {"models": ["x"]},
        "Google (Gemini)": "not-a-dict",
    })
    assert providers.PROVIDERS["Google (Gemini)"]["models"] == before


def test_catalogue_override_drops_bad_entries(monkeypatch):
    import providers
    monkeypatch.setitem(
        providers.PROVIDERS, "Anthropic (Claude)",
        dict(providers.PROVIDERS["Anthropic (Claude)"]),
    )
    providers.apply_catalogue_overrides({
        "Anthropic (Claude)": {
            "models": [{"no_id": True}, 42, {"id": "claude-x", "stars": 99}],
        },
    })
    models = providers.PROVIDERS["Anthropic (Claude)"]["models"]
    assert [m["id"] for m in models] == ["claude-x"]
    assert models[0]["stars"] == 3  # out-of-range stars clamped to default


def test_load_catalogue_from_file(tmp_path, monkeypatch):
    import json as _json

    import providers
    monkeypatch.setitem(
        providers.PROVIDERS, "OpenAI (ChatGPT)",
        dict(providers.PROVIDERS["OpenAI (ChatGPT)"]),
    )
    mf = tmp_path / "models.json"
    mf.write_text(_json.dumps({"OpenAI (ChatGPT)": {"models": ["my-model"]}}))
    monkeypatch.setattr(providers, "MODELS_FILE", str(mf))

    providers._load_catalogue_overrides()
    assert providers.PROVIDERS["OpenAI (ChatGPT)"]["models"][0]["id"] == "my-model"


def test_invalid_models_file_is_ignored(tmp_path, monkeypatch):
    import providers
    mf = tmp_path / "models.json"
    mf.write_text("{invalid json")
    monkeypatch.setattr(providers, "MODELS_FILE", str(mf))
    before = {k: v["models"] for k, v in providers.PROVIDERS.items()}

    providers._load_catalogue_overrides()  # must not raise
    assert {k: v["models"] for k, v in providers.PROVIDERS.items()} == before
