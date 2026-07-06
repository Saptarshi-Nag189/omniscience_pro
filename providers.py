"""LLM provider abstraction.

Supports the local Ollama backend (default, no API key) plus optional cloud
providers — OpenAI (ChatGPT), Anthropic (Claude), Google (Gemini) — and any
OpenAI-compatible custom endpoint. Cloud SDKs are imported lazily so an
Ollama-only install keeps working with no extra dependencies.

All factory functions return an object whose ``.invoke(prompt) -> str`` and
streaming-via-callback behaviour is uniform across backends, so the rest of the
app (chat, SQL, academic extraction) needs no per-provider branching.
"""
import importlib.util
import logging
from typing import Optional

from config import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


# ── Optional dependency detection ─────────────────────────────────────────────

def _installed(module: str) -> bool:
    """True if an import package is importable, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_OPENAI = _installed("langchain_openai")
HAS_ANTHROPIC = _installed("langchain_anthropic")
HAS_GOOGLE = _installed("langchain_google_genai")


# ── Model catalogue ───────────────────────────────────────────────────────────
# Each model: {"id", "stars" (1-5), "tags" [..]}. `id` is the value sent to the
# provider; the catalogue is advisory — users may type any model name instead.

def _m(model_id: str, stars: int, *tags: str) -> dict:
    return {"id": model_id, "stars": stars, "tags": list(tags)}


PROVIDERS = {
    "Ollama (Local)": {
        "type": "ollama",
        "needs_key": False,
        "needs_base_url": False,
        "pip": None,
        "env": None,
        "available": True,  # always available; the local server may still be down
        "models": [
            _m("qwen2.5-coder:7b", 4, "Coding", "Default"),
            _m("qwen3:4b", 4, "Balanced"),
            _m("mistral:7b", 4, "General"),
            _m("llama3.2:3b", 3, "Small", "Fast"),
            _m("qwen2.5-coder:1.5b", 3, "Tiny"),
        ],
        "vision_models": [
            _m("llama3.2-vision", 4, "Vision"),
            _m("llava:7b", 4, "Vision", "Light"),
        ],
    },
    "OpenAI (ChatGPT)": {
        "type": "openai",
        "needs_key": True,
        "needs_base_url": False,
        "pip": "langchain-openai",
        "env": "OPENAI_API_KEY",
        "available": HAS_OPENAI,
        "models": [
            _m("gpt-4o", 5, "Flagship", "Vision"),
            _m("gpt-4o-mini", 4, "Fast", "Cheap", "Vision"),
            _m("gpt-4-turbo", 4, "Vision"),
            _m("o1-mini", 4, "Reasoning"),
            _m("gpt-3.5-turbo", 3, "Legacy", "Cheap"),
        ],
        "vision_models": [
            _m("gpt-4o", 5, "Vision", "Flagship"),
            _m("gpt-4o-mini", 4, "Vision", "Cheap"),
        ],
    },
    "Anthropic (Claude)": {
        "type": "anthropic",
        "needs_key": True,
        "needs_base_url": False,
        "pip": "langchain-anthropic",
        "env": "ANTHROPIC_API_KEY",
        "available": HAS_ANTHROPIC,
        "models": [
            _m("claude-3-5-sonnet-latest", 5, "Recommended", "Coding", "Vision"),
            _m("claude-3-5-haiku-latest", 4, "Fast", "Cheap"),
            _m("claude-3-opus-latest", 4, "Deep reasoning", "Vision"),
        ],
        "vision_models": [
            _m("claude-3-5-sonnet-latest", 5, "Vision", "Recommended"),
            _m("claude-3-opus-latest", 4, "Vision"),
        ],
    },
    "Google (Gemini)": {
        "type": "google",
        "needs_key": True,
        "needs_base_url": False,
        "pip": "langchain-google-genai",
        "env": "GOOGLE_API_KEY",
        "available": HAS_GOOGLE,
        "models": [
            _m("gemini-2.0-flash", 5, "Recommended", "Fast", "Vision"),
            _m("gemini-1.5-pro", 5, "Long context", "Vision"),
            _m("gemini-1.5-flash", 4, "Fast", "Cheap", "Vision"),
            _m("gemini-2.0-flash-lite", 3, "Cheapest"),
        ],
        "vision_models": [
            _m("gemini-2.0-flash", 5, "Vision", "Recommended"),
            _m("gemini-1.5-pro", 5, "Vision", "Long context"),
        ],
    },
    "Custom (OpenAI-compatible)": {
        "type": "custom_openai",
        "needs_key": True,
        "needs_base_url": True,
        "pip": "langchain-openai",
        "env": None,
        "available": HAS_OPENAI,
        "models": [],          # user supplies the model name
        "vision_models": [],
    },
}

# Provider types that can analyse images.
_VISION_CAPABLE = {"ollama", "openai", "anthropic", "google", "custom_openai"}


def format_model_label(meta: dict) -> str:
    """Render a catalogue entry as 'id  ★★★★☆ · Tag · Tag' for the selector."""
    stars = meta.get("stars", 0)
    star_str = "★" * stars + "☆" * (5 - stars)
    tags = meta.get("tags", [])
    label = f"{meta['id']}  {star_str}"
    if tags:
        label += "  ·  " + " · ".join(tags)
    return label


def provider_available(provider_name: str) -> bool:
    """True if the provider's SDK is installed (Ollama is always 'available')."""
    conf = PROVIDERS.get(provider_name)
    return bool(conf and conf.get("available"))


def supports_vision(provider_type: str) -> bool:
    return provider_type in _VISION_CAPABLE


# ── Chat LLM factory ──────────────────────────────────────────────────────────

def build_chat_llm(provider_type: str, model: str, api_key: Optional[str] = None,
                   base_url: Optional[str] = None, callback=None,
                   temperature: float = 0.2):
    """Build a chat LLM whose ``.invoke(prompt)`` returns a plain ``str``.

    Returns ``None`` if the backend can't be constructed (missing package,
    missing key, bad config) — callers already guard on a falsy LLM.
    """
    callbacks = [callback] if callback else []
    streaming = bool(callbacks)

    try:
        if provider_type == "ollama":
            from langchain_ollama import OllamaLLM
            # Completion model already yields str from .invoke — return as-is to
            # preserve the exact legacy behaviour and streaming semantics.
            return OllamaLLM(
                model=model, temperature=temperature, base_url=OLLAMA_BASE_URL,
                callbacks=callbacks, streaming=streaming,
            )

        if provider_type in ("openai", "custom_openai"):
            if not api_key:
                logger.warning("OpenAI-compatible provider requires an API key")
                return None
            from langchain_openai import ChatOpenAI
            kwargs = dict(
                model=model, temperature=temperature, api_key=api_key,
                callbacks=callbacks, streaming=streaming,
            )
            if base_url:
                kwargs["base_url"] = base_url
            chat = ChatOpenAI(**kwargs)

        elif provider_type == "anthropic":
            if not api_key:
                logger.warning("Anthropic provider requires an API key")
                return None
            from langchain_anthropic import ChatAnthropic
            chat = ChatAnthropic(
                model=model, temperature=temperature, api_key=api_key,
                callbacks=callbacks, streaming=streaming,
            )

        elif provider_type == "google":
            if not api_key:
                logger.warning("Google provider requires an API key")
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI
            chat = ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=api_key,
                callbacks=callbacks, streaming=streaming,
            )
        else:
            logger.warning(f"Unknown provider type: {provider_type}")
            return None

        # Chat models return an AIMessage from .invoke(); normalise to str so
        # downstream code (chat render, SQL .strip(), academic extraction) is
        # backend-agnostic. Callbacks still fire on the underlying model.
        from langchain_core.output_parsers import StrOutputParser
        return chat | StrOutputParser()

    except Exception as e:
        from security import redact_secrets
        logger.warning(f"Failed to build LLM for {provider_type}: {redact_secrets(e)}")
        return None
