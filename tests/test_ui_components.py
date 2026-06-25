import sys
from unittest.mock import MagicMock

# Stub Streamlit and LangChain before importing ui_components
for mod in ["streamlit", "streamlit.components", "streamlit.components.v1"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
if "langchain_core.callbacks.base" not in sys.modules:
    sys.modules["langchain_core"] = MagicMock()
    sys.modules["langchain_core.callbacks"] = MagicMock()
    sys.modules["langchain_core.callbacks.base"] = MagicMock()

import ui_components  # noqa: E402
from ui_components import build_conversation_history  # noqa: E402

# ── build_conversation_history ────────────────────────────────────────────────

def test_empty_messages_returns_empty_list():
    assert build_conversation_history([]) == []


def test_formats_user_and_assistant_roles():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    parts = build_conversation_history(messages)
    assert parts[0] == "USER: Hello"
    assert parts[1] == "ASSISTANT: Hi there"


def test_truncates_long_content(monkeypatch):
    monkeypatch.setattr(ui_components, "build_conversation_history", build_conversation_history)
    import config
    monkeypatch.setattr(config, "HISTORY_EXCERPT_LENGTH", 10)
    long_msg = [{"role": "user", "content": "A" * 20}]
    parts = build_conversation_history(long_msg)
    assert parts[0] == "USER: AAAAAAAAAA..."


def test_short_content_not_truncated(monkeypatch):
    import config
    monkeypatch.setattr(config, "HISTORY_EXCERPT_LENGTH", 100)
    msg = [{"role": "user", "content": "short"}]
    parts = build_conversation_history(msg)
    assert "..." not in parts[0]


def test_respects_history_window(monkeypatch):
    import config
    monkeypatch.setattr(config, "HISTORY_WINDOW", 3)
    monkeypatch.setattr(config, "HISTORY_EXCERPT_LENGTH", 500)
    messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    parts = build_conversation_history(messages)
    assert len(parts) == 3
    assert "msg7" in parts[0]
    assert "msg9" in parts[2]
