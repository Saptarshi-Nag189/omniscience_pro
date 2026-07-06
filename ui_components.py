"""UI building blocks: CSS theme, thinking indicator, stream handler, clipboard helper."""
import json
import logging
from typing import List

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

# ── CSS theme ─────────────────────────────────────────────────────────────────
PURPLE_THEME_CSS = """
<style>
    /* System font stack — no external font fetch, keeping the app fully offline. */
    .stApp { background-color: #0f0f12; color: #e0e0e0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; }
    [data-testid="stSidebar"] { background-color: #15151a; border-right: 1px solid #2d2d33; }
    h1, h2, h3, h4 { color: #ffffff !important; font-weight: 600; letter-spacing: -0.5px; }

    .custom-title {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #a78bfa, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e1e24 !important; color: #ffffff !important; border: 1px solid #333 !important; border-radius: 6px;
    }

    .stButton button {
        background-color: #7c3aed !important; color: #ffffff !important; border: none !important; border-radius: 6px; font-weight: 600;
    }
    .stButton button:hover { background-color: #6d28d9 !important; color: #ffffff !important; }

    div[data-testid="column"] + div[data-testid="column"] .stButton button, .stButton.delete-btn button {
        background-color: #2d2d2d !important; border: 1px solid #444 !important;
    }

    [data-testid="stChatMessage"] { background-color: #1e1e24; border: 1px solid #2d2d33; border-radius: 8px; }
    div[data-testid="chatAvatarIcon-user"] { background-color: #2d2d33 !important; }
    div[data-testid="chatAvatarIcon-assistant"] { background-color: #7c3aed !important; }

    code { background-color: #000000 !important; color: #a78bfa !important; border: 1px solid #333; border-radius: 4px; }

    [data-testid="collapsedControl"] {
        display: block !important;
        color: #ffffff !important;
        background-color: #15151a !important;
        border-radius: 50%;
        border: 1px solid #333;
    }

    @keyframes pulse-border {
        0%, 100% { border-color: #7c3aed; box-shadow: 0 0 5px rgba(124, 58, 237, 0.3); }
        50% { border-color: #a78bfa; box-shadow: 0 0 15px rgba(167, 139, 250, 0.5); }
    }

    .thinking-box {
        animation: pulse-border 1.5s ease-in-out infinite;
        border: 2px solid #7c3aed;
        border-radius: 8px;
        padding: 12px 16px;
        background-color: #1e1e24;
        margin: 8px 0;
    }

    @keyframes dot-pulse {
        0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
        40% { opacity: 1; transform: scale(1); }
    }

    .thinking-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .thinking-dots span {
        width: 8px;
        height: 8px;
        background-color: #a78bfa;
        border-radius: 50%;
        animation: dot-pulse 1.4s ease-in-out infinite;
    }

    .thinking-dots span:nth-child(1) { animation-delay: 0s; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

    .thinking-text {
        color: #a78bfa;
        font-size: 0.9rem;
        margin-left: 8px;
    }

    @keyframes vision-pulse {
        0%, 100% {
            background-color: #7c3aed !important;
            box-shadow: 0 0 5px rgba(124, 58, 237, 0.5);
            transform: scale(1);
        }
        50% {
            background-color: #a78bfa !important;
            box-shadow: 0 0 20px rgba(167, 139, 250, 0.8);
            transform: scale(1.05);
        }
    }

    .vision-highlight {
        animation: vision-pulse 0.5s ease-in-out 3;
    }

    .screen-dim {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 999;
        pointer-events: none;
        animation: fade-out 2s forwards;
    }

    @keyframes fade-out {
        0% { opacity: 1; }
        70% { opacity: 1; }
        100% { opacity: 0; }
    }

    #MainMenu {visibility: visible;} footer {visibility: hidden;} header {visibility: visible;}
</style>
"""

VISION_PULSE_JS = ""
SQL_PULSE_JS = ""

# ── Thinking indicator ────────────────────────────────────────────────────────
THINKING_HTML = """
<div class="thinking-box">
    <div class="thinking-dots">
        <span></span><span></span><span></span>
    </div>
    <span class="thinking-text">Thinking...</span>
</div>
"""


# ── Streaming callback ────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    """LangChain callback that streams tokens directly to a Streamlit placeholder."""

    def __init__(self, container, initial_text="", thinking_placeholder=None):
        self.container = container
        self.text = initial_text
        self.thinking_placeholder = thinking_placeholder
        self.first_token = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if st.session_state.get('stop_generation', False):
            st.session_state.stop_generation = False
            raise StopIteration("Generation stopped by user")
        if self.first_token and self.thinking_placeholder:
            self.thinking_placeholder.empty()
            self.first_token = False
        self.text += token
        self.container.markdown(self.text + "▌")


# ── Startup marker ────────────────────────────────────────────────────────────
@st.cache_resource
def _get_startup_marker():
    """Cached startup ID — changes on fresh starts, stable across browser refreshes."""
    import time
    return {"startup_time": time.time(), "session_created": False}


# ── Conversation history builder ─────────────────────────────────────────────

def build_conversation_history(messages: list) -> List[str]:
    """Return ['USER: ...', 'ASSISTANT: ...'] strings for the last HISTORY_WINDOW messages."""
    from config import HISTORY_EXCERPT_LENGTH, HISTORY_WINDOW
    window = messages[-HISTORY_WINDOW:] if len(messages) > HISTORY_WINDOW else messages
    parts = []
    for msg in window:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        content = msg["content"]
        if len(content) > HISTORY_EXCERPT_LENGTH:
            content = content[:HISTORY_EXCERPT_LENGTH] + "..."
        parts.append(f"{role}: {content}")
    return parts


# ── Clipboard helper ──────────────────────────────────────────────────────────

def _js_string(value: str) -> str:
    """JSON-encode a string for safe embedding inside a <script> block.

    Escapes '<', '>' and '&' so content like '</script>' or HTML tags cannot
    break out of the script context, regardless of what the LLM produced.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def copy_to_clipboard(text: str, label: str = "Copy Response") -> None:
    """Render a one-click copy-to-clipboard button using the browser JS clipboard API."""
    components.html(
        f"""<button id="copy-btn"
            style="background:#4a4a8a;color:white;border:none;padding:6px 14px;
                   border-radius:6px;cursor:pointer;font-size:13px;"></button>
<script>
    var TEXT = {_js_string(text)};
    var LABEL = {_js_string(label)};
    var btn = document.getElementById('copy-btn');
    btn.textContent = LABEL;
    btn.addEventListener('click', function () {{
        navigator.clipboard.writeText(TEXT).then(function () {{
            btn.textContent = 'Copied!';
            setTimeout(function () {{ btn.textContent = LABEL; }}, 2000);
        }});
    }});
</script>""",
        height=42,
    )
