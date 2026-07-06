"""Vision mode: multimodal image analysis via Ollama or cloud providers."""
import base64
import logging

from config import OLLAMA_BASE_URL
from security import sanitize_error_message

logger = logging.getLogger(__name__)


class BytesWrapper:
    """Adapt raw bytes to the file-like interface expected by process_vision_request."""
    def __init__(self, b: bytes):
        self.b = b

    def getvalue(self) -> bytes:
        return self.b


def _guess_mime(image_bytes: bytes) -> str:
    """Best-effort image MIME detection from magic bytes; defaults to PNG."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _ollama_vision(base64_image: str, prompt: str, model_name: str) -> str:
    """Send an image + prompt to Ollama's multimodal endpoint."""
    import requests

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
    }
    # Cap the request so a stalled vision model can't hang the UI indefinitely.
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    if response.status_code == 200:
        return response.json().get("response", "No response from vision model.")
    return "Vision Error: Unable to process image."


def _cloud_vision(provider_type: str, base64_image: str, mime: str, prompt: str,
                  model_name: str, api_key: str, base_url: str = None) -> str:
    """Analyse an image via a cloud chat model using LangChain's multimodal format."""
    from langchain_core.messages import HumanMessage

    # For image input we need the raw chat model (multimodal message content),
    # not the str-parsed chain that build_chat_llm returns for text chat.
    chat = _raw_chat_model(provider_type, model_name, api_key, base_url)
    if chat is None:
        return f"Vision Error: provider '{provider_type}' is unavailable or misconfigured."

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
    ])
    result = chat.invoke([message])
    return getattr(result, "content", str(result))


def _raw_chat_model(provider_type: str, model: str, api_key: str, base_url: str = None):
    """Construct a raw (unparsed) chat model for multimodal message input."""
    try:
        if provider_type in ("openai", "custom_openai"):
            from langchain_openai import ChatOpenAI
            kwargs = dict(model=model, temperature=0.2, api_key=api_key)
            if base_url:
                kwargs["base_url"] = base_url
            return ChatOpenAI(**kwargs)
        if provider_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=0.2, api_key=api_key)
        if provider_type == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=0.2, google_api_key=api_key)
    except Exception as e:
        from security import redact_secrets
        logger.warning(f"Failed to build vision model for {provider_type}: {redact_secrets(e)}")
    return None


def process_vision_request(image_file, prompt: str, model_name: str = "llava",
                           provider_type: str = "ollama", api_key: str = None,
                           base_url: str = None) -> str:
    """Analyse an image with a prompt, dispatching to the selected provider.

    Defaults to the local Ollama backend so existing call sites keep working.
    """
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        if provider_type == "ollama":
            return _ollama_vision(base64_image, prompt, model_name)

        if not api_key:
            return f"Vision Error: {provider_type} requires an API key."

        mime = _guess_mime(image_bytes)
        return _cloud_vision(provider_type, base64_image, mime, prompt,
                             model_name, api_key, base_url)
    except Exception as e:
        return f"Connection Error: {sanitize_error_message(e)}"
