"""Vision mode: multimodal image analysis via Ollama."""
import base64
import logging

from config import OLLAMA_BASE_URL
from security import sanitize_error_message

logger = logging.getLogger(__name__)


def process_vision_request(image_file, prompt: str, model_name: str = "llava") -> str:
    """Send an image + prompt to Ollama's multimodal endpoint and return the response."""
    import requests

    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
    }

    try:
        # Cap the request so a stalled vision model can't hang the UI indefinitely.
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "No response from vision model.")
        return "Vision Error: Unable to process image."
    except Exception as e:
        return f"Connection Error: {sanitize_error_message(e)}"
