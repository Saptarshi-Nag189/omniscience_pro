"""
Omniscience Pro - Local RAG System
Theme: Dark/Purple (Clean, Professional, No Glow, No Emojis)
Features:
- Multi-Threaded Chat History (Sessions) with Auto-Titles
- Vision Support (Analyze Images)
- TRUE Frontend Streaming
- Drag & Drop File Upload
- Local Folder Scanning
- Hybrid Intelligence
- File Management
- SQL & Web Search
- Code Interpreter (Execute Python)

Key Features:
- Fixed JSON serialization for image bytes
- Fixed deprecated `use_column_width` -> `use_container_width`
- Added matplotlib cache fix for temp directory
- Improved session management with titles
- Fixed SQL mode implementation
- Removed all emojis
- Added proper error handling
- Added file size limits
"""

import os
import shutil
import json
import time
import base64
import uuid
import tempfile
import re
import logging
import fcntl
from datetime import datetime
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FIX MATPLOTLIB CACHE ISSUE
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks.base import BaseCallbackHandler
import pypdf
import traceback
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

# Try importing DuckDuckGo (Graceful fallback if not installed)
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False

# Academic Search APIs (Semantic Scholar, arXiv, OpenAlex)
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

try:
    from semanticscholar import SemanticScholar
    HAS_SEMANTIC_SCHOLAR = True
except ImportError:
    HAS_SEMANTIC_SCHOLAR = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (Use environment variables with secure defaults)
# ═══════════════════════════════════════════════════════════════════

# Validate and sanitize environment variables
def _get_env_int(key: str, default: int, min_val: int = 1, max_val: int = 100000) -> int:
    """Get integer from environment with validation."""
    try:
        val = int(os.environ.get(key, str(default)))
        return max(min_val, min(val, max_val))
    except (ValueError, TypeError):
        return default

DB_DIRECTORY = os.environ.get('OMNISCIENCE_DB_DIR', './db_omniscience')
CHATS_DIR = os.environ.get('OMNISCIENCE_CHATS_DIR', './chats')
UPLOAD_DIR = os.environ.get('OMNISCIENCE_UPLOAD_DIR', './uploads')
EMBEDDING_MODEL = os.environ.get('OMNISCIENCE_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')

# Security Configuration (with input validation)
MAX_FILE_SIZE_MB = _get_env_int('OMNISCIENCE_MAX_FILE_SIZE_MB', 10, 1, 100)
MAX_FILES_PER_SCAN = _get_env_int('OMNISCIENCE_MAX_FILES_PER_SCAN', 1000, 10, 10000)
MAX_MESSAGES_PER_SESSION = _get_env_int('OMNISCIENCE_MAX_MESSAGES', 100, 10, 500)

# Session Lifecycle Configuration
SESSION_EXPIRY_HOURS = _get_env_int('OMNISCIENCE_SESSION_EXPIRY_HOURS', 24, 1, 168)  # Max 1 week
SESSION_IDLE_TIMEOUT_HOURS = _get_env_int('OMNISCIENCE_IDLE_TIMEOUT_HOURS', 4, 1, 24)
MAX_SESSIONS = _get_env_int('OMNISCIENCE_MAX_SESSIONS', 100, 10, 1000)

# Rate limiting (simple in-memory - resets on restart)
_rate_limit_cache = {}
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# Ensure directories exist with restricted permissions
for directory in [UPLOAD_DIR, CHATS_DIR]:
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(directory, 0o700)  # Owner only
    except OSError:
        pass  # May fail on some systems

# EXTENSION MAP
SUPPORTED_EXTENSIONS = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.cpp': Language.CPP,
    '.c': Language.CPP,
    '.html': Language.HTML,
    '.css': None,
    '.md': Language.MARKDOWN,
    '.txt': None,
    '.pdf': 'pdf'
}

IGNORED_DIRS = {
    'node_modules', 'venv', '.venv', 'env', 'wenv', '.git', '.idea', '.vscode', 
    '__pycache__', 'dist', 'build', 'coverage', 'target', 'bin', 'obj',
    '__MACOSX', '.pytest_cache', '.mypy_cache', '.tox', '*.egg-info',
    'package', 'logs', 'log', '.DS_Store', '.obsidian', 'Temp_Files', 'AWID3_Dataset', 
    'DATASET_AWID2', 'AWID3_Dataset_CSV', 'data', 'wordlists', 'dict', 'datasets', 'dataset'
}

IGNORED_SUFFIXES = ('.egg-info', '.min.js', '.map', '.lock')
IGNORED_FILE_EXTENSIONS = {'.csv', '.pcap', '.pcapng', '.cap', '.log', '.json', '.xml', '.bin', '.dat'}
IGNORED_FILES = {
    'rockyou.txt', 'package-lock.json', 'yarn.lock', 'poetry.lock', 
    '.DS_Store', 'thumbs.db'
}

# ═══════════════════════════════════════════════════════════════════
# SECURITY UTILITIES
# ═══════════════════════════════════════════════════════════════════

def sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal attacks."""
    if not session_id:
        raise ValueError("Session ID cannot be empty")
    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise ValueError(f"Invalid session ID format: {session_id}")
    # Prevent path traversal
    if '..' in session_id or '/' in session_id or '\\' in session_id:
        raise ValueError(f"Invalid session ID: path traversal detected")
    return session_id


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks."""
    if not filename:
        raise ValueError("Filename cannot be empty")
    # Remove path components
    filename = os.path.basename(filename)
    # Remove null bytes
    filename = filename.replace('\x00', '')
    # Remove path traversal sequences
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    # Normalize unicode
    import unicodedata
    filename = unicodedata.normalize('NFKC', filename)
    # Only allow safe characters
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    if not filename or filename in ('.', '..'):
        raise ValueError("Invalid filename after sanitization")
    return filename


def validate_path_within_directory(path: Path, allowed_dir: Path) -> bool:
    """Ensure a path is within an allowed directory (no escape via symlinks)."""
    try:
        # Resolve both paths to absolute, following symlinks
        resolved_path = path.resolve()
        resolved_allowed = allowed_dir.resolve()
        # Check if path is within allowed directory
        return str(resolved_path).startswith(str(resolved_allowed))
    except (OSError, ValueError):
        return False


def check_rate_limit(user_id: str = "default") -> bool:
    """Simple rate limiting. Returns True if request is allowed."""
    current_time = time.time()
    
    if user_id not in _rate_limit_cache:
        _rate_limit_cache[user_id] = []
    
    # Remove old requests outside the window
    _rate_limit_cache[user_id] = [
        t for t in _rate_limit_cache[user_id] 
        if current_time - t < RATE_LIMIT_WINDOW_SECONDS
    ]
    
    if len(_rate_limit_cache[user_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    _rate_limit_cache[user_id].append(current_time)
    return True


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to prevent information disclosure."""
    error_str = str(error)
    # Remove file paths
    error_str = re.sub(r'(/[^\s]+)+', '[PATH]', error_str)
    # Remove potential SQL
    error_str = re.sub(r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)[\s\S]*', '[SQL]', error_str, flags=re.IGNORECASE)
    # Limit length
    if len(error_str) > 200:
        error_str = error_str[:200] + '...'
    return error_str

# ═══════════════════════════════════════════════════════════════════
# CSS THEME
# ═══════════════════════════════════════════════════════════════════

PURPLE_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    .stApp { background-color: #0f0f12; color: #e0e0e0; font-family: 'Inter', sans-serif; }
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
    
    /* Sidebar Toggle Button Fix */
    [data-testid="collapsedControl"] { 
        display: block !important; 
        color: #ffffff !important; 
        background-color: #15151a !important;
        border-radius: 50%;
        border: 1px solid #333;
    }
    
    /* Thinking Animation - Pulsing Border */
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
    
    /* Animated Dots */
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
    
    /* Vision Tab Pulse Animation - triggered via JavaScript */
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
    
    /* Screen dim overlay */
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

# Visual guidance removed - just using warning messages now
VISION_PULSE_JS = ""  # Scroll/highlight was not working reliably
SQL_PULSE_JS = ""  # Keeping only the warning messages

# ═══════════════════════════════════════════════════════════════════
# SESSION & HISTORY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def cleanup_expired_sessions():
    """Remove expired sessions and enforce max session limit.
    
    Call this periodically (e.g., on app startup or via cron).
    """
    if not os.path.exists(CHATS_DIR):
        return 0
    
    removed = 0
    now = datetime.now()
    session_files = []
    
    for f in os.listdir(CHATS_DIR):
        if not f.endswith('.json'):
            continue
        
        path = os.path.join(CHATS_DIR, f)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age_hours = (now - mtime).total_seconds() / 3600
            
            # Remove if expired
            if age_hours > SESSION_EXPIRY_HOURS:
                os.remove(path)
                removed += 1
                logger.info(f"Removed expired session: {f}")
            else:
                session_files.append((path, mtime))
        except (OSError, ValueError) as e:
            logger.warning(f"Error checking session {f}: {e}")
    
    # Enforce max sessions limit (remove oldest)
    if len(session_files) > MAX_SESSIONS:
        session_files.sort(key=lambda x: x[1])  # Sort by mtime
        to_remove = session_files[:len(session_files) - MAX_SESSIONS]
        for path, _ in to_remove:
            try:
                os.remove(path)
                removed += 1
                logger.info(f"Removed session (limit exceeded): {os.path.basename(path)}")
            except OSError:
                pass
    
    return removed

def get_session_title(messages):
    """Generate a title from the first user message."""
    if not messages:
        return "New Chat"
    
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if len(content) > 30:
                return content[:30] + "..."
            return content if content else "New Chat"
    return "New Chat"


def get_session_files():
    """Returns list of available chat sessions with titles."""
    if not os.path.exists(CHATS_DIR):
        return []
    
    files = [f for f in os.listdir(CHATS_DIR) if f.endswith('.json')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHATS_DIR, x)), reverse=True)
    
    session_list = []
    for f in files:
        sid = f.replace(".json", "")
        try:
            # Validate session ID format
            sanitize_session_id(sid)
            with open(os.path.join(CHATS_DIR, f), "r") as file:
                data = json.load(file)
                if isinstance(data, list):
                    title = get_session_title(data)
                else:
                    title = data.get("title", get_session_title(data.get("messages", [])))
                session_list.append({"id": sid, "title": title})
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Skipping invalid session file {f}: {e}")
            continue
            
    return session_list


def load_session(session_id):
    """Load a session's messages from disk with validation."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on load: {e}")
        return []
    
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    
    # Verify path is within CHATS_DIR (prevent symlink escape)
    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt detected: {path}")
        return []
    
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                # Use file locking for read
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read().strip()
                    if not content:
                        return []
                    
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data[:MAX_MESSAGES_PER_SESSION]  # Limit messages
                    messages = data.get("messages", [])
                    return messages[:MAX_MESSAGES_PER_SESSION]  # Limit messages
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return []
    return []


def save_session(session_id, messages):
    """Save session with file locking and restricted permissions."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on save: {e}")
        return
    
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    
    # Verify path is within CHATS_DIR
    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt detected on save: {path}")
        return
    
    # Limit message count
    messages = messages[:MAX_MESSAGES_PER_SESSION]
    
    try:
        title = get_session_title(messages)
        
        data = {
            "title": title,
            "timestamp": str(datetime.now()),
            "messages": []
        }
        
        # Convert image bytes to base64 for JSON serialization
        for msg in messages:
            msg_copy = msg.copy()
            if "image" in msg_copy and isinstance(msg_copy["image"], bytes):
                msg_copy["image"] = base64.b64encode(msg_copy["image"]).decode('utf-8')
                msg_copy["is_image_base64"] = True
            data["messages"].append(msg_copy)
        
        # Write with exclusive lock and restricted permissions
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Restrict file permissions (owner read/write only)
        os.chmod(path, 0o600)
        
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {sanitize_error_message(e)}")


def create_new_session():
    """Create a new chat session."""
    session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"
    st.session_state.current_session = session_id
    st.session_state.messages = []
    save_session(session_id, [])
    return session_id


def delete_session(session_id):
    """Delete a chat session file with validation."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on delete: {e}")
        return
    
    path = os.path.join(CHATS_DIR, f"{session_id}.json")
    
    # Verify path is within CHATS_DIR
    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt on delete: {path}")
        return
    
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Deleted session: {session_id}")


# ═══════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════

# Thinking indicator HTML
THINKING_HTML = """
<div class="thinking-box">
    <div class="thinking-dots">
        <span></span><span></span><span></span>
    </div>
    <span class="thinking-text">Thinking...</span>
</div>
"""

class StreamHandler(BaseCallbackHandler):
    """Handler for streaming LLM responses to the UI."""
    def __init__(self, container, initial_text="", thinking_placeholder=None):
        self.container = container
        self.text = initial_text
        self.thinking_placeholder = thinking_placeholder
        self.first_token = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Clear thinking indicator on first token
        if self.first_token and self.thinking_placeholder:
            self.thinking_placeholder.empty()
            self.first_token = False
        self.text += token
        self.container.markdown(self.text + "▌")


@st.cache_resource
def load_embeddings():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource
def load_llm(model_name: str):
    """Load and validate an Ollama LLM."""
    try:
        llm = Ollama(
            model=model_name,
            temperature=0.2,
            base_url=OLLAMA_BASE_URL
        )
        llm.invoke("test")
        return llm, None
    except Exception as e:
        error_msg = f"Ollama Error: {str(e)}"
        if "404" in str(e):
            error_msg = f"Model '{model_name}' not found. Run: ollama pull {model_name}"
        return None, error_msg


def get_llm_for_chain(model_name: str, callback_handler=None):
    """Get an LLM instance for use in chains with optional streaming."""
    try:
        callbacks = [callback_handler] if callback_handler else []
        llm = Ollama(
            model=model_name,
            temperature=0.2,
            base_url=OLLAMA_BASE_URL,
            callbacks=callbacks
        )
        return llm
    except Exception as e:
        return None


def process_vision_request(image_file, prompt, model_name="llava"):
    """Handles image analysis using Ollama's multi-modal capabilities."""
    import requests
    
    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response from vision model.")
        else:
            return f"Vision Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def read_file_content(file_path: Path) -> str:
    """Read content from a file with size limit and encoding handling."""
    try:
        # Skip files larger than 1MB
        if file_path.stat().st_size > 1_000_000:
            return ""
            
        if file_path.suffix == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        else:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            return ""
    except Exception:
        return ""


def get_text_splitter(file_ext: str):
    """Get appropriate text splitter for file type."""
    language = SUPPORTED_EXTENSIONS.get(file_ext)
    if language and language != 'pdf':
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=1000, chunk_overlap=200
            )
        except Exception:
            pass
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def process_uploaded_files(uploaded_files) -> List[Document]:
    """Process uploaded files into Document objects with security validation."""
    documents = []
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Sanitize filename to prevent path traversal
            safe_filename = sanitize_filename(uploaded_file.name)
            
            # Check file size BEFORE writing (use buffer size)
            file_size = len(uploaded_file.getbuffer())
            max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            
            if file_size > max_size_bytes:
                logger.warning(f"File {safe_filename} exceeds size limit ({file_size} > {max_size_bytes})")
                st.warning(f"Skipped {safe_filename}: exceeds {MAX_FILE_SIZE_MB}MB limit")
                continue
            
            file_path = Path(UPLOAD_DIR) / safe_filename
            
            # Verify the path is within UPLOAD_DIR
            if not validate_path_within_directory(file_path, Path(UPLOAD_DIR)):
                logger.error(f"Path traversal attempt in upload: {uploaded_file.name}")
                continue
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Restrict file permissions
            os.chmod(file_path, 0o600)
            
            content = read_file_content(file_path)
            if content.strip():
                splitter = get_text_splitter(file_path.suffix)
                chunks = splitter.split_text(content)
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": str(file_path), "filename": safe_filename, "chunk": j}
                    )
                    documents.append(doc)
        except ValueError as e:
            logger.warning(f"Invalid filename {uploaded_file.name}: {e}")
            st.warning(f"Skipped invalid file: {uploaded_file.name}")
            continue
            
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    return documents


def scan_directory(root_path: str) -> List[Document]:
    """Scan a directory for supported files with security limits."""
    # Validate input path
    if not root_path or not root_path.strip():
        logger.warning("Empty path provided to scan_directory")
        st.warning("Please provide a valid folder path")
        return []
    
    documents = []
    root = Path(root_path)
    
    if not root.exists():
        logger.warning(f"Path does not exist: {root_path}")
        return []
    
    # Prevent scanning sensitive system directories (but allow user home directories)
    root_resolved = root.resolve()
    sensitive_paths = ['/etc', '/var', '/usr', '/bin', '/sbin', '/root', '/boot', '/sys', '/proc']
    for sensitive in sensitive_paths:
        if str(root_resolved).startswith(sensitive):
            logger.warning(f"Attempted to scan sensitive path: {root_resolved}")
            st.warning("Cannot scan system directories for security reasons")
            return []

    valid_files = []
    # Use followlinks=False to prevent symlink escape attacks
    for current_root, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith('.') and not d.endswith(IGNORED_SUFFIXES)]
        
        for file in files:
            if file in IGNORED_FILES:
                continue
            if Path(file).suffix in SUPPORTED_EXTENSIONS:
                file_path = Path(current_root) / file
                # Ensure file is really within the root (no symlink escape)
                if validate_path_within_directory(file_path, root):
                    valid_files.append(file_path)
        
        # Enforce file count limit
        if len(valid_files) >= MAX_FILES_PER_SCAN:
            logger.warning(f"File scan limit reached ({MAX_FILES_PER_SCAN} files)")
            st.warning(f"Scan limited to {MAX_FILES_PER_SCAN} files")
            break

    if not valid_files:
        return []

    progress_bar = st.progress(0)
    for i, file_path in enumerate(valid_files):
        content = read_file_content(file_path)
        if content.strip():
            splitter = get_text_splitter(file_path.suffix)
            chunks = splitter.split_text(content)
            for j, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": str(file_path), "filename": file_path.name, "chunk": j}
                )
                documents.append(doc)
        if i % max(1, len(valid_files) // 20) == 0:
            progress_bar.progress((i + 1) / len(valid_files))
    progress_bar.empty()
    return documents


# ═══════════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════════

def initialize_vectorstore(embeddings, force_recreate=False):
    """Initialize or recreate the ChromaDB vector store."""
    try:
        if force_recreate and os.path.exists(DB_DIRECTORY):
            shutil.rmtree(DB_DIRECTORY)
            st.toast("Database cleared")
        client = chromadb.PersistentClient(
            path=DB_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        return Chroma(client=client, collection_name="omniscience", embedding_function=embeddings)
    except Exception:
        return None


def ingest_documents(vectorstore, documents: List[Document]):
    """Add documents to the vector store in batches."""
    if documents:
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            vectorstore.add_documents(documents[i:i + batch_size])
        st.success(f"Indexed {len(documents)} chunks")


def delete_file_from_db(vectorstore, filename):
    """Delete all chunks for a specific file from the vector store."""
    try:
        vectorstore._collection.delete(where={"filename": filename})
        st.toast(f"Deleted: {filename}")
    except Exception:
        pass


def get_all_filenames(vectorstore):
    """Get all unique filenames in the vector store."""
    try:
        data = vectorstore._collection.get(include=['metadatas'])
        filenames = set()
        for meta in data['metadatas']:
            if 'filename' in meta:
                filenames.add(meta['filename'])
        return list(filenames)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# TOOLS: Web Search, SQL, Code Execution
# ═══════════════════════════════════════════════════════════════════

def run_web_search(query):
    """Run a web search using DuckDuckGo."""
    if not HAS_WEB_SEARCH:
        return "Web search unavailable. Install: pip install duckduckgo-search"
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        results = wrapper.results(query, max_results=5)
        
        if not results:
            return "No results found."
        
        formatted_response = "### Web Search Results\n\n"
        sources_list = []
        
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            snippet = res.get('snippet', 'No content available.')
            link = res.get('link', '#')
            formatted_response += f"**{i+1}. {title}**\n{snippet}\n\n"
            sources_list.append(f"{i+1}. [{title}]({link})")
            
        formatted_response += "---\n**Sources:**\n" + "\n".join(sources_list)
        return formatted_response
    except Exception as e:
        return f"Search failed: {str(e)}"


def extract_academic_query(user_prompt: str, rag_context: str, llm) -> str:
    """Use LLM to extract relevant academic search terms from the RAG context.
    
    Instead of searching for the user's exact prompt, this extracts the underlying
    research topic from the code/document context.
    """
    extraction_prompt = f"""Extract an academic search query.

USER QUESTION:
{user_prompt}

CONTEXT:
{rag_context[:3000]}

TASK:
Write ONE short search query (5–7 keywords) for academic papers.

RULES:
- Use technical terms
- No explanations
- No punctuation
- Max 200 characters

OUTPUT:"""

    try:
        result = llm.invoke(extraction_prompt)
        # Clean up the response - get first line, remove quotes
        query = str(result).strip().split('\n')[0].strip('"\'')
        # Limit length for API
        if len(query) > 200:
            query = query[:200]
        logger.info(f"Extracted academic query: {query}")
        return query
    except Exception as e:
        logger.warning(f"Failed to extract academic query: {e}")
        # Fallback to first 100 chars of user prompt
        return user_prompt[:100]


def run_academic_search(query, max_results=100, rag_context=None, llm=None):
    """Search academic databases: Semantic Scholar, arXiv, OpenAlex.
    
    If rag_context and llm are provided, uses LLM to extract smart search terms.
    Results are ranked by a combination of relevance and citation count.
    """
    # Smart query extraction if context is available
    if rag_context and llm:
        query = extract_academic_query(query, rag_context, llm)
    
    all_papers = []
    
    # 1. Semantic Scholar (has citations, authors, venue)
    # Note: Rate limit is 100 requests/5 min without API key
    if HAS_SEMANTIC_SCHOLAR:
        try:
            # Disable auto-retry to avoid blocking UI for minutes
            sch = SemanticScholar(timeout=5, retry=False)
            results = sch.search_paper(
                query, 
                limit=min(max_results, 20),  # Reduced to avoid rate limits
                fields=['title', 'abstract', 'year', 'authors', 'citationCount', 'venue', 'url', 'openAccessPdf']
            )
            for paper in results:
                if paper.title:
                    authors = ", ".join([a.name for a in (paper.authors or [])[:3]])
                    if len(paper.authors or []) > 3:
                        authors += " et al."
                    all_papers.append({
                        'source': 'Semantic Scholar',
                        'title': paper.title,
                        'authors': authors,
                        'year': paper.year or 'N/A',
                        'citations': paper.citationCount or 0,
                        'venue': paper.venue or '',
                        'abstract': (paper.abstract or '')[:300] + '...' if paper.abstract and len(paper.abstract) > 300 else (paper.abstract or 'No abstract'),
                        'url': paper.url or '',
                        'open_access': paper.openAccessPdf.get('url') if paper.openAccessPdf else None
                    })
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                logger.warning("Semantic Scholar rate limited - skipping (try again in a few minutes)")
            else:
                logger.warning(f"Semantic Scholar search failed: {e}")
    
    # 2. arXiv (no citations, but has preprints)
    if HAS_ARXIV:
        try:
            # Use Client API to avoid deprecation warning
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=min(max_results, 100),
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in client.results(search):
                authors = ", ".join([a.name for a in result.authors[:3]])
                if len(result.authors) > 3:
                    authors += " et al."
                all_papers.append({
                    'source': 'arXiv',
                    'title': result.title,
                    'authors': authors,
                    'year': result.published.year if result.published else 'N/A',
                    'citations': -1,  # arXiv doesn't track citations
                    'venue': 'arXiv preprint',
                    'abstract': result.summary[:300] + '...' if len(result.summary) > 300 else result.summary,
                    'url': result.entry_id,
                    'open_access': result.pdf_url
                })
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
    
    # 3. OpenAlex (free, has citations, broad coverage)
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.openalex.org/works?search={encoded_query}&per_page={min(max_results, 100)}&sort=relevance_score:desc"
        req = urllib.request.Request(url, headers={'User-Agent': 'OmnisciencePro/1.0 (mailto:user@example.com)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for work in data.get('results', []):
                authors_list = work.get('authorships', [])[:3]
                authors = ", ".join([a.get('author', {}).get('display_name', '') for a in authors_list])
                if len(work.get('authorships', [])) > 3:
                    authors += " et al."
                
                abstract = ''
                if work.get('abstract_inverted_index'):
                    # OpenAlex stores abstract as inverted index - reconstruct
                    inv_idx = work['abstract_inverted_index']
                    words = [''] * (max(max(positions) for positions in inv_idx.values()) + 1)
                    for word, positions in inv_idx.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = ' '.join(words)[:300] + '...'
                
                all_papers.append({
                    'source': 'OpenAlex',
                    'title': work.get('title', 'Untitled'),
                    'authors': authors,
                    'year': work.get('publication_year', 'N/A'),
                    'citations': work.get('cited_by_count', 0),
                    'venue': work.get('primary_location', {}).get('source', {}).get('display_name', '') if work.get('primary_location') else '',
                    'abstract': abstract or 'No abstract available',
                    'url': work.get('doi') or work.get('id', ''),
                    'open_access': work.get('open_access', {}).get('oa_url')
                })
    except Exception as e:
        logger.warning(f"OpenAlex search failed: {e}")
    
    if not all_papers:
        return "No academic papers found for this query."
    
    # Rank papers: balance relevance (order) with citations
    # Papers already come sorted by relevance from APIs
    # Add citation boost but don't let super-cited old papers dominate
    current_year = datetime.now().year
    for i, paper in enumerate(all_papers):
        relevance_score = 100 - i  # Higher for earlier results
        citation_score = min(paper['citations'], 500) / 10 if paper['citations'] >= 0 else 0  # Cap at 50 points
        recency_boost = max(0, (10 - (current_year - (paper['year'] if isinstance(paper['year'], int) else 0)))) if isinstance(paper['year'], int) else 0
        paper['score'] = relevance_score + citation_score + recency_boost
    
    # Sort by combined score and deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in sorted(all_papers, key=lambda x: x['score'], reverse=True):
        title_key = paper['title'].lower()[:50]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(paper)
    
    # Format results (show top 30)
    formatted = "### Academic Research Results\n\n"
    for i, paper in enumerate(unique_papers[:30]):
        citations_str = f"Citations: {paper['citations']}" if paper['citations'] >= 0 else "Citations: N/A"
        formatted += f"**{i+1}. {paper['title']}**\n"
        formatted += f"   *{paper['authors']}* ({paper['year']}) | {citations_str} | {paper['source']}\n"
        if paper['venue']:
            formatted += f"   Venue: {paper['venue']}\n"
        formatted += f"   {paper['abstract']}\n"
        if paper['open_access']:
            formatted += f"   [Open Access PDF]({paper['open_access']})\n"
        elif paper['url']:
            formatted += f"   [Link]({paper['url']})\n"
        formatted += "\n"
    
    formatted += f"---\n*Found {len(unique_papers)} unique papers from Semantic Scholar, arXiv, and OpenAlex*"
    return formatted


def query_sqlite_db(db_path, query, llm):
    """Query a SQLite database using natural language with comprehensive security restrictions."""
    if not os.path.exists(db_path):
        return "Database file not found."
    
    # Rate limiting
    if not check_rate_limit("sql_query"):
        return "Rate limit exceeded. Please wait before making more queries."
    
    # Security limits
    QUERY_TIMEOUT_SECONDS = 5
    MAX_ROWS = 1000
    
    try:
        # Open database in READ-ONLY mode to prevent modification
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=QUERY_TIMEOUT_SECONDS)
        
        # Disable dangerous features at the connection level
        conn.execute("PRAGMA query_only = ON")  # Extra read-only protection
        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_str = str(tables)
        
        prompt = f"""Generate a SQLite SELECT query.

SCHEMA:
{schema_str}

QUESTION:
{query}

RULES:
- SELECT only
- No INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, PRAGMA
- Single statement
- Simple query

Return ONLY the SQL query."""
        
        sql_query = llm.invoke(prompt).strip().replace("```sql", "").replace("```", "").strip()
        
        # SECURITY: Comprehensive blocking of dangerous SQL patterns
        sql_upper = sql_query.upper().strip()
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            logger.warning(f"Blocked non-SELECT SQL query: {sql_query[:100]}")
            return "Only SELECT queries are allowed for security reasons."
        
        # Block all dangerous keywords - including bypass attempts
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
            'ATTACH', 'DETACH', 'PRAGMA', 'LOAD_EXTENSION',  # SQLite-specific dangers
            'INTO OUTFILE', 'INTO DUMPFILE',  # MySQL-style bypass attempts
            'VACUUM', 'REINDEX', 'ANALYZE',  # Can be resource-intensive
            ';--', '/*', '*/',  # Comment injection attempts
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"Blocked dangerous SQL keyword '{keyword}': {sql_query[:100]}")
                return f"Query contains prohibited keyword: {keyword}"
        
        # Block multiple statements (prevent command chaining)
        if sql_query.count(';') > 1:
            logger.warning(f"Blocked multi-statement SQL: {sql_query[:100]}")
            return "Multiple SQL statements are not allowed."
        
        # Execute with row limit
        cursor.execute(sql_query)
        results = cursor.fetchmany(MAX_ROWS)
        
        # Check if more rows were available
        has_more = cursor.fetchone() is not None
        
        conn.close()
        
        result_text = f"**SQL:** `{sql_query}`\n\n**Results ({len(results)} rows"
        if has_more:
            result_text += f", limited to {MAX_ROWS}"
        result_text += f"):**\n{str(results)}"
        
        logger.info(f"Executed SQL query: {sql_query[:100]}")
        return result_text
        
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "timeout" in str(e).lower():
            return "Query timeout exceeded. Please simplify your query."
        return f"SQL Error: {sanitize_error_message(e)}"
    except Exception as e:
        return f"SQL Error: {sanitize_error_message(e)}"

# ═══════════════════════════════════════════════════════════════════
# CODE EXECUTION REMOVED FOR SECURITY
# ═══════════════════════════════════════════════════════════════════
# The execute_python_code function has been PERMANENTLY REMOVED.
# Feature flags are NOT security boundaries.
# If you need code execution, use a separate sandboxed service.
# ═══════════════════════════════════════════════════════════════════


def get_qa_chain_for_streaming(llm, vectorstore):
    """Create a QA chain with streaming support."""
    template = """You are Omniscience, an expert AI assistant with access to a local codebase.

=== RETRIEVED FILE CONTENTS ===
{context}
=== END OF RETRIEVED CONTENTS ===

USER QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. FIRST, assess if the retrieved content is RELEVANT to the user's question:
   - If the question is about general knowledge (sports, news, history, science facts) and the retrieved content is about code/programming, the context is IRRELEVANT.
   - If asking about "Formula 1" racing and you see code with "F1-score" (a machine learning metric), these are DIFFERENT things - the code is IRRELEVANT.
   - If asking about current events/news and the context is old code files, the context is IRRELEVANT.

2. IF CONTEXT IS IRRELEVANT:
   - DO NOT use the retrieved code content to answer general knowledge questions.
   - Say "The local codebase doesn't contain information about [topic]. Based on general knowledge:" and answer from your training.
   - If web search results are appended below, USE THOSE INSTEAD.

3. IF CONTEXT IS RELEVANT (question is about the code/files):
   - Read and analyze the content carefully before responding.
   - Base your answer primarily on the retrieved content.
   - Quote specific code, functions, or text when relevant.

4. NEVER say "please provide the context" - the context has ALREADY been provided above.
5. For questions about code changes across versions, look for version numbers (v1, v2, v3), dates, or changelog entries.
6. Format your response clearly with headers, bullet points, and code blocks as appropriate.

YOUR RESPONSE:"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Increased from 5 to 8 for more context
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )



# ═══════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Omniscience Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PURPLE_THEME_CSS, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-title">OMNISCIENCE PRO</div>', unsafe_allow_html=True)
    st.markdown("##### Local RAG System // Offline Mode")
    st.markdown("---")
    
    # Session Management
    if 'current_session' not in st.session_state:
        sessions = get_session_files()
        if sessions:
            st.session_state.current_session = sessions[0]["id"]
            st.session_state.messages = load_session(st.session_state.current_session)
        else:
            create_new_session()
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # ════════════ SIDEBAR ════════════
    with st.sidebar:
        st.markdown("### SYSTEM CONFIG")
        
        # Mode Selection (Web Search is now a toggle, not a mode)
        mode = st.radio("Mode", ["Chat (RAG)", "Vision (Images)", "Database (SQL)"], index=0)
        
        # Web Search Toggle (persistent across messages)
        if 'web_search_enabled' not in st.session_state:
            st.session_state.web_search_enabled = False
        
        st.session_state.web_search_enabled = st.toggle(
            "Augment with Web Search", 
            value=st.session_state.web_search_enabled,
            help="When enabled, responses will be augmented with web search results"
        )
        
        if st.session_state.web_search_enabled and not HAS_WEB_SEARCH:
            st.warning("Web search unavailable. Install: pip install duckduckgo-search")
        
        # Academic Research Toggle (Semantic Scholar, arXiv, OpenAlex)
        if 'academic_search_enabled' not in st.session_state:
            st.session_state.academic_search_enabled = False
        
        st.session_state.academic_search_enabled = st.toggle(
            "Academic Research", 
            value=st.session_state.academic_search_enabled,
            help="Search Semantic Scholar, arXiv, OpenAlex for academic papers"
        )
        
        if st.session_state.academic_search_enabled and not (HAS_SEMANTIC_SCHOLAR or HAS_ARXIV):
            st.warning("Install: pip install semanticscholar arxiv")
        
        st.markdown("---")
        
        # Export Chat - Direct download button (no double-click needed)
        if st.session_state.messages:
            md_content = f"# Chat Export\n\n**Session:** {st.session_state.current_session}\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            for msg in st.session_state.messages:
                role = "**User:**" if msg["role"] == "user" else "**Assistant:**"
                md_content += f"{role}\n\n{msg['content']}\n\n"
                if msg.get("sources"):
                    md_content += "**Sources:**\n" + "\n".join([f"- `{s}`" for s in msg["sources"]]) + "\n\n"
                md_content += "---\n\n"
            
            st.download_button(
                label="Export Chat (Markdown)",
                data=md_content,
                file_name=f"chat_export_{st.session_state.current_session}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.markdown("### CHAT SESSIONS")
        
        if st.button("+ NEW CHAT", use_container_width=True):
            create_new_session()
            st.rerun()
            
        # Session Selector
        sessions = get_session_files()
        session_ids = [s["id"] for s in sessions]
        session_titles = [s["title"] for s in sessions]
        
        if session_ids:
            try:
                idx = session_ids.index(st.session_state.current_session)
            except ValueError:
                idx = 0
                
            selected_idx = st.selectbox(
                "History", 
                range(len(session_titles)), 
                format_func=lambda x: session_titles[x],
                index=idx
            )
            
            selected_id = session_ids[selected_idx]
            
            if selected_id != st.session_state.current_session:
                st.session_state.current_session = selected_id
                st.session_state.messages = load_session(selected_id)
                st.rerun()
                
            if st.button("DELETE CHAT", type="primary"):
                delete_session(st.session_state.current_session)
                # Select another existing session, or create new only if none exist
                remaining_sessions = get_session_files()
                if remaining_sessions:
                    st.session_state.current_session = remaining_sessions[0]["id"]
                    st.session_state.messages = load_session(remaining_sessions[0]["id"])
                else:
                    # No sessions left, create a fresh one
                    create_new_session()
                st.rerun()

        st.markdown("---")
        
        # Model Selector logic based on mode
        if mode == "Vision (Images)":
            model_options = ["llava:7b", "llama3.2-vision"]
            model_name = st.selectbox("Vision Model", options=model_options, index=0)
            st.info(f"Using Vision Model: {model_name}")
        else:
            model_options = ["qwen3:4b", "qwen2.5-coder:7b", "qwen2.5-coder:1.5b", "llama3.2:3b", "mistral:7b"]
            model_name = st.selectbox("Model", options=model_options, index=1)

        # Context/File Controls (Only in Chat Mode)
        if mode == "Chat (RAG)":
            st.markdown("#### DATA SOURCE")
            root_path = st.text_input("Folder Path", value=".")
            c1, c2 = st.columns(2)
            with c1: 
                if st.button("SCAN"):
                    with st.spinner("Scanning folder..."):
                        st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)
                        docs = scan_directory(root_path)
                        ingest_documents(st.session_state.vectorstore, docs)
                    st.success(f"✅ Scanned {len(docs)} documents from {root_path}")
            with c2:
                if st.button("PURGE"):
                    initialize_vectorstore(load_embeddings(), True)
                    st.session_state.vectorstore = None
                    st.info("🗑️ Vector database purged")
            
            uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
            
            # Check if user uploaded images - guide them to Vision mode
            if uploaded_files:
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.svg'}
                image_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in image_extensions]
                
                if image_files:
                    st.warning(f"Detected {len(image_files)} image file(s). For image analysis, please use **Vision (Images)** mode in the sidebar.")
                    st.markdown(VISION_PULSE_JS, unsafe_allow_html=True)
                    # Filter out images from processing
                    uploaded_files = [f for f in uploaded_files if Path(f.name).suffix.lower() not in image_extensions]
            
            # Check if user uploaded SQL/database files - guide them to Database mode
            if uploaded_files:
                sql_extensions = {'.sql', '.db', '.sqlite', '.sqlite3'}
                sql_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in sql_extensions]
                
                if sql_files:
                    st.warning(f"Detected {len(sql_files)} database/SQL file(s). For database queries, please use **Database (SQL)** mode in the sidebar.")
                    st.markdown(SQL_PULSE_JS, unsafe_allow_html=True)
                    # Filter out SQL files from processing
                    uploaded_files = [f for f in uploaded_files if Path(f.name).suffix.lower() not in sql_extensions]
            
            if uploaded_files and st.button("PROCESS"):
                st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)
                docs = process_uploaded_files(uploaded_files)
                ingest_documents(st.session_state.vectorstore, docs)
            
            with st.expander("Manage Knowledge Base"):
                if st.session_state.vectorstore:
                    all_files = get_all_filenames(st.session_state.vectorstore)
                    if all_files:
                        del_file = st.selectbox("Delete File:", options=all_files)
                        if st.button("DELETE"):
                            delete_file_from_db(st.session_state.vectorstore, del_file)
                            st.rerun()
        
        # Database (SQL) Mode Controls
        if mode == "Database (SQL)":
            st.markdown("#### SQL SOURCE")
            uploaded_db = st.file_uploader("Upload SQLite DB", type=['db', 'sqlite', 'sqlite3'])
            if uploaded_db:
                try:
                    safe_dbname = sanitize_filename(uploaded_db.name)
                    db_path = os.path.join(UPLOAD_DIR, safe_dbname)
                    
                    # Size check
                    file_size = len(uploaded_db.getbuffer())
                    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
                    if file_size > max_size_bytes:
                        st.error(f"Database file too large: {file_size // (1024*1024)}MB > {MAX_FILE_SIZE_MB}MB limit")
                    else:
                        with open(db_path, "wb") as f:
                            f.write(uploaded_db.getbuffer())
                        os.chmod(db_path, 0o600)
                        st.session_state.db_path = db_path
                        st.success(f"Loaded: {safe_dbname}")
                except ValueError as e:
                    st.error(f"Invalid filename: {sanitize_error_message(e)}")

    # ════════════ MAIN LOGIC ════════════
    
    # Load Vectorstore if exists
    if mode == "Chat (RAG)" and st.session_state.vectorstore is None and os.path.exists(DB_DIRECTORY):
        st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)

    # Display History
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Handle Image in history
            if "image" in message:
                image_data = message["image"]
                if message.get("is_image_base64"):
                    try:
                        image_data = base64.b64decode(image_data)
                    except Exception:
                        pass
                st.image(image_data, caption="Uploaded Image", use_container_width=True)
            
            st.markdown(message["content"])
            
            if "sources" in message:
                with st.expander("Sources"):
                    for s in message["sources"]:
                        st.markdown(f"- `{s}`")
            
            # Copy button for assistant responses - shows code block for copying
            if message["role"] == "assistant":
                if st.button("Copy Response", key=f"copy_{idx}", type="secondary"):
                    st.code(message["content"], language=None)
                    st.info("Select all (Ctrl+A) and copy (Ctrl+C)")

    # Input Area
    if mode == "Vision (Images)":
        img_file = st.file_uploader("Upload Image to Analyze", type=["png", "jpg", "jpeg", "webp"])
        if img_file and (prompt := st.chat_input("Ask about this image...")):
            image_bytes = img_file.getvalue()
            st.session_state.messages.append({"role": "user", "content": prompt, "image": image_bytes})
            save_session(st.session_state.current_session, st.session_state.messages)
            st.rerun()

        # Processing logic for vision
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "image" in st.session_state.messages[-1]:
            last_msg = st.session_state.messages[-1]
            image_bytes = last_msg["image"]
            prompt_text = last_msg["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing image..."):
                    class BytesWrapper:
                        def __init__(self, b):
                            self.b = b
                        def getvalue(self):
                            return self.b
                    
                    response_content = process_vision_request(BytesWrapper(image_bytes), prompt_text, model_name)
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    save_session(st.session_state.current_session, st.session_state.messages)
                    st.rerun()

    # Standard Chat Input
    if prompt := st.chat_input("Enter query..."):
        if mode == "Vision (Images)":
            pass  # Handled above
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_session(st.session_state.current_session, st.session_state.messages)
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Show thinking indicator
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)
                
                response_placeholder = st.empty()
                stream_handler = StreamHandler(response_placeholder, thinking_placeholder=thinking_placeholder)
                llm = get_llm_for_chain(model_name, stream_handler)
                
                response_content = ""
                sources = []
                
                if llm:
                    try:
                        if mode == "Database (SQL)":
                            thinking_placeholder.empty()  # Clear thinking indicator
                            if hasattr(st.session_state, 'db_path') and st.session_state.db_path:
                                response_content = query_sqlite_db(st.session_state.db_path, prompt, llm)
                                response_placeholder.markdown(response_content)
                            else:
                                response_content = "Please upload a database file first."
                                response_placeholder.markdown(response_content)

                        else:  # Chat RAG
                            # STEP 1: Fetch web/academic results FIRST (so model can use them)
                            web_results = ""
                            academic_results = ""
                            
                            if st.session_state.web_search_enabled and HAS_WEB_SEARCH:
                                web_results = run_web_search(prompt)
                            
                            # STEP 2: Get RAG context if vectorstore exists
                            if st.session_state.vectorstore:
                                # Get RAG retrieval results (but don't run full QA chain yet)
                                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})
                                retrieved_docs = retriever.invoke(prompt)
                                rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                                sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
                                
                                # Academic search with smart query extraction
                                if st.session_state.academic_search_enabled:
                                    extraction_llm = get_llm_for_chain(model_name, None)
                                    academic_results = run_academic_search(
                                        prompt, 
                                        rag_context=rag_context, 
                                        llm=extraction_llm
                                    )
                                
                                # STEP 3: Build unified prompt with ALL context (HARDENED VERSION)
                                
                                # Build conversation history (last 10 messages for context)
                                conversation_history = ""
                                history_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                                if history_messages:
                                    history_parts = []
                                    for msg in history_messages:
                                        role = "USER" if msg["role"] == "user" else "ASSISTANT"
                                        # Truncate long messages to save tokens
                                        content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                                        history_parts.append(f"{role}: {content}")
                                    conversation_history = "\n\n".join(history_parts)
                                
                                unified_prompt = f"""You are Omniscience, an AI assistant.

You may be given:
- Conversation history
- Local code or documents
- Web search results
- Academic search results

==============================
CONVERSATION HISTORY
==============================
{conversation_history if conversation_history else "(No previous messages)"}

==============================
LOCAL CONTEXT
==============================
{rag_context}

"""
                                if web_results:
                                    logger.info(f"Web results included in prompt: {len(web_results)} chars")
                                    unified_prompt += f"""==============================
WEB RESULTS
==============================
{web_results}

"""
                                if academic_results:
                                    unified_prompt += f"""==============================
ACADEMIC RESULTS
==============================
{academic_results}

"""
                                unified_prompt += f"""USER QUESTION:
{prompt}

INSTRUCTIONS:
- First decide: Is the LOCAL CONTEXT useful for answering the question?
- If YES:
  - Answer using the LOCAL CONTEXT
  - Quote or refer to it when helpful
- If NO:
  - Ignore LOCAL CONTEXT completely
  - Answer using WEB or ACADEMIC results only

RULES:
- Do not mix unrelated sources
- Do not invent facts, code, or citations
- If none of the sources help, say: "The provided sources do not answer this."

ANSWER:"""
                                
                                response_content = llm.invoke(unified_prompt)
                                thinking_placeholder.empty()  # Clear AFTER LLM completes
                                response_placeholder.markdown(response_content)
                                
                                # Only show code sources if model actually used them
                                # (Check if response indicates external search was used instead)
                                used_external_only = "based on external search" in response_content.lower() or "based on web search" in response_content.lower()
                                
                                if not used_external_only and sources:
                                    with st.expander("Code Sources"):
                                        for s in sources:
                                            st.markdown(f"- `{s}`")
                                else:
                                    # Show what external sources were used
                                    external_sources = []
                                    if web_results:
                                        external_sources.append("Web Search (DuckDuckGo)")
                                    if academic_results:
                                        external_sources.append("Academic (arXiv, Semantic Scholar, OpenAlex)")
                                    if external_sources:
                                        st.info(f"Sources: {', '.join(external_sources)}")
                                
                                # Immediate Copy Response button
                                if st.button(" Copy Response", key="copy_immediate", type="secondary"):
                                    st.code(response_content, language=None)
                                    st.info("Select all (Ctrl+A) and copy (Ctrl+C)")
                            else:
                                # No vectorstore - use LLM with optional search results
                                context_parts = []
                                
                                # Build conversation history (last 10 messages)
                                history_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                                if history_messages:
                                    history_parts = []
                                    for msg in history_messages:
                                        role = "USER" if msg["role"] == "user" else "ASSISTANT"
                                        content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                                        history_parts.append(f"{role}: {content}")
                                    context_parts.append(f"=== CONVERSATION HISTORY ===\n{chr(10).join(history_parts)}\n=== END CONVERSATION HISTORY ===")
                                
                                if academic_results:
                                    context_parts.append(f"=== ACADEMIC RESEARCH RESULTS ===\n{academic_results}\n=== END ACADEMIC RESULTS ===")
                                
                                if web_results:
                                    context_parts.append(f"""=== WEB SEARCH RESULTS (evaluate for relevance) ===
{web_results}
=== END WEB SEARCH RESULTS ===

IMPORTANT: Critically evaluate web results - ignore irrelevant ones (hotels, random websites, etc.).""")
                                
                                if context_parts:
                                    augmented_prompt = f"""Answer the question using the sources below.

CONVERSATION HISTORY:
{chr(10).join(history_parts) if history_parts else "(None)"}

ACADEMIC RESULTS:
{academic_results if academic_results else "(None)"}

WEB RESULTS:
{web_results if web_results else "(None)"}

QUESTION:
{prompt}

RULES:
- Use conversation history only to understand follow-up questions
- Prefer academic results when available
- Ignore irrelevant web results
- Do not invent information
- If the sources do not answer the question, say so clearly

ANSWER:"""
                                    response_content = llm.invoke(augmented_prompt)
                                else:
                                    response_content = llm.invoke(prompt)
                                thinking_placeholder.empty()  # Clear after LLM completes
                                response_placeholder.markdown(response_content)

                        st.session_state.messages.append({"role": "assistant", "content": response_content, "sources": sources})
                        save_session(st.session_state.current_session, st.session_state.messages)
                    except Exception as e:
                        logger.error(f"Error processing request: {e}")
                        st.error(f"Error: {sanitize_error_message(e)}")


if __name__ == "__main__":
    main()
