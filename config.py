"""
Configuration: all environment variables, constants, and startup side effects.
Import this module first — it creates necessary directories on import.
"""
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix matplotlib cache directory (must happen before matplotlib is imported elsewhere)
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


def _get_env_int(key: str, default: int, min_val: int = 1, max_val: int = 100000) -> int:
    """Get integer from environment with bounds validation."""
    try:
        val = int(os.environ.get(key, str(default)))
        return max(min_val, min(val, max_val))
    except (ValueError, TypeError):
        return default


# ── Directory paths ───────────────────────────────────────────────────────────
DB_DIRECTORY = os.environ.get('OMNISCIENCE_DB_DIR', './db_omniscience')
CHATS_DIR = os.environ.get('OMNISCIENCE_CHATS_DIR', './chats')
UPLOAD_DIR = os.environ.get('OMNISCIENCE_UPLOAD_DIR', './uploads')
EMBEDDING_MODEL = os.environ.get('OMNISCIENCE_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')

# ── Security limits ───────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = _get_env_int('OMNISCIENCE_MAX_FILE_SIZE_MB', 10, 1, 100)
MAX_FILES_PER_SCAN = _get_env_int('OMNISCIENCE_MAX_FILES_PER_SCAN', 1000, 10, 10000)
MAX_MESSAGES_PER_SESSION = _get_env_int('OMNISCIENCE_MAX_MESSAGES', 100, 10, 500)

# ── Session lifecycle ─────────────────────────────────────────────────────────
SESSION_EXPIRY_HOURS = _get_env_int('OMNISCIENCE_SESSION_EXPIRY_HOURS', 24, 1, 168)
SESSION_IDLE_TIMEOUT_HOURS = _get_env_int('OMNISCIENCE_IDLE_TIMEOUT_HOURS', 4, 1, 24)
MAX_SESSIONS = _get_env_int('OMNISCIENCE_MAX_SESSIONS', 100, 10, 1000)

# ── Rate limiting ─────────────────────────────────────────────────────────────
_RL_DB = os.path.join(DB_DIRECTORY, "rate_limits.db")
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# ── Conversation history ──────────────────────────────────────────────────────
HISTORY_WINDOW = 10           # last N messages included in context
HISTORY_EXCERPT_LENGTH = 500  # max chars per message in history

# ── File scanning constants ───────────────────────────────────────────────────
IGNORED_DIRS = {
    'node_modules', 'venv', '.venv', 'env', 'wenv', '.git', '.idea', '.vscode',
    '__pycache__', 'dist', 'build', 'coverage', 'target', 'bin', 'obj',
    '__MACOSX', '.pytest_cache', '.mypy_cache', '.tox', '*.egg-info',
    'package', 'logs', 'log', '.DS_Store', '.obsidian', 'Temp_Files', 'AWID3_Dataset',
    'DATASET_AWID2', 'AWID3_Dataset_CSV', 'data', 'wordlists', 'dict', 'datasets', 'dataset',
}

IGNORED_SUFFIXES = ('.egg-info', '.min.js', '.map', '.lock')
IGNORED_FILE_EXTENSIONS = {'.csv', '.pcap', '.pcapng', '.cap', '.log', '.json', '.xml', '.bin', '.dat'}
IGNORED_FILES = {
    'rockyou.txt', 'package-lock.json', 'yarn.lock', 'poetry.lock',
    '.DS_Store', 'thumbs.db',
}

# ── Ensure directories exist with restricted permissions ──────────────────────
for _dir in [UPLOAD_DIR, CHATS_DIR, DB_DIRECTORY]:
    os.makedirs(_dir, exist_ok=True)
    try:
        os.chmod(_dir, 0o700)
    except OSError:
        pass
