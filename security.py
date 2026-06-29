"""Security utilities: input sanitizers, path validation, rate limiter, error redaction."""
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

from config import _RL_DB, DB_DIRECTORY, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

logger = logging.getLogger(__name__)


def sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal attacks."""
    if not session_id:
        raise ValueError("Session ID cannot be empty")
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        raise ValueError(f"Invalid session ID format: {session_id}")
    if '..' in session_id or '/' in session_id or '\\' in session_id:
        raise ValueError("Invalid session ID: path traversal detected")
    return session_id


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks."""
    import unicodedata
    if not filename:
        raise ValueError("Filename cannot be empty")
    filename = os.path.basename(filename)
    filename = filename.replace('\x00', '')
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    filename = unicodedata.normalize('NFKC', filename)
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)
    if not filename or filename in ('.', '..'):
        raise ValueError("Invalid filename after sanitization")
    return filename


def validate_path_within_directory(path: Path, allowed_dir: Path) -> bool:
    """Ensure a path is within an allowed directory (no symlink escape)."""
    try:
        resolved_path = path.resolve()
        resolved_allowed = allowed_dir.resolve()
        return str(resolved_path).startswith(str(resolved_allowed))
    except (OSError, ValueError):
        return False


def _rl_conn() -> sqlite3.Connection:
    """Open (and initialize) the rate-limit SQLite database."""
    os.makedirs(DB_DIRECTORY, exist_ok=True)
    conn = sqlite3.connect(_RL_DB, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (user_id TEXT NOT NULL, ts REAL NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_ts ON requests(user_id, ts)"
    )
    conn.commit()
    return conn


def check_rate_limit(user_id: str = "default") -> bool:
    """SQLite-backed rate limiting. Returns True if the request is allowed.

    Persists across app restarts. Fails open on DB errors.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    try:
        with _rl_conn() as conn:
            conn.execute("DELETE FROM requests WHERE ts < ?", (window_start,))
            count = conn.execute(
                "SELECT COUNT(*) FROM requests WHERE user_id=? AND ts >= ?",
                (user_id, window_start),
            ).fetchone()[0]
            if count >= RATE_LIMIT_REQUESTS:
                return False
            conn.execute(
                "INSERT INTO requests(user_id, ts) VALUES (?,?)", (user_id, now)
            )
        return True
    except Exception as e:
        logger.warning(f"Rate limit DB error (failing open): {e}")
        return True


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to prevent information disclosure."""
    error_str = str(error)
    error_str = _redact_api_keys(error_str)
    error_str = re.sub(r'(/[^\s]+)+', '[PATH]', error_str)
    error_str = re.sub(
        r'(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)[\s\S]*', '[SQL]',
        error_str, flags=re.IGNORECASE
    )
    if len(error_str) > 200:
        error_str = error_str[:200] + '...'
    return error_str


# API-key-shaped tokens that must never surface in a UI error message.
_API_KEY_PATTERNS = [
    re.compile(r'sk-ant-[A-Za-z0-9_\-]{8,}'),        # Anthropic
    re.compile(r'sk-[A-Za-z0-9_\-]{16,}'),           # OpenAI / compatible
    re.compile(r'AIza[A-Za-z0-9_\-]{10,}'),          # Google
    re.compile(r'(?i)bearer\s+[A-Za-z0-9._\-]{8,}'),  # Authorization headers
]


def _redact_api_keys(text: str) -> str:
    """Replace anything resembling an API key/bearer token with [REDACTED]."""
    for pat in _API_KEY_PATTERNS:
        text = pat.sub('[REDACTED]', text)
    return text
