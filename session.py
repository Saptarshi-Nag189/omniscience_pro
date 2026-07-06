"""Session management: create, load, save, delete, and clean up chat sessions."""
import base64
import fcntl
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from config import (
    CHATS_DIR,
    MAX_MESSAGES_PER_SESSION,
    MAX_SESSIONS,
    SESSION_EXPIRY_HOURS,
    SESSION_IDLE_TIMEOUT_HOURS,
)
from security import sanitize_error_message, sanitize_session_id, validate_path_within_directory

logger = logging.getLogger(__name__)

LAST_SESSION_FILE = os.path.join(CHATS_DIR, '.last_session')


def _parse_session_ctime(filename: str) -> Optional[datetime]:
    """Extract creation time from a 'chat_YYYYMMDD_HHMMSS_*.json' filename.

    Returns None for filenames that don't encode a parseable timestamp.
    """
    match = re.match(r'chat_(\d{8})_(\d{6})_', filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
    except ValueError:
        return None


def cleanup_expired_sessions() -> int:
    """Remove expired/idle sessions and enforce the max session limit.

    Two independent signals decide removal:
      - expiry: total age since the session was created (from the filename
        timestamp, falling back to mtime when the name isn't parseable).
      - idle:   time since the last write (file mtime).

    Called once on app startup. Returns the number of sessions removed.
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
            idle_hours = (now - mtime).total_seconds() / 3600

            # Total age is measured from creation time; mtime is the fallback
            # for sessions whose filename predates the chat_<timestamp> scheme.
            created = _parse_session_ctime(f) or mtime
            age_hours = (now - created).total_seconds() / 3600

            if age_hours > SESSION_EXPIRY_HOURS:
                os.remove(path)
                removed += 1
                logger.info(f"Removed expired session: {f}")
                continue

            if idle_hours > SESSION_IDLE_TIMEOUT_HOURS:
                os.remove(path)
                removed += 1
                logger.info(f"Removed idle session: {f}")
                continue

            session_files.append((path, mtime))
        except (OSError, ValueError) as e:
            logger.warning(f"Error checking session {f}: {e}")

    if len(session_files) > MAX_SESSIONS:
        session_files.sort(key=lambda x: x[1])
        for path, _ in session_files[:len(session_files) - MAX_SESSIONS]:
            try:
                os.remove(path)
                removed += 1
                logger.info(f"Removed session (limit exceeded): {os.path.basename(path)}")
            except OSError:
                pass

    return removed


def get_session_title(messages: list) -> str:
    """Generate a session title from the first user message."""
    if not messages:
        return "New Chat"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return (content[:30] + "...") if len(content) > 30 else (content or "New Chat")
    return "New Chat"


def get_session_files() -> List[dict]:
    """Return list of available chat sessions sorted newest-first."""
    if not os.path.exists(CHATS_DIR):
        return []

    files = [f for f in os.listdir(CHATS_DIR) if f.endswith('.json')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CHATS_DIR, x)), reverse=True)

    session_list = []
    for f in files:
        sid = f.replace(".json", "")
        try:
            sanitize_session_id(sid)
            with open(os.path.join(CHATS_DIR, f), "r") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    title = get_session_title(data)
                else:
                    title = data.get("title", get_session_title(data.get("messages", [])))
                session_list.append({"id": sid, "title": title})
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Skipping invalid session file {f}: {e}")
    return session_list


def save_last_session(session_id: str) -> None:
    """Persist the last used session ID to disk."""
    try:
        sanitize_session_id(session_id)
        with open(LAST_SESSION_FILE, 'w') as f:
            f.write(session_id)
    except Exception as e:
        logger.debug(f"Could not save last session: {e}")


def get_last_session() -> Optional[str]:
    """Restore the last used session ID from disk."""
    try:
        if os.path.exists(LAST_SESSION_FILE):
            with open(LAST_SESSION_FILE, 'r') as f:
                session_id = f.read().strip()
            sanitize_session_id(session_id)
            if os.path.exists(os.path.join(CHATS_DIR, f"{session_id}.json")):
                return session_id
    except Exception as e:
        logger.debug(f"Could not load last session: {e}")
    return None


def load_session(session_id: str) -> list:
    """Load a session's messages from disk with validation and file locking."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on load: {e}")
        return []

    path = os.path.join(CHATS_DIR, f"{session_id}.json")

    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt detected: {path}")
        return []

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read().strip()
                    if not content:
                        return []
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data[:MAX_MESSAGES_PER_SESSION]
                    return data.get("messages", [])[:MAX_MESSAGES_PER_SESSION]
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
    return []


def save_session(session_id: str, messages: list) -> None:
    """Save session messages to disk with file locking and restricted permissions."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on save: {e}")
        return

    path = os.path.join(CHATS_DIR, f"{session_id}.json")

    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt on save: {path}")
        return

    messages = messages[:MAX_MESSAGES_PER_SESSION]
    try:
        data = {
            "title": get_session_title(messages),
            "timestamp": str(datetime.now()),
            "messages": [],
        }
        for msg in messages:
            msg_copy = msg.copy()
            if "image" in msg_copy and isinstance(msg_copy["image"], bytes):
                msg_copy["image"] = base64.b64encode(msg_copy["image"]).decode('utf-8')
                msg_copy["is_image_base64"] = True
            data["messages"].append(msg_copy)

        # Atomic replace: write to a temp file (created 0o600) and rename over
        # the target, so concurrent readers never observe a truncated file and
        # the data is never on disk with default-umask permissions.
        tmp_path = f"{path}.tmp"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {sanitize_error_message(e)}")


def find_empty_session() -> Optional[str]:
    """Return the ID of an existing empty session if one exists."""
    for session in get_session_files():
        if len(load_session(session["id"])) == 0:
            return session["id"]
    return None


def create_new_session() -> str:
    """Create a new chat session, or reuse an existing empty one."""
    empty_session = find_empty_session()
    if empty_session:
        st.session_state.current_session = empty_session
        st.session_state.messages = []
        save_last_session(empty_session)
        return empty_session

    session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    st.session_state.current_session = session_id
    st.session_state.messages = []
    save_session(session_id, [])
    save_last_session(session_id)
    return session_id


def delete_session(session_id: str) -> None:
    """Delete a chat session file with validation."""
    try:
        session_id = sanitize_session_id(session_id)
    except ValueError as e:
        logger.error(f"Invalid session ID on delete: {e}")
        return

    path = os.path.join(CHATS_DIR, f"{session_id}.json")

    if not validate_path_within_directory(Path(path), Path(CHATS_DIR)):
        logger.error(f"Path traversal attempt on delete: {path}")
        return

    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Deleted session: {session_id}")
