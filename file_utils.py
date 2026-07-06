"""File reading, upload processing, and directory scanning."""
import logging
import os
import time
from pathlib import Path
from typing import List

import pypdf
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import (
    IGNORED_DIRS,
    IGNORED_FILE_EXTENSIONS,
    IGNORED_FILES,
    IGNORED_SUFFIXES,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_SCAN,
    UPLOAD_DIR,
    UPLOAD_RETENTION_HOURS,
)
from security import sanitize_filename, validate_path_within_directory

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.cpp': Language.CPP,
    '.c': Language.CPP,
    '.html': Language.HTML,
    '.css': None,
    '.md': Language.MARKDOWN,
    '.txt': None,
    '.pdf': 'pdf',
}


def read_file_content(file_path: Path) -> str:
    """Read text content from a file with size guard and multi-encoding fallback."""
    try:
        # Align with the upload limit — a file that passed the MAX_FILE_SIZE_MB
        # check must not be silently dropped here by a smaller hidden cap.
        if file_path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.info(f"Skipping {file_path.name}: exceeds {MAX_FILE_SIZE_MB} MB limit")
            return ""
        if file_path.suffix.lower() == '.pdf':
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
    except Exception:
        pass
    return ""


def get_text_splitter(file_ext: str) -> RecursiveCharacterTextSplitter:
    """Return a language-aware text splitter for the given file extension."""
    language = SUPPORTED_EXTENSIONS.get(file_ext.lower())
    if language and language != 'pdf':
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=1000, chunk_overlap=200
            )
        except Exception:
            pass
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def process_uploaded_files(uploaded_files) -> List[Document]:
    """Validate, write, and chunk uploaded Streamlit file objects into Documents."""
    documents: List[Document] = []
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            safe_filename = sanitize_filename(uploaded_file.name)
            file_size = len(uploaded_file.getbuffer())
            max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

            if file_size > max_size_bytes:
                st.warning(f"Skipped {safe_filename}: exceeds {MAX_FILE_SIZE_MB} MB limit")
                continue

            file_path = Path(UPLOAD_DIR) / safe_filename
            if not validate_path_within_directory(file_path, Path(UPLOAD_DIR)):
                logger.error(f"Path traversal attempt in upload: {uploaded_file.name}")
                continue

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            os.chmod(file_path, 0o600)

            content = read_file_content(file_path)
            if content.strip():
                splitter = get_text_splitter(file_path.suffix)
                for j, chunk in enumerate(splitter.split_text(content)):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": str(file_path), "filename": safe_filename, "chunk": j},
                    ))
        except ValueError as e:
            logger.warning(f"Invalid filename {uploaded_file.name}: {e}")
            st.warning(f"Skipped invalid file: {uploaded_file.name}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()
    return documents


def cleanup_old_uploads() -> int:
    """Remove uploaded files older than UPLOAD_RETENTION_HOURS.

    Sessions get expiry cleanup; without this, uploads/ grows forever (slow
    disk leak and a privacy tail of stale user documents). Called on startup.
    Returns the number of files removed.
    """
    if not os.path.isdir(UPLOAD_DIR):
        return 0

    removed = 0
    cutoff = time.time() - UPLOAD_RETENTION_HOURS * 3600
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
                logger.info(f"Removed expired upload: {name}")
        except OSError as e:
            logger.warning(f"Could not clean upload {name}: {e}")
    return removed


def scan_directory(root_path: str) -> List[Document]:
    """Recursively scan a directory for supported files, with security limits."""
    if not root_path or not root_path.strip():
        st.warning("Please provide a valid folder path")
        return []

    root = Path(root_path)
    if not root.exists():
        return []

    root_resolved = root.resolve()
    # Component-wise containment (not string prefix): '/etcetera' must not match
    # '/etc', and scanning '/' itself would walk every system directory below.
    sensitive = ['/etc', '/var', '/usr', '/bin', '/sbin', '/root', '/boot',
                 '/sys', '/proc', '/dev', '/run']
    if str(root_resolved) == '/' or any(
        root_resolved.is_relative_to(p) for p in sensitive
    ):
        st.warning("Cannot scan system directories for security reasons")
        return []

    valid_files: List[Path] = []
    for current_root, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRS and not d.startswith('.') and not d.endswith(IGNORED_SUFFIXES)
        ]
        for file in files:
            if file in IGNORED_FILES:
                continue
            fp = Path(file)
            suffix = fp.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS or suffix in IGNORED_FILE_EXTENSIONS:
                continue
            file_path = Path(current_root) / file
            if validate_path_within_directory(file_path, root):
                valid_files.append(file_path)
        if len(valid_files) >= MAX_FILES_PER_SCAN:
            st.warning(f"Scan limited to {MAX_FILES_PER_SCAN} files")
            break

    if not valid_files:
        return []

    documents: List[Document] = []
    progress_bar = st.progress(0)
    for i, file_path in enumerate(valid_files):
        content = read_file_content(file_path)
        if content.strip():
            splitter = get_text_splitter(file_path.suffix)
            for j, chunk in enumerate(splitter.split_text(content)):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": str(file_path), "filename": file_path.name, "chunk": j},
                ))
        if i % max(1, len(valid_files) // 20) == 0:
            progress_bar.progress((i + 1) / len(valid_files))
    progress_bar.empty()
    return documents
