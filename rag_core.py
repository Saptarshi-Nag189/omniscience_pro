"""RAG core: embeddings, LLM loading, vectorstore operations, and query parsing."""
import logging
import os
import re
import shutil
import urllib.request
from typing import List, Tuple

import chromadb
import streamlit as st
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import DB_DIRECTORY, EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


# ── Embeddings ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


# ── LLM ──────────────────────────────────────────────────────────────────────
# Chat-model construction now lives in providers.build_chat_llm, which supports
# Ollama plus optional cloud providers behind a uniform str-output interface.


# ── Vectorstore ───────────────────────────────────────────────────────────────

def initialize_vectorstore(embeddings, force_recreate: bool = False):
    """Initialize or recreate the ChromaDB vector store."""
    try:
        if force_recreate and os.path.exists(DB_DIRECTORY):
            shutil.rmtree(DB_DIRECTORY)
            st.toast("Database cleared")
        client = chromadb.PersistentClient(
            path=DB_DIRECTORY,
            settings=Settings(anonymized_telemetry=False),
        )
        return Chroma(
            client=client, collection_name="omniscience",
            embedding_function=embeddings,
        )
    except Exception:
        return None


def ingest_documents(vectorstore, documents: List[Document]) -> None:
    """Batch-add documents to the vector store."""
    if not documents:
        return
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vectorstore.add_documents(documents[i:i + batch_size])
    st.success(f"Indexed {len(documents)} chunks")


def delete_file_from_db(vectorstore, filename: str) -> None:
    """Remove all chunks for a given filename from the vector store."""
    try:
        vectorstore._collection.delete(where={"filename": filename})
        st.toast(f"Deleted: {filename}")
    except Exception:
        pass


def get_all_filenames(vectorstore) -> List[str]:
    """Return unique filenames stored in the vector store metadata."""
    try:
        data = vectorstore._collection.get(include=['metadatas'])
        return list({m['filename'] for m in data['metadatas'] if 'filename' in m})
    except Exception:
        return []


# ── Ollama model discovery ────────────────────────────────────────────────────

def list_ollama_models() -> List[str]:
    """Return model names available in Ollama, or [] on connection error."""
    import json as _json
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = _json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_loaded_documents(vectorstore) -> List[str]:
    """Return unique source filenames currently in the vectorstore."""
    try:
        data = vectorstore._collection.get(include=["metadatas"])
        sources = {
            m.get("filename") or m.get("source", "")
            for m in data["metadatas"]
        }
        return sorted(s for s in sources if s)
    except Exception:
        return []


# ── File mention parsing ──────────────────────────────────────────────────────

def parse_file_mentions(query: str) -> Tuple[List[str], str]:
    """Parse @filename mentions from a query string.

    Returns (list_of_mentions, cleaned_query_without_mentions).
    """
    pattern = r'@"([^"]+)"|@(\S+)'
    mentions = [m.group(1) or m.group(2) for m in re.finditer(pattern, query)]
    clean_query = ' '.join(re.sub(pattern, '', query).split())
    return mentions, clean_query if clean_query else query


def fuzzy_match_filenames(mentions: List[str], available_files: List[str]) -> List[str]:
    """Match @mentions against available vectorstore files (case-insensitive, partial match)."""
    matched = []
    for mention in mentions:
        mention_lower = mention.lower()
        for filepath in available_files:
            filename = os.path.basename(filepath).lower()
            if filename == mention_lower:
                matched.append(filepath)
                continue
            if mention_lower in filename or filename in mention_lower:
                matched.append(filepath)
                continue
            name_no_ext = os.path.splitext(filename)[0]
            mention_no_ext = os.path.splitext(mention_lower)[0]
            if name_no_ext == mention_no_ext:
                matched.append(filepath)
                continue
            if filepath.lower().endswith(mention_lower):
                matched.append(filepath)
    return list(set(matched))
