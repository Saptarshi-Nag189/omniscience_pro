"""
Omniscience Pro — Local RAG System

Streamlit entry point. All application logic lives in the project modules:
  config.py        — environment constants and directory setup
  security.py      — sanitizers, rate limiter, error redaction
  session.py       — session persistence and lifecycle management
  ui_components.py — CSS theme, streaming handler, clipboard helper
  file_utils.py    — file reading, upload processing, directory scanning
  rag_core.py      — embeddings, vectorstore, query parsing
  providers.py     — LLM provider catalogue and factory
  vision.py        — multimodal image analysis
  search.py        — web search and academic search integrations
  sql_mode.py      — natural-language SQLite querying

This file is organised as small render/handler functions orchestrated by
main(): _render_sidebar() → _render_history() → _handle_vision_mode() /
_handle_chat_input().
"""

import base64
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import streamlit as st

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

from config import DB_DIRECTORY, MAX_FILE_SIZE_MB, UPLOAD_DIR
from file_utils import cleanup_old_uploads, process_uploaded_files, scan_directory
from providers import (
    PROVIDERS,
    build_chat_llm,
    format_model_label,
    provider_available,
)
from rag_core import (
    delete_file_from_db,
    fuzzy_match_filenames,
    get_all_filenames,
    get_loaded_documents,
    ingest_documents,
    initialize_vectorstore,
    list_ollama_models,
    load_embeddings,
    parse_file_mentions,
)
from search import (
    HAS_ARXIV,
    HAS_SEMANTIC_SCHOLAR,
    HAS_WEB_SEARCH,
    run_academic_search,
    run_web_search,
)
from security import (
    check_rate_limit,
    redact_secrets,
    sanitize_error_message,
    sanitize_filename,
)
from session import (
    cleanup_expired_sessions,
    create_new_session,
    delete_session,
    get_last_session,
    get_session_files,
    load_session,
    save_last_session,
    save_session,
)
from sql_mode import query_sqlite_db
from streaming import LLMWorker, QueueStreamHandler
from ui_components import (
    PURPLE_THEME_CSS,
    SQL_PULSE_JS,
    THINKING_HTML,
    VISION_PULSE_JS,
    StreamHandler,
    _get_startup_marker,
    build_conversation_history,
    copy_to_clipboard,
)
from vision import BytesWrapper, process_vision_request

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LLM SELECTION
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LLMSelection:
    """The sidebar's provider/model choice. Lives in st.session_state only —
    api_key is never persisted to disk or logs."""

    provider_name: str = "Ollama (Local)"
    provider_type: str = "ollama"
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def _current_selection() -> LLMSelection:
    sel = st.session_state.get("llm_selection")
    return sel if isinstance(sel, LLMSelection) else LLMSelection()


def _make_llm(callback=None) -> "Optional[Runnable]":
    """Build the chat LLM from the current sidebar provider/model selection."""
    sel = _current_selection()
    return build_chat_llm(
        sel.provider_type,
        sel.model,
        api_key=sel.api_key,
        base_url=sel.base_url,
        callback=callback,
    )


# ═══════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def _build_prompt(query, history_parts, rag_context=None,
                  web_results="", academic_results=""):
    """Build the LLM prompt in one place.

    With ``rag_context`` supplied, returns the unified RAG template (local
    context + optional web/academic sources). Without it, returns the
    augmented-search template used when no vectorstore is available.
    """
    if rag_context is not None:
        conversation_history = "\n\n".join(history_parts)
        prompt_text = f"""You are Omniscience, an AI assistant.

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
            prompt_text += f"""==============================
WEB RESULTS
==============================
{web_results}

"""
        if academic_results:
            prompt_text += f"""==============================
ACADEMIC RESULTS
==============================
{academic_results}

"""
        prompt_text += f"""USER QUESTION:
{query}

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
        return prompt_text

    return f"""Answer the question using the sources below.

CONVERSATION HISTORY:
{chr(10).join(history_parts) if history_parts else "(None)"}

ACADEMIC RESULTS:
{academic_results if academic_results else "(None)"}

WEB RESULTS:
{web_results if web_results else "(None)"}

QUESTION:
{query}

RULES:
- Use conversation history only to understand follow-up questions
- Prefer academic results when available
- Ignore irrelevant web results
- Do not invent information
- If the sources do not answer the question, say so clearly

ANSWER:"""


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

def _init_session_state():
    """Create a new chat on fresh start, restore the last one on refresh."""
    startup_marker = _get_startup_marker()

    if 'current_session' not in st.session_state:
        if not startup_marker["session_created"]:
            startup_marker["session_created"] = True
            create_new_session()
        else:
            sessions = get_session_files()
            if sessions:
                last_session = get_last_session()
                if last_session and any(s["id"] == last_session for s in sessions):
                    st.session_state.current_session = last_session
                else:
                    st.session_state.current_session = sessions[0]["id"]
                st.session_state.messages = load_session(st.session_state.current_session)
            else:
                create_new_session()

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def _render_search_toggles():
    if 'web_search_enabled' not in st.session_state:
        st.session_state.web_search_enabled = False

    st.session_state.web_search_enabled = st.toggle(
        "Augment with Web Search",
        value=st.session_state.web_search_enabled,
        help="When enabled, responses will be augmented with web search results",
    )

    if st.session_state.web_search_enabled and not HAS_WEB_SEARCH:
        st.warning("Web search unavailable. Install: pip install duckduckgo-search")

    if 'academic_search_enabled' not in st.session_state:
        st.session_state.academic_search_enabled = False

    st.session_state.academic_search_enabled = st.toggle(
        "Academic Research",
        value=st.session_state.academic_search_enabled,
        help="Search Semantic Scholar, arXiv, OpenAlex for academic papers",
    )

    if st.session_state.academic_search_enabled and not (HAS_SEMANTIC_SCHOLAR or HAS_ARXIV):
        st.warning("Install: pip install semanticscholar arxiv")


def _render_chat_export():
    if not st.session_state.messages:
        return

    md_content = (
        f"# Chat Export\n\n**Session:** {st.session_state.current_session}\n"
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    )
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
        use_container_width=True,
    )


def _render_session_controls():
    st.markdown("### CHAT SESSIONS")

    if st.button("+ NEW CHAT", use_container_width=True):
        create_new_session()
        st.rerun()

    sessions = get_session_files()
    session_ids = [s["id"] for s in sessions]
    session_titles = [s["title"] for s in sessions]

    if not session_ids:
        return

    try:
        idx = session_ids.index(st.session_state.current_session)
    except ValueError:
        idx = 0

    selected_idx = st.selectbox(
        "History",
        range(len(session_titles)),
        format_func=lambda x: session_titles[x],
        index=idx,
    )

    selected_id = session_ids[selected_idx]

    if selected_id != st.session_state.current_session:
        st.session_state.current_session = selected_id
        st.session_state.messages = load_session(selected_id)
        save_last_session(selected_id)
        st.rerun()

    if st.button("DELETE CHAT", type="primary"):
        delete_session(st.session_state.current_session)
        remaining_sessions = get_session_files()
        if remaining_sessions:
            st.session_state.current_session = remaining_sessions[0]["id"]
            st.session_state.messages = load_session(remaining_sessions[0]["id"])
        else:
            create_new_session()
        st.rerun()


def _render_model_selector(mode: str):
    """Provider → model (stars + tags) → API key → base URL. Stores the result
    as an LLMSelection in st.session_state (in-memory only)."""
    st.markdown("#### MODEL")
    is_vision_mode = mode == "Vision (Images)"

    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
    pconf = PROVIDERS[provider_name]
    provider_type = pconf["type"]

    if not provider_available(provider_name):
        st.warning(f"{provider_name} needs: `pip install {pconf['pip']}`")

    # Model picker — catalogue entries are advisory (stars + tags); users can
    # always type a custom model id via the "Custom model…" sentinel.
    catalogue = pconf["vision_models"] if is_vision_mode else pconf["models"]
    _CUSTOM = "✏️ Custom model…"
    if catalogue:
        ids = [m["id"] for m in catalogue]
        meta_by_id = {m["id"]: m for m in catalogue}
        choice = st.selectbox(
            "Vision Model" if is_vision_mode else "Model",
            options=ids + [_CUSTOM],
            index=0,
            format_func=lambda x: x if x == _CUSTOM else format_model_label(meta_by_id[x]),
        )
        model_name = st.text_input("Custom model name", value="").strip() if choice == _CUSTOM else choice
    else:
        model_name = st.text_input(
            "Vision Model" if is_vision_mode else "Model name", value="",
            placeholder="e.g. gpt-4o or your custom model",
        ).strip()

    # API key — kept in-memory only (never written to session files / logs).
    api_key = None
    if pconf["needs_key"]:
        env_name = pconf.get("env")
        env_key = os.environ.get(env_name, "") if env_name else ""
        typed_key = st.text_input(
            f"{provider_name} API Key", type="password",
            help=f"Or set the {env_name} environment variable." if env_name else None,
        )
        api_key = typed_key or env_key
        if not typed_key and env_key:
            st.caption(f"Using key from `{env_name}` environment variable.")
        if not api_key:
            st.warning("Enter an API key to use this provider.")

    base_url = None
    if pconf["needs_base_url"]:
        base_url = st.text_input(
            "Base URL", value="",
            placeholder="https://api.example.com/v1",
            help="OpenAI-compatible endpoint (Groq, OpenRouter, vLLM, LM Studio, …).",
        ).strip() or None

    st.session_state.llm_selection = LLMSelection(
        provider_name=provider_name,
        provider_type=provider_type,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    # Local Ollama: warn when the chosen model isn't pulled yet.
    if provider_type == "ollama" and model_name:
        available_models = list_ollama_models()
        if available_models and model_name not in available_models:
            st.warning(f"Model **{model_name}** not found in Ollama. Run: `ollama pull {model_name}`")


def _render_rag_data_source():
    st.markdown("#### DATA SOURCE")
    root_path = st.text_input("Folder Path", value=".")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("SCAN"):
            with st.spinner("Scanning folder..."):
                st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)
                if st.session_state.vectorstore is None:
                    st.error("Could not initialize the vector database. Check the logs and try again.")
                else:
                    docs = scan_directory(root_path)
                    ingest_documents(st.session_state.vectorstore, docs)
                    st.success(f"✅ Scanned {len(docs)} documents from {root_path}")
    with c2:
        if st.button("PURGE"):
            initialize_vectorstore(load_embeddings(), True)
            st.session_state.vectorstore = None
            st.info("🗑️ Vector database purged")

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

    if uploaded_files:
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.svg'}
        image_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in image_extensions]
        if image_files:
            st.warning(
                f"Detected {len(image_files)} image file(s). "
                "For image analysis, please use **Vision (Images)** mode in the sidebar."
            )
            st.markdown(VISION_PULSE_JS, unsafe_allow_html=True)
            uploaded_files = [f for f in uploaded_files if Path(f.name).suffix.lower() not in image_extensions]

    if uploaded_files:
        sql_extensions = {'.sql', '.db', '.sqlite', '.sqlite3'}
        sql_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in sql_extensions]
        if sql_files:
            st.warning(
                f"Detected {len(sql_files)} database/SQL file(s). "
                "For database queries, please use **Database (SQL)** mode in the sidebar."
            )
            st.markdown(SQL_PULSE_JS, unsafe_allow_html=True)
            uploaded_files = [f for f in uploaded_files if Path(f.name).suffix.lower() not in sql_extensions]

    if uploaded_files and st.button("PROCESS"):
        st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)
        if st.session_state.vectorstore is None:
            st.error("Could not initialize the vector database. Check the logs and try again.")
        else:
            docs = process_uploaded_files(uploaded_files)
            ingest_documents(st.session_state.vectorstore, docs)

    if st.session_state.vectorstore:
        loaded_docs = get_loaded_documents(st.session_state.vectorstore)
        if loaded_docs:
            with st.expander(f"Loaded documents ({len(loaded_docs)})"):
                for doc in loaded_docs:
                    st.markdown(f"- `{os.path.basename(doc)}`")

    with st.expander("Manage Knowledge Base"):
        if st.session_state.vectorstore:
            all_files = get_all_filenames(st.session_state.vectorstore)
            if all_files:
                del_file = st.selectbox("Delete File:", options=all_files)
                if st.button("DELETE"):
                    delete_file_from_db(st.session_state.vectorstore, del_file)
                    st.rerun()


def _render_sql_source():
    st.markdown("#### SQL SOURCE")
    uploaded_db = st.file_uploader("Upload SQLite DB", type=['db', 'sqlite', 'sqlite3'])
    if not uploaded_db:
        return
    try:
        safe_dbname = sanitize_filename(uploaded_db.name)
        db_path = os.path.join(UPLOAD_DIR, safe_dbname)
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


def _render_sidebar() -> str:
    """Render the full sidebar; returns the selected mode."""
    with st.sidebar:
        st.markdown("### SYSTEM CONFIG")
        mode = st.radio("Mode", ["Chat (RAG)", "Vision (Images)", "Database (SQL)"], index=0)

        _render_search_toggles()
        st.markdown("---")
        _render_chat_export()
        _render_session_controls()
        st.markdown("---")
        _render_model_selector(mode)

        if mode == "Chat (RAG)":
            _render_rag_data_source()
        if mode == "Database (SQL)":
            _render_sql_source()

    return mode


# ═══════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════

def _render_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                image_data = message["image"]
                if message.get("is_image_base64"):
                    try:
                        image_data = base64.b64decode(image_data)
                    except Exception as e:
                        logger.warning(f"Failed to decode stored image, showing raw value: {e}")
                st.image(image_data, caption="Uploaded Image", use_container_width=True)

            st.markdown(message["content"])

            if "sources" in message:
                with st.expander("Sources"):
                    for s in message["sources"]:
                        st.markdown(f"- `{s}`")

            if message["role"] == "assistant":
                copy_to_clipboard(message["content"], label="Copy Response")


# ═══════════════════════════════════════════════════════════════════
# VISION MODE
# ═══════════════════════════════════════════════════════════════════

def _handle_vision_mode():
    img_file = st.file_uploader("Upload Image to Analyze", type=["png", "jpg", "jpeg", "webp"])
    if img_file and (prompt := st.chat_input("Ask about this image...")):
        image_bytes = img_file.getvalue()
        # Bound the image the same way RAG/SQL uploads are bounded — otherwise the
        # raw bytes are base64-encoded into the session JSON and shipped to the
        # provider unbounded (memory, disk and payload blowup).
        if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"Image exceeds the {MAX_FILE_SIZE_MB} MB limit. Please upload a smaller image.")
            return
        st.session_state.messages.append({"role": "user", "content": prompt, "image": image_bytes})
        save_session(st.session_state.current_session, st.session_state.messages)
        save_last_session(st.session_state.current_session)
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "image" in st.session_state.messages[-1]:
        last_msg = st.session_state.messages[-1]
        image_bytes = last_msg["image"]
        prompt_text = last_msg["content"]

        with st.chat_message("assistant"):
            # Vision is the most expensive request type — gate it on the same
            # rate limiter as chat and SQL so it can't bypass the cost control.
            if not check_rate_limit("vision_request"):
                st.error("Rate limit exceeded. Please wait a moment before analyzing another image.")
                return
            with st.spinner("Analyzing image..."):
                sel = _current_selection()
                response_content = process_vision_request(
                    BytesWrapper(image_bytes), prompt_text,
                    model_name=sel.model or "llava",
                    provider_type=sel.provider_type,
                    api_key=sel.api_key,
                    base_url=sel.base_url,
                )
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                save_session(st.session_state.current_session, st.session_state.messages)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════
# CHAT / SQL MODE
# ═══════════════════════════════════════════════════════════════════

def _answer_sql(prompt: str, llm, thinking_placeholder, response_placeholder) -> str:
    thinking_placeholder.empty()
    if st.session_state.get('db_path'):
        response_content = query_sqlite_db(st.session_state.db_path, prompt, llm)
    else:
        response_content = "Please upload a database file first."
    response_placeholder.markdown(response_content)
    return response_content


def _prepare_rag(prompt: str) -> tuple[str, list, list]:
    """Build the prompt to stream for a chat/RAG turn (runs on the main thread).

    Does retrieval, @mention filtering, and optional web/academic augmentation,
    then returns ``(prompt_to_stream, sources, notices)`` where ``notices`` is a
    list of ``(level, message)`` pairs to render alongside the streamed answer.
    The actual token streaming is done by the caller in a background thread.
    """
    web_results = ""
    academic_results = ""
    sources: list = []
    notices: list = []

    if st.session_state.web_search_enabled and HAS_WEB_SEARCH:
        web_results = run_web_search(prompt)

    if st.session_state.vectorstore:
        file_mentions, clean_query = parse_file_mentions(prompt)

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 16 if file_mentions else 8}
        )
        retrieved_docs = retriever.invoke(clean_query if clean_query else prompt)

        if file_mentions:
            available_files = get_all_filenames(st.session_state.vectorstore)
            matched_files = fuzzy_match_filenames(file_mentions, available_files)

            if matched_files:
                filtered_docs = [
                    doc for doc in retrieved_docs
                    if doc.metadata.get('source', '') in matched_files
                    or doc.metadata.get('filename', '') in matched_files
                ]
                if filtered_docs:
                    retrieved_docs = filtered_docs
                    notices.append(("info", f"📎 Focused on: {', '.join([os.path.basename(f) for f in matched_files[:5]])}"))
                else:
                    notices.append(("warning", "⚠️ No content found in mentioned files. Showing general results."))
            else:
                notices.append(("warning", f"⚠️ Could not find files matching: {', '.join(file_mentions)}"))

        rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

        if st.session_state.academic_search_enabled:
            extraction_llm = _make_llm()
            academic_results = run_academic_search(
                prompt, rag_context=rag_context, llm=extraction_llm
            )

        history_parts = build_conversation_history(st.session_state.messages)
        prompt_to_stream = _build_prompt(
            prompt, history_parts, rag_context=rag_context,
            web_results=web_results, academic_results=academic_results,
        )
    else:
        # No vectorstore — use LLM with optional search results.
        history_parts = build_conversation_history(st.session_state.messages)
        if history_parts or academic_results or web_results:
            prompt_to_stream = _build_prompt(
                prompt, history_parts,
                web_results=web_results, academic_results=academic_results,
            )
        else:
            prompt_to_stream = prompt

    external_sources = []
    if web_results:
        external_sources.append("Web Search (DuckDuckGo)")
    if academic_results:
        external_sources.append("Academic (arXiv, Semantic Scholar, OpenAlex)")
    if external_sources:
        notices.append(("info", f"Sources: {', '.join(external_sources)}"))

    return prompt_to_stream, sources, notices


def _report_model_load_failure():
    sel = _current_selection()
    if sel.provider_type == "ollama":
        st.error("Failed to load the model. Please check if Ollama is running and the model is installed.")
    elif not sel.api_key:
        st.error("Failed to load the model. Please enter a valid API key for the selected provider.")
    else:
        st.error("Failed to load the model. Check the provider package is installed, the model name, and your API key.")


def _append_assistant(content: str, sources=None) -> None:
    """Record an assistant message and persist the session."""
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "sources": sources or []}
    )
    save_session(st.session_state.current_session, st.session_state.messages)


def _render_notices(notices) -> None:
    for level, text in notices:
        (st.info if level == "info" else st.warning)(text)


def _run_sql_synchronously(prompt: str) -> None:
    """Answer a SQL turn synchronously.

    SQL is a multi-step generate→run→summarise chain rather than a single token
    stream, so it is not interruptible and shows no Stop button.
    """
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(THINKING_HTML, unsafe_allow_html=True)
        response_placeholder = st.empty()
        llm = _make_llm(StreamHandler(response_placeholder, thinking_placeholder=thinking))
        if not llm:
            thinking.empty()
            _report_model_load_failure()
            content = "Failed to load the model."
        else:
            try:
                content = _answer_sql(prompt, llm, thinking, response_placeholder)
            except Exception as e:
                thinking.empty()
                logger.error(f"Error processing request: {redact_secrets(e)}")
                st.error(f"Error: {sanitize_error_message(e)}")
                content = f"Error: {sanitize_error_message(e)}"
    _append_assistant(content)
    st.rerun()


def _start_generation(mode: str, prompt: str) -> None:
    """Begin answering the pending user turn.

    SQL, rate-limit and setup failures resolve synchronously (they append an
    assistant message and rerun). A chat/RAG turn spawns a background
    :class:`LLMWorker` and stores it in ``st.session_state.gen`` for the
    polling fragment to drive.
    """
    if not check_rate_limit("llm_request"):
        msg = "Rate limit exceeded. Please wait a moment before sending another message."
        with st.chat_message("assistant"):
            st.error(msg)
        _append_assistant(msg)
        st.rerun()
        return

    if mode == "Database (SQL)":
        _run_sql_synchronously(prompt)
        return

    worker = LLMWorker()
    llm = _make_llm(QueueStreamHandler(worker.queue, worker.stop_event))
    if not llm:
        with st.chat_message("assistant"):
            _report_model_load_failure()
        _append_assistant("Failed to load the model.")
        st.rerun()
        return

    try:
        with st.spinner("Retrieving context..."):
            prompt_to_stream, sources, notices = _prepare_rag(prompt)
    except Exception as e:
        with st.chat_message("assistant"):
            logger.error(f"Error preparing request: {redact_secrets(e)}")
            st.error(f"Error: {sanitize_error_message(e)}")
        _append_assistant(f"Error: {sanitize_error_message(e)}")
        st.rerun()
        return

    worker.start(llm, prompt_to_stream)
    st.session_state.gen = {
        "worker": worker,
        "sources": sources,
        "notices": notices,
        "accumulated": "",
    }


def _finalize_generation(*, stopped: bool = False) -> None:
    """Persist the completed (or stopped) generation as an assistant message."""
    gen = st.session_state.pop("gen", None)
    if not gen:
        return
    worker = gen["worker"]
    if worker.error is not None:
        logger.error(f"Error processing request: {redact_secrets(worker.error)}")
        _append_assistant(f"Error: {sanitize_error_message(worker.error)}")
        return
    text = worker.result if (not stopped and worker.result) else gen["accumulated"]
    if stopped:
        _append_assistant((text or "(Generation stopped)") + "\n\n*[Generation stopped by user]*",
                          gen["sources"])
    else:
        _append_assistant(text or "(No response)", gen["sources"])


@st.fragment
def _run_generation_fragment() -> None:
    """Poll the background worker and render streamed tokens with a working Stop
    button. Self-reruns at fragment scope (so the Stop click is processed
    between polls) until the worker finishes, then does a full rerun to hand
    off to ``_render_history``.
    """
    gen = st.session_state.get("gen")
    if not gen:
        return
    worker = gen["worker"]

    with st.chat_message("assistant"):
        _render_notices(gen["notices"])

        if st.button("⏹ Stop Generation", key="stop_generation"):
            worker.stop()
            gen["accumulated"] += worker.drain()
            _finalize_generation(stopped=True)
            st.rerun()
            return

        placeholder = st.empty()
        gen["accumulated"] += worker.drain()

        if worker.done:
            gen["accumulated"] += worker.drain()
            _finalize_generation(stopped=worker.stopped)
            st.rerun()
            return

        if gen["accumulated"]:
            placeholder.markdown(gen["accumulated"] + "▌")
        else:
            placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

        time.sleep(0.08)
        st.rerun(scope="fragment")


def _handle_chat_input(mode: str) -> None:
    if mode == "Vision (Images)":
        return

    # A generation in flight takes priority: keep polling it and ignore new
    # input until it finishes (a full rerun re-enters here after each poll).
    if st.session_state.get("gen") is not None:
        _run_generation_fragment()
        return

    prompt = st.chat_input("Enter query...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_session(st.session_state.current_session, st.session_state.messages)
        st.rerun()

    msgs = st.session_state.messages
    pending = bool(msgs) and msgs[-1]["role"] == "user" and "image" not in msgs[-1]
    if not pending:
        return

    _start_generation(mode, msgs[-1]["content"])
    if st.session_state.get("gen") is not None:
        _run_generation_fragment()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Omniscience Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PURPLE_THEME_CSS, unsafe_allow_html=True)

    st.markdown('<div class="custom-title">OMNISCIENCE PRO</div>', unsafe_allow_html=True)
    st.markdown("##### Local RAG System // Offline Mode")
    st.markdown("---")

    _init_session_state()
    mode = _render_sidebar()

    if mode == "Chat (RAG)" and st.session_state.vectorstore is None and os.path.exists(DB_DIRECTORY):
        st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)

    _render_history()

    if mode == "Vision (Images)":
        _handle_vision_mode()

    _handle_chat_input(mode)


if __name__ == "__main__":
    cleanup_expired_sessions()
    cleanup_old_uploads()
    main()
