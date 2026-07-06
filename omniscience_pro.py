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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

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


def _make_llm(callback=None):
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
                    except Exception:
                        pass
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
        st.session_state.messages.append({"role": "user", "content": prompt, "image": image_bytes})
        save_session(st.session_state.current_session, st.session_state.messages)
        save_last_session(st.session_state.current_session)
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "image" in st.session_state.messages[-1]:
        last_msg = st.session_state.messages[-1]
        image_bytes = last_msg["image"]
        prompt_text = last_msg["content"]

        with st.chat_message("assistant"):
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


def _answer_rag(prompt: str, llm, thinking_placeholder, response_placeholder):
    """Answer a chat prompt via RAG (or plain LLM + optional search results).

    Returns (response_content, sources).
    """
    web_results = ""
    academic_results = ""
    sources = []

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
                    st.info(f"📎 Focused on: {', '.join([os.path.basename(f) for f in matched_files[:5]])}")
                else:
                    st.warning("⚠️ No content found in mentioned files. Showing general results.")
            else:
                st.warning(f"⚠️ Could not find files matching: {', '.join(file_mentions)}")

        rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))

        if st.session_state.academic_search_enabled:
            extraction_llm = _make_llm()
            academic_results = run_academic_search(
                prompt, rag_context=rag_context, llm=extraction_llm
            )

        history_parts = build_conversation_history(st.session_state.messages)

        unified_prompt = _build_prompt(
            prompt, history_parts, rag_context=rag_context,
            web_results=web_results, academic_results=academic_results,
        )

        response_content = llm.invoke(unified_prompt)
        thinking_placeholder.empty()
        response_placeholder.markdown(response_content)

        used_external_only = (
            "based on external search" in response_content.lower()
            or "based on web search" in response_content.lower()
        )

        if not used_external_only and sources:
            with st.expander("Code Sources"):
                for s in sources:
                    st.markdown(f"- `{s}`")
        else:
            external_sources = []
            if web_results:
                external_sources.append("Web Search (DuckDuckGo)")
            if academic_results:
                external_sources.append("Academic (arXiv, Semantic Scholar, OpenAlex)")
            if external_sources:
                st.info(f"Sources: {', '.join(external_sources)}")

        copy_to_clipboard(response_content, label="Copy Response")
    else:
        # No vectorstore — use LLM with optional search results
        history_parts = build_conversation_history(st.session_state.messages)

        if history_parts or academic_results or web_results:
            augmented_prompt = _build_prompt(
                prompt, history_parts,
                web_results=web_results, academic_results=academic_results,
            )
            response_content = llm.invoke(augmented_prompt)
        else:
            response_content = llm.invoke(prompt)
        thinking_placeholder.empty()
        response_placeholder.markdown(response_content)

    return response_content, sources


def _report_model_load_failure():
    sel = _current_selection()
    if sel.provider_type == "ollama":
        st.error("Failed to load the model. Please check if Ollama is running and the model is installed.")
    elif not sel.api_key:
        st.error("Failed to load the model. Please enter a valid API key for the selected provider.")
    else:
        st.error("Failed to load the model. Check the provider package is installed, the model name, and your API key.")


def _handle_chat_input(mode: str):
    prompt = st.chat_input("Enter query...")
    if not prompt or mode == "Vision (Images)":
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_session(st.session_state.current_session, st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)

        stop_button_placeholder = st.empty()
        if stop_button_placeholder.button("⏹ Stop Generation", key=f"stop_{len(st.session_state.messages)}"):
            st.session_state.stop_generation = True
            stop_button_placeholder.empty()

        # A stale stop flag from a click after the previous generation finished
        # must not kill this request at its first token.
        st.session_state.stop_generation = False

        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder, thinking_placeholder=thinking_placeholder)
        llm = _make_llm(stream_handler)

        sources = []

        if not check_rate_limit("llm_request"):
            thinking_placeholder.empty()
            stop_button_placeholder.empty()
            st.error("Rate limit exceeded. Please wait a moment before sending another message.")
        elif llm:
            try:
                if mode == "Database (SQL)":
                    response_content = _answer_sql(prompt, llm, thinking_placeholder, response_placeholder)
                else:
                    response_content, sources = _answer_rag(prompt, llm, thinking_placeholder, response_placeholder)

                st.session_state.messages.append({"role": "assistant", "content": response_content, "sources": sources})
                save_session(st.session_state.current_session, st.session_state.messages)
                stop_button_placeholder.empty()
            except StopIteration:
                thinking_placeholder.empty()
                stop_button_placeholder.empty()
                partial_response = stream_handler.text if stream_handler.text else "(Generation stopped)"
                response_placeholder.markdown(partial_response + "\n\n*[Generation stopped by user]*")
                st.session_state.messages.append({"role": "assistant", "content": partial_response, "sources": sources})
                save_session(st.session_state.current_session, st.session_state.messages)
            except Exception as e:
                thinking_placeholder.empty()
                stop_button_placeholder.empty()
                logger.error(f"Error processing request: {redact_secrets(e)}")
                st.error(f"Error: {sanitize_error_message(e)}")
        else:
            thinking_placeholder.empty()
            stop_button_placeholder.empty()
            _report_model_load_failure()


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
