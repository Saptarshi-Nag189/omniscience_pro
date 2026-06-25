"""
Omniscience Pro — Local RAG System

Streamlit entry point. All application logic lives in the project modules:
  config.py        — environment constants and directory setup
  security.py      — sanitizers, rate limiter, error redaction
  session.py       — session persistence and lifecycle management
  ui_components.py — CSS theme, streaming handler, clipboard helper
  file_utils.py    — file reading, upload processing, directory scanning
  rag_core.py      — embeddings, LLM loading, vectorstore, query parsing
  vision.py        — multimodal image analysis
  search.py        — web search and academic search integrations
  sql_mode.py      — natural-language SQLite querying
"""

import os
import base64
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import DB_DIRECTORY, UPLOAD_DIR, MAX_FILE_SIZE_MB
from security import check_rate_limit, sanitize_filename, sanitize_error_message
from session import (
    cleanup_expired_sessions,
    create_new_session, get_session_files, get_last_session,
    load_session, save_session, save_last_session, delete_session,
)
from ui_components import (
    PURPLE_THEME_CSS, VISION_PULSE_JS, SQL_PULSE_JS, THINKING_HTML,
    StreamHandler, _get_startup_marker, copy_to_clipboard, build_conversation_history,
)
from rag_core import (
    load_embeddings, get_llm_for_chain,
    initialize_vectorstore, ingest_documents, delete_file_from_db, get_all_filenames,
    parse_file_mentions, fuzzy_match_filenames,
)
from file_utils import process_uploaded_files, scan_directory
from vision import process_vision_request, BytesWrapper
from search import run_web_search, run_academic_search, HAS_WEB_SEARCH, HAS_SEMANTIC_SCHOLAR, HAS_ARXIV
from sql_mode import query_sqlite_db

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Omniscience Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(PURPLE_THEME_CSS, unsafe_allow_html=True)

    st.markdown('<div class="custom-title">OMNISCIENCE PRO</div>', unsafe_allow_html=True)
    st.markdown("##### Local RAG System // Offline Mode")
    st.markdown("---")

    # Session Management - create new chat on fresh start, restore on refresh
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

    # ════════════ SIDEBAR ════════════
    with st.sidebar:
        st.markdown("### SYSTEM CONFIG")

        mode = st.radio("Mode", ["Chat (RAG)", "Vision (Images)", "Database (SQL)"], index=0)

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

        st.markdown("---")

        if st.session_state.messages:
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

        st.markdown("### CHAT SESSIONS")

        if st.button("+ NEW CHAT", use_container_width=True):
            create_new_session()
            st.rerun()

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

        st.markdown("---")

        if mode == "Vision (Images)":
            model_options = ["llava:7b", "llama3.2-vision"]
            model_name = st.selectbox("Vision Model", options=model_options, index=0)
            st.info(f"Using Vision Model: {model_name}")
        else:
            model_options = ["qwen3:4b", "qwen2.5-coder:7b", "qwen2.5-coder:1.5b", "llama3.2:3b", "mistral:7b"]
            model_name = st.selectbox("Model", options=model_options, index=1)

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

            with st.expander("Manage Knowledge Base"):
                if st.session_state.vectorstore:
                    all_files = get_all_filenames(st.session_state.vectorstore)
                    if all_files:
                        del_file = st.selectbox("Delete File:", options=all_files)
                        if st.button("DELETE"):
                            delete_file_from_db(st.session_state.vectorstore, del_file)
                            st.rerun()

        if mode == "Database (SQL)":
            st.markdown("#### SQL SOURCE")
            uploaded_db = st.file_uploader("Upload SQLite DB", type=['db', 'sqlite', 'sqlite3'])
            if uploaded_db:
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

    # ════════════ MAIN LOGIC ════════════

    if mode == "Chat (RAG)" and st.session_state.vectorstore is None and os.path.exists(DB_DIRECTORY):
        st.session_state.vectorstore = initialize_vectorstore(load_embeddings(), False)

    for idx, message in enumerate(st.session_state.messages):
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

    if mode == "Vision (Images)":
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
                    response_content = process_vision_request(BytesWrapper(image_bytes), prompt_text, model_name)
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    save_session(st.session_state.current_session, st.session_state.messages)
                    st.rerun()

    if prompt := st.chat_input("Enter query..."):
        if mode == "Vision (Images)":
            pass
        else:
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

                response_placeholder = st.empty()
                stream_handler = StreamHandler(response_placeholder, thinking_placeholder=thinking_placeholder)
                llm = get_llm_for_chain(model_name, stream_handler)

                response_content = ""
                sources = []

                if not check_rate_limit("llm_request"):
                    thinking_placeholder.empty()
                    stop_button_placeholder.empty()
                    st.error("Rate limit exceeded. Please wait a moment before sending another message.")
                elif llm:
                    try:
                        if mode == "Database (SQL)":
                            thinking_placeholder.empty()
                            if st.session_state.get('db_path'):
                                response_content = query_sqlite_db(st.session_state.db_path, prompt, llm)
                                response_placeholder.markdown(response_content)
                            else:
                                response_content = "Please upload a database file first."
                                response_placeholder.markdown(response_content)

                        else:  # Chat RAG
                            web_results = ""
                            academic_results = ""

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
                                    extraction_llm = get_llm_for_chain(model_name, None)
                                    academic_results = run_academic_search(
                                        prompt, rag_context=rag_context, llm=extraction_llm
                                    )

                                history_parts = build_conversation_history(st.session_state.messages)
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
                                context_parts = []

                                history_parts = build_conversation_history(st.session_state.messages)
                                if history_parts:
                                    context_parts.append(
                                        f"=== CONVERSATION HISTORY ===\n{chr(10).join(history_parts)}\n=== END CONVERSATION HISTORY ==="
                                    )

                                if academic_results:
                                    context_parts.append(
                                        f"=== ACADEMIC RESEARCH RESULTS ===\n{academic_results}\n=== END ACADEMIC RESULTS ==="
                                    )

                                if web_results:
                                    context_parts.append(
                                        f"=== WEB SEARCH RESULTS (evaluate for relevance) ===\n"
                                        f"{web_results}\n=== END WEB SEARCH RESULTS ===\n\n"
                                        "IMPORTANT: Critically evaluate web results - ignore irrelevant ones."
                                    )

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
                                thinking_placeholder.empty()
                                response_placeholder.markdown(response_content)

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
                        logger.error(f"Error processing request: {e}")
                        st.error(f"Error: {sanitize_error_message(e)}")
                else:
                    thinking_placeholder.empty()
                    stop_button_placeholder.empty()
                    st.error("Failed to load the model. Please check if Ollama is running and the model is installed.")


if __name__ == "__main__":
    cleanup_expired_sessions()
    main()
