# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development commands

### Local (Streamlit) workflow

The app is a single Streamlit entrypoint in `omniscience_pro.py` and talks to a local Ollama instance.

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running in another terminal
ollama serve

# Start the UI
streamlit run omniscience_pro.py
```

By default the app connects to Ollama at `http://127.0.0.1:11434`. You can override this with:

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"  # or another host
```

Recommended models (must be pulled into Ollama outside this repo): `qwen3:4b`, `qwen2.5-coder:7b`, `llava`.

### Docker workflows

This repo includes three docker-compose setups that all mount the same persistent directories:

- `./db_omniscience` → ChromaDB vector store
- `./chats` → chat history JSON files
- `./uploads` → uploaded document and database files

Common flows:

```bash
# Auto-detect GPU and start appropriate stack (Ollama + app)
./docker-start.sh

# Full stack with GPU (uses docker-compose.yml)
sudo docker compose up -d

# Full stack CPU-only (uses docker-compose.cpu.yml)
sudo docker compose -f docker-compose.cpu.yml up -d

# App only, using an existing local Ollama (docker-compose.local-ollama.yml)
sudo docker compose -f docker-compose.local-ollama.yml up -d

# Stop stack
sudo docker compose down

# Tail logs
sudo docker compose logs -f
```

In compose-based flows the app is reachable at `http://localhost:8501` and Ollama at `http://localhost:11434`.

### Tests and linting

There is currently no dedicated test suite, lint configuration, or automation in this repo (no `tests/`, `pytest.ini`, `tox.ini`, or lint config files). If you add tests or tooling, document the commands here and prefer reusing the existing virtualenv / Docker flows.

## High-level architecture

### Overall

- Single Python module: `omniscience_pro.py` implements configuration, security, retrieval, external tools, and the entire Streamlit UI.
- Persistence is file-based only: vector store via Chroma on disk, chat history as JSON files, and uploads saved to a local directory.
- Three user-facing modes are handled in one Streamlit app:
  - **Chat (RAG)** – local file/code/document retrieval plus optional web and academic search.
  - **Vision (Images)** – image upload and multimodal inference via Ollama (e.g., `llava`).
  - **Database (SQL)** – natural-language querying over uploaded SQLite databases.

### Configuration and security layer

- Environment-driven configuration at the top of `omniscience_pro.py` controls directories and limits:
  - `OMNISCIENCE_DB_DIR` (default `./db_omniscience`) – ChromaDB storage path.
  - `OMNISCIENCE_CHATS_DIR` (default `./chats`) – chat session JSONs.
  - `OMNISCIENCE_UPLOAD_DIR` (default `./uploads`) – uploaded files and databases.
  - `OMNISCIENCE_MAX_FILE_SIZE_MB`, `OMNISCIENCE_MAX_FILES_PER_SCAN`, `OMNISCIENCE_MAX_MESSAGES`, session expiry/idle limits, and a simple in-memory rate limiter.
- Security helpers (`sanitize_session_id`, `sanitize_filename`, `validate_path_within_directory`, `sanitize_error_message`, `check_rate_limit`) are used throughout to enforce:
  - No path traversal or symlink escape when reading/writing sessions or uploads.
  - Restricted permissions (`0o700` for dirs, `0o600` for files) where possible.
  - Size and count limits on scanned or uploaded content.
  - Redaction of sensitive paths and SQL fragments from error messages.

### Session and history management

- Chat sessions are stored as JSON in `CHATS_DIR` with filenames like `chat_YYYYMMDD_HHMMSS_xxxx.json`.
- Each session includes a title (derived from the first user message), timestamp, and a bounded list of messages.
- Messages can contain an `image` field; binary data is converted to base64 on save and decoded on load.
- File locking via `fcntl` is used on load/save to avoid concurrent write corruption.
- `cleanup_expired_sessions` (not wired into a scheduler) implements session expiry and a maximum number of sessions; if you add background maintenance, call this instead of reimplementing deletion.

### File ingestion and vector store (RAG pipeline)

- In **Chat (RAG)** mode, context comes from:
  - Uploaded files processed via `process_uploaded_files` → text extraction (`read_file_content`), chunking (`RecursiveCharacterTextSplitter` with language-specific configuration for common code/text types), and conversion into LangChain `Document` objects.
  - On-disk project directories scanned via `scan_directory`, which:
    - Applies path validation and rejects obviously sensitive root-level system directories unless they are within the current working directory.
    - Skips `IGNORED_DIRS`, `IGNORED_FILES`, and various large/binary extensions.
- `initialize_vectorstore` wraps a persistent Chroma collection under `DB_DIRECTORY` and is invoked in three situations:
  - When first scanning a folder.
  - When processing uploaded files.
  - On app startup if a DB directory already exists (to rehydrate the vector store).
- `ingest_documents` batches adds to Chroma and surfaces chunk counts; `get_all_filenames` and `delete_file_from_db` provide per-file management in the UI.

### External tools and integrations

- **LLM backends** (Ollama):
  - `load_llm` and `get_llm_for_chain` centralize Ollama model configuration and health checks (simple `.invoke("test")` on load).
  - UI exposes different model choices depending on mode (general chat vs. vision-capable models).
- **Web search** (optional):
  - `run_web_search` uses `duckduckgo-search` via LangChain community wrappers when installed and is gated by a sidebar toggle.
  - Results are formatted into a markdown block that is appended to the unified prompt when enabled.
- **Academic research** (optional):
  - `run_academic_search` combines Semantic Scholar, arXiv, and OpenAlex.
  - If a vector store is present, `extract_academic_query` uses the LLM and retrieved RAG context to synthesize a compact, paper-style search query instead of passing the raw user prompt.
  - Results from the three providers are normalized into a common structure, scored by relevance/citations/recency, deduplicated, and injected into the LLM prompt as structured markdown.
- **SQLite querying**:
  - In **Database (SQL)** mode, users upload a SQLite file which is saved under `UPLOAD_DIR` and referenced via `st.session_state.db_path`.
  - `query_sqlite_db` generates a single `SELECT` query using the LLM against a lightweight schema summary and enforces multiple safeguards:
    - Read-only connection (`mode=ro`, `PRAGMA query_only = ON`).
    - Hard block on non-`SELECT` beginnings and a large set of dangerous keywords (`INSERT`, `UPDATE`, `DROP`, `PRAGMA`, `ATTACH`, etc.).
    - Single-statement enforcement and row/result limits.
  - The executed SQL and a truncated result set are returned together as markdown.

### Streamlit UI and interaction model

- `main()` is the Streamlit entrypoint guarded by `if __name__ == "__main__":` and is responsible for:
  - Page config and global theming (dark/purple CSS injected via `PURPLE_THEME_CSS`).
  - Initializing `st.session_state` for the current session ID, messages, and vector store.
  - Rendering the sidebar:
    - Mode radio: `Chat (RAG)`, `Vision (Images)`, `Database (SQL)`.
    - Global toggles for **Web Search** and **Academic Research**, which influence how prompts are built but not which mode is active.
    - Chat export to Markdown using the in-memory message list.
    - Session list, creation, selection, and deletion, all backed by the JSON files in `CHATS_DIR`.
    - In **Chat (RAG)** mode, folder scan controls, upload-based ingestion, and vector store management (per-file delete/purge DB).
    - In **Database (SQL)** mode, SQLite upload, validation, and storage in `UPLOAD_DIR`.
  - Rendering the main chat area:
    - Iterates over `st.session_state.messages`, displaying user/assistant messages and any inline images or source lists.
    - Per-assistant-message “Copy Response” helper.
    - Separate handling for **Vision (Images)**: file uploader + prompt, with responses generated via `process_vision_request` against Ollama’s HTTP API.
    - Shared chat input box for non-vision modes.

### Prompting strategy and streaming

- A custom `StreamHandler` (subclass of `BaseCallbackHandler`) drives token-level streaming into a placeholder, replacing a “Thinking…” box (`THINKING_HTML`) once the first token arrives.
- In **Chat (RAG)** mode with a vector store:
  - Web search and academic results are fetched *before* retrieval so they can be included alongside local context.
  - A small set of top documents is retrieved via a Chroma retriever and their contents concatenated into a `rag_context` string; file paths are extracted into a `sources` list for optional UI display.
  - A single **unified prompt** is constructed that includes:
    - Recent conversation history (last N messages, truncated).
    - Local RAG context.
    - Web search and academic result blocks when enabled.
    - A detailed set of instructions about relevance checking, source prioritization, and non-hallucination.
  - The unified prompt is passed directly to an Ollama LLM instance with streaming callbacks; code/document sources are only shown if the model appears to rely on local context rather than purely external search.
- When no vector store is available, a simpler augmented prompt includes conversation history and any web/academic blocks, again with instructions on source evaluation.

### Notes for future changes

- A previous “code interpreter” / arbitrary Python execution feature has been deliberately removed for security; do **not** reintroduce arbitrary code execution inside the Streamlit process. If needed, use a separate, sandboxed service with clear boundaries.
- Any new feature that reads/writes files or touches user-controlled paths should reuse the existing sanitization (`sanitize_filename`, `validate_path_within_directory`) and permission patterns.
- If you split `omniscience_pro.py` into multiple modules in the future, keep this WARP file updated to reflect where configuration, security utilities, vector store logic, and UI code live.
