# WARP.md

This file provides guidance to WARP (warp.dev) and human developers working with code in this repository. It covers how to run the app, how the codebase is structured, and how to safely extend it.

## Quick overview

- Single Streamlit app entrypoint: `omniscience_pro.py`.
- Local-first LLM setup using Ollama (HTTP API on `http://127.0.0.1:11434` by default).
- File-based persistence only (no external managed services by default).
- Three main user modes in the UI:
  - **Chat (RAG)** – question answering over local files / code / documents + optional web and academic search.
  - **Vision (Images)** – image upload and multimodal chat via vision-capable Ollama models (e.g. `llava`).
  - **Database (SQL)** – natural-language querying over uploaded SQLite databases, with strong safety restrictions.

If you are new to the project, start with the **Local development** section, then skim **High-level architecture** to understand how the pieces fit together.

## Development commands

### Local (Streamlit) workflow

The app is a single Streamlit entrypoint in `omniscience_pro.py` and talks to a local Ollama instance.

#### Prerequisites

- Python 3.10+ (3.11 recommended).
- [Ollama](https://ollama.com) installed and able to run models locally.
- Recommended Ollama models (must be pulled outside this repo, for example via `ollama pull <model>`):
  - `qwen3:4b` – general chat / code.
  - `qwen2.5-coder:7b` – code-heavy tasks.
  - `llava` – image/vision tasks.

#### Create a virtual environment and install dependencies

This project assumes you are using a Python virtualenv (do **not** install system-wide packages unless you know what you are doing).

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Start Ollama

```bash
# In a separate terminal (or systemd/service of your choice)
ollama serve
```

By default the app connects to Ollama at `http://127.0.0.1:11434`. You can override this with:

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"  # or another host/port
```

#### Run the Streamlit app

```bash
# From the repo root, with the venv activated
streamlit run omniscience_pro.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) where you can access the UI.

### Docker workflows

This repo includes three docker-compose setups that all mount the same persistent directories:

- `./db_omniscience` → ChromaDB vector store.
- `./chats` → chat history JSON files.
- `./uploads` → uploaded document and database files.

All compose setups use these host directories so that data persists across container restarts.

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

In compose-based flows the app is reachable at `http://localhost:8501` and Ollama at `http://localhost:11434` (unless you change ports in the compose files).

### Tests and linting

There is currently no dedicated test suite, lint configuration, or automation in this repo (no `tests/`, `pytest.ini`, `tox.ini`, or lint config files).

If you add tests or tooling:

- Prefer reusing the existing virtualenv / Docker flows described above.
- Document new commands in this section (e.g., `pytest`, `ruff`, `mypy`) so other developers and WARP know how to run them.

## Configuration and environment

Most configuration is done via environment variables defined and read near the top of `omniscience_pro.py`.

Key paths (all default to directories under the repo root):

- `OMNISCIENCE_DB_DIR` (default `./db_omniscience`) – ChromaDB storage path.
- `OMNISCIENCE_CHATS_DIR` (default `./chats`) – chat session JSON files.
- `OMNISCIENCE_UPLOAD_DIR` (default `./uploads`) – uploaded files and SQLite databases.

Operational limits and safety-related settings:

- `OMNISCIENCE_MAX_FILE_SIZE_MB` – max size of a single file to ingest.
- `OMNISCIENCE_MAX_FILES_PER_SCAN` – cap on how many files a single scan can process.
- `OMNISCIENCE_MAX_MESSAGES` – limit on messages kept per chat session.
- Session expiry and idle timeouts – used by `cleanup_expired_sessions` to prune old sessions.
- A simple in-memory rate limiter for per-client requests.

Ollama configuration:

- `OLLAMA_BASE_URL` – base URL for the Ollama HTTP API.
- Model names are currently hard-coded or chosen via UI controls; if you change them, ensure that your local Ollama instance has those models pulled.

Security-related note: any new configuration that impacts file paths or external network calls should follow the existing patterns for sanitization and validation (see the next section).

## High-level architecture

### Overall structure

- Single Python module: `omniscience_pro.py` implements configuration, security, retrieval/RAG, external tools, and the entire Streamlit UI.
- Persistence is file-based only: vector store via Chroma on disk, chat history as JSON files, and uploads saved to a local directory.
- Three user-facing modes are handled in one Streamlit app:
  - **Chat (RAG)** – local file/code/document retrieval plus optional web and academic search.
  - **Vision (Images)** – image upload and multimodal inference via Ollama (e.g., `llava`).
  - **Database (SQL)** – natural-language querying over uploaded SQLite databases.

At a very high level:

1. The user interacts with the Streamlit UI.
2. Each user action (sending a message, uploading a file, scanning a folder, etc.) updates `st.session_state`.
3. Depending on the mode, the app builds an appropriate prompt, optionally retrieves local context (RAG) and/or web/academic results, and sends everything to an Ollama-backed LLM.
4. Responses are streamed back to the UI and persisted into the current chat session.

### Configuration and security layer

Environment-driven configuration at the top of `omniscience_pro.py` controls directories and limits (see **Configuration and environment** above). On top of that, security helpers are used throughout to avoid common pitfalls:

- `sanitize_session_id` – ensures session IDs are simple, safe strings.
- `sanitize_filename` – strips/normalizes filenames to prevent path traversal.
- `validate_path_within_directory` – hard check that a path is inside an allowed base directory (defense against `..`, symlinks, etc.).
- `sanitize_error_message` – removes sensitive paths and SQL fragments before surfacing them to the UI.
- `check_rate_limit` – simple in-memory rate limiting.

Patterns to follow:

- Never build file paths from user input without running them through `sanitize_filename` and `validate_path_within_directory`.
- When creating directories and files, use the existing helper(s) that set permissions to `0o700` for directories and `0o600` for files.
- When adding new error messages that may contain paths, SQL, or external data, consider routing them through `sanitize_error_message` before logging or displaying.

### Session and history management

Chat sessions are stored as JSON in `CHATS_DIR` with filenames like `chat_YYYYMMDD_HHMMSS_xxxx.json`.

Each session contains:

- A generated title (usually derived from the first user message).
- Timestamps and basic metadata.
- A bounded list of messages (limited by `OMNISCIENCE_MAX_MESSAGES`).
- Optional image data stored as base64 when present in messages.

Implementation details:

- Binary image data is converted to base64 when saving and decoded back on load.
- File locking via `fcntl` is used to avoid concurrent write corruption when multiple requests try to update the same session.
- `cleanup_expired_sessions` implements session expiry and a maximum number of sessions; it is not wired to a scheduler by default. If you add background maintenance (cron, separate worker, etc.), call this function instead of reimplementing deletion.

### File ingestion and vector store (RAG pipeline)

In **Chat (RAG)** mode, local context can come from two main sources:

1. **Uploaded files** – handled via `process_uploaded_files`:
   - Extracts text using `read_file_content` (with format-specific handling where needed).
   - Splits text into chunks using `RecursiveCharacterTextSplitter`, with language-specific settings tuned for common code/text types.
   - Wraps each chunk into a LangChain `Document` object with metadata (file path, etc.).

2. **On-disk project directories** – handled via `scan_directory`:
   - Validates the requested path and rejects obviously sensitive root-level system directories unless they are within the current working directory.
   - Skips `IGNORED_DIRS`, `IGNORED_FILES`, and various large/binary extensions.

Vector store management:

- `initialize_vectorstore` wraps a persistent Chroma collection stored under `DB_DIRECTORY` (usually `./db_omniscience`). It is called:
  - When first scanning a folder.
  - When processing uploaded files.
  - On app startup if a DB directory already exists (rehydrating the vector store).
- `ingest_documents` batches adds to Chroma and returns human-friendly counts for display in the UI.
- `get_all_filenames` and `delete_file_from_db` support per-file management, enabling users to see and remove specific files from the vector store.

When extending the RAG pipeline (e.g., new file types or metadata):

- Add new extraction branches in `read_file_content` or similar helpers.
- Ensure new metadata fields remain JSON-serializable and reasonably small.
- Keep an eye on chunk size and count to avoid blowing up prompts.

### External tools and integrations

#### LLM backends (Ollama)

- `load_llm` and `get_llm_for_chain` centralize Ollama model configuration and a basic health check (simple `.invoke("test")` on load).
- Different model choices are exposed in the UI depending on mode:
  - General chat models for Chat/RAG.
  - Vision-capable models for Vision/Image mode.

If you change model names or add new ones:

- Update the model selection logic in `omniscience_pro.py`.
- Ensure your local Ollama environment has those models pulled (`ollama pull <model>`).

#### Web search (optional)

- Implemented via `run_web_search` using `duckduckgo-search` through LangChain community wrappers (when installed).
- Controlled via a toggle in the sidebar.
- Results are formatted into a markdown block and appended to the unified prompt before sending to the LLM.

#### Academic research (optional)

- Implemented via `run_academic_search`, which queries Semantic Scholar, arXiv, and OpenAlex.
- When a vector store is present, `extract_academic_query` uses the LLM plus retrieved RAG context to synthesize a compact query rather than using the raw user question.
- Results from multiple providers are normalized into a shared structure, scored (relevance, citations, recency), deduplicated, and injected into the LLM prompt as a structured markdown block.

#### SQLite querying

In **Database (SQL)** mode:

- Users upload a SQLite file that is stored under `UPLOAD_DIR` and referenced via `st.session_state.db_path`.
- `query_sqlite_db` generates a single `SELECT` query via the LLM using a lightweight schema summary.
- Strong safety checks are enforced:
  - Connection in read-only mode (`mode=ro`, `PRAGMA query_only = ON`).
  - Hard block on non-`SELECT` statements and a wide set of dangerous keywords (`INSERT`, `UPDATE`, `DROP`, `PRAGMA`, `ATTACH`, etc.).
  - Single-statement enforcement (no query batching).
  - Row and result-size limits.
- The actual SQL and a truncated result set are returned together as markdown and displayed in the UI.

When modifying this flow, preserve the read-only guarantees and statement checks. Do not allow arbitrary SQL execution.

### Streamlit UI and interaction model

`main()` is the Streamlit entrypoint (guarded by `if __name__ == "__main__":`) and is responsible for wiring everything together:

- Page config and global theming (dark/purple CSS injected via `PURPLE_THEME_CSS`).
- Initialization of `st.session_state` for the current session ID, messages, and vector store.

Sidebar behavior:

- Mode radio: `Chat (RAG)`, `Vision (Images)`, `Database (SQL)`.
- Toggles for **Web Search** and **Academic Research** that control how prompts are built.
- Chat export to Markdown using the in-memory message list.
- Session list, creation, selection, and deletion, all backed by the JSON files in `CHATS_DIR`.
- In **Chat (RAG)** mode:
  - Folder scan controls for local directories.
  - Upload-based ingestion for ad-hoc documents.
  - Vector store management (per-file delete, purge DB).
- In **Database (SQL)** mode:
  - SQLite file upload, validation, and storage under `UPLOAD_DIR`.

Main chat area:

- Iterates over `st.session_state.messages`, displaying user and assistant messages plus inline images and optional source lists.
- Provides a per-assistant-message “Copy Response” helper.
- In **Vision (Images)** mode:
  - Uses a file uploader + prompt pattern.
  - Calls `process_vision_request` to post combined image + text prompts to a vision-capable Ollama model.
- A shared chat input box is used for non-vision modes.

### Prompting strategy and streaming

- A custom `StreamHandler` (subclass of `BaseCallbackHandler`) drives token-level streaming into a placeholder element in the UI.
- While the LLM is thinking, a “Thinking…” box (`THINKING_HTML`) is shown, which is replaced as tokens stream in.

In **Chat (RAG)** mode with a vector store:

- Web search and academic results (if enabled) are fetched *before* retrieval so they can be included alongside local context.
- A small set of top documents is retrieved from Chroma via a retriever and concatenated into a `rag_context` string.
- File paths of retrieved chunks are collected into a `sources` list for optional display.
- A single **unified prompt** is constructed that includes:
  - Recent conversation history (last N messages, truncated for length).
  - Local RAG context (from the vector store).
  - Web search and academic result blocks when enabled.
  - A detailed system-style instruction block about relevance checking, source prioritization, and avoiding hallucinations.
- The unified prompt is passed directly to an Ollama LLM instance with streaming callbacks. The UI only highlights local-code/document sources when the answer appears to rely on them instead of purely external search.

When no vector store is available:

- A simpler augmented prompt is used that includes conversation history and any enabled web/academic result blocks.
- The same non-hallucination/relevance instructions are still applied.

## Extending the project

When adding new features, keep the following in mind:

- **Do not** reintroduce arbitrary code execution inside the Streamlit process (a previous “code interpreter” feature was removed for security reasons).
- Reuse the existing security helpers (`sanitize_filename`, `validate_path_within_directory`, `sanitize_error_message`, etc.) whenever dealing with user-controlled file paths or content.
- For new modes or major UI sections, follow the existing Streamlit patterns in `main()` (mode selection via sidebar, state kept in `st.session_state`).
- For new retrieval sources (e.g., another database type, an API-backed document store):
  - Consider adapting them into the existing RAG pipeline (produce `Document` objects and feed them into Chroma) instead of bypassing it.
  - Make sure any new background network calls have timeouts and sensible error handling.

## Troubleshooting

A few common issues and how to diagnose them:

- **UI loads but every call fails with an Ollama error**:
  - Check that `ollama serve` is running and reachable on `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`).
  - Verify that the requested model is pulled in Ollama (`ollama list`).

- **Vector store errors or missing context**:
  - Ensure the `db_omniscience` directory is writable by your user or the container.
  - Try clearing the DB directory and re-ingesting documents if the schema has changed drastically.

- **Permission issues on `chats/` or `uploads/`**:
  - Check directory permissions; they should typically be owned by the user running Streamlit and not world-readable.

If you change core behaviors (e.g., where data is stored, how prompts are constructed, or how sessions are managed), please update this WARP file so both WARP and other developers stay in sync with the real behavior of the system.
