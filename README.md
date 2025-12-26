# Omniscience Pro

A local RAG (Retrieval-Augmented Generation) system with web search, academic research integration, and multi-session chat management. Runs entirely offline using Ollama.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- 🔍 **Local RAG**: Query your codebase and documents using natural language
- 🌐 **Web Search**: Augment responses with DuckDuckGo search results
- 📚 **Academic Research**: Search Semantic Scholar, arXiv, and OpenAlex
- 💾 **Session Management**: Persistent chat history with auto-titles
- 🖼️ **Vision Mode**: Analyze images with LLaVA
- 🗃️ **SQL Mode**: Query SQLite databases with natural language
- 🔒 **Security Hardened**: Input validation, path traversal protection, SQL injection prevention

### Supported Models

The following models are supported out-of-the-box *(requires download via Ollama)*:

| Model | Type | Size | Use Case |
| ----- | ---- | ---- | -------- |
| `qwen3:4b` | Chat | 4B | Recommended general purpose |
| `qwen2.5-coder:7b` | Chat | 7B | Code-focused |
| `qwen2.5-coder:1.5b` | Chat | 1.5B | Lightweight, fast |
| `llama3.2:3b` | Chat | 3B | General purpose |
| `mistral:7b` | Chat | 7B | Alternative general |
| `llava:7b` | Vision | 7B | Image analysis |
| `llama3.2-vision` | Vision | 11B | Advanced image analysis |

## Prerequisites

### 1. Install Ollama

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**

```bash
brew install ollama
```

**Windows:**

Download from: <https://ollama.com/download>

### 2. Download LLM Models

```bash
# Start Ollama service
ollama serve

# Download models (in a new terminal)
# Chat models (pick at least one):
ollama pull qwen3:4b            # Recommended: Good balance of speed/quality
ollama pull qwen2.5-coder:7b    # Code-focused (7B params)
ollama pull qwen2.5-coder:1.5b  # Lightweight, fast
ollama pull llama3.2:3b         # General purpose (3B params)
ollama pull mistral:7b          # Alternative general purpose (7B params)

# Vision model (for image analysis):
ollama pull llava:7b            # Required for Vision mode
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, run the app
streamlit run omniscience_pro.py
```

Open <http://localhost:8501> in your browser.

## Usage

### Chat with Your Code

1. Enter a folder path in "DATA SOURCE"
2. Click "SCAN" to index the files
3. Ask questions about your code

### Web Search

Toggle "Augment with Web Search" for current events or general knowledge questions.

### Academic Research

Toggle "Academic Research" to search academic papers from:

- Semantic Scholar
- arXiv
- OpenAlex

### Vision Mode

1. Select "Vision (Images)" mode
2. Upload an image
3. Ask questions about the image

### SQL Mode

1. Select "Database (SQL)" mode
2. Upload a SQLite database file
3. Ask natural language questions about your data

## Configuration

Environment variables (optional):

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"  # Ollama server URL
export OMNISCIENCE_DB_DIR="./db_omniscience"     # Vector DB location
export OMNISCIENCE_CHATS_DIR="./chats"           # Chat history location
```

## Deployment Options

| Method | Pros | Cons |
| ------ | ---- | ---- |
| **Local** | Full control, private data | Manual setup, single machine |
| **Docker** | Portable, GPU support, easy server deploy | Needs Docker installed |
| **Streamlit Cloud** | Free hosting, easy deploy | Public URL, need external Ollama |

### Docker Deployment

The script **auto-detects GPU** and uses the appropriate configuration.

**Prerequisites:**

- Docker & Docker Compose
- (Optional) NVIDIA Container Toolkit for GPU: [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Quick Start:**

```bash
./docker-start.sh
```

This will:

1. Detect if NVIDIA GPU is available
2. Use `docker-compose.yml` (GPU) or `docker-compose.cpu.yml` (CPU)
3. Start Ollama + download models automatically
4. Start Omniscience Pro at <http://localhost:8501>

**Docker Options:**

| File | Use Case |
| ---- | -------- |
| `docker-compose.yml` | Full stack with NVIDIA GPU |
| `docker-compose.cpu.yml` | Full stack, CPU only |
| `docker-compose.local-ollama.yml` | App only, use existing local Ollama |

**Manual Start:**

```bash
# Full stack with GPU
sudo docker compose up -d

# Full stack without GPU (CPU-only)
sudo docker compose -f docker-compose.cpu.yml up -d

# App only (requires local Ollama running)
sudo docker compose -f docker-compose.local-ollama.yml up -d

# Stop
sudo docker compose down
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Ollama](https://ollama.com) - Local LLM inference
- [LangChain](https://langchain.com) - LLM framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Streamlit](https://streamlit.io) - Web interface
- [Sentence Transformers](https://www.sbert.net) - Embeddings
- [DuckDuckGo](https://duckduckgo.com) - Web search API
- [Semantic Scholar](https://www.semanticscholar.org) - Academic search
- [arXiv](https://arxiv.org) - Research papers
