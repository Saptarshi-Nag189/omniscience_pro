FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install with pip cache mount (BuildKit feature)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY *.py .
COPY prompts.md .

# Create directories for data persistence and drop root privileges.
# uid 1000 matches the default first user on most host systems, so bind-mounted
# volumes (./chats, ./uploads, ./db_omniscience) stay writable.
RUN mkdir -p db_omniscience chats uploads \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "omniscience_pro.py", "--server.address", "0.0.0.0"]
