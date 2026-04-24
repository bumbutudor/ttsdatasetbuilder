# TTS/STT Dataset Builder - GPU Version
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
# Adăugăm poppler-utils pentru pdf2image și git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    poppler-utils \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements.txt requirements-docker.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r ${REQUIREMENTS_FILE}

# Download Spacy Romanian Model (CRITIC PENTRU TEXT)
RUN python -m spacy download ro_core_news_lg

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/projects

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
