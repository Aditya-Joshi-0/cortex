FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy and NLTK models
RUN python -m nltk.downloader punkt && \
    python -m nltk.downloader punkt_tab

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/documents

# Expose ports
EXPOSE 7860

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 7860"]