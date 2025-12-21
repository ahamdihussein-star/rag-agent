FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ara \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/data /app/uploads /app/doc_images

EXPOSE 8080

# Use shell form to expand $PORT variable
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
