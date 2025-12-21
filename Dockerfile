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

# Copy and make start script executable
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080

# Use start script
CMD ["./start.sh"]
