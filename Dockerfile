FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential gcc g++ \
        tesseract-ocr tesseract-ocr-ben \
        libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python/pip point to the right binaries
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for persistent storage
RUN mkdir -p /app/chroma_db

EXPOSE 8000 8501

CMD ["./start.sh"]
