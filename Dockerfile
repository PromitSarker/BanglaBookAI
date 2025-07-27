FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential gcc g++ \
        tesseract-ocr tesseract-ocr-ben \
        libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlink python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code + start script
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Create persistent dir
RUN mkdir -p /app/chroma_db

EXPOSE 8000 8501
CMD ["./start.sh"]