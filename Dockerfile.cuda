# CUDA + Python base image (runtime only is enough)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first (layer caching)
COPY requirements.txt .

# Install Python deps (CPU+GPU agnostic ones)
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CUDA-enabled PyTorch explicitly (example versions; adjust if needed)
# If torch is already in requirements.txt, this will override it.
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.3.1+cu121" \
    "torchvision==0.18.1+cu121" \
    "torchaudio==2.3.1+cu121"

# Now copy the rest of the app
COPY . .

# Expose the FastAPI port
EXPOSE 8888

# Start the server
CMD ["python3", "main.py", "--host", "0.0.0.0"]