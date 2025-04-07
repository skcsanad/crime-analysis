FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Basic Python setup
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl wget build-essential unzip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# ✅ Install CUDA-compatible PyTorch manually
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Installing gdown for downloading the data
RUN pip install gdown

# Add your other packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter

# Setup workspace
WORKDIR /workspace

# Copy the preprocess script
COPY preprocess.py /workspace

EXPOSE 8888

CMD python preprocess.py && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
