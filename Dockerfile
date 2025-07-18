# run the container:
# docker run --gpus '"device=0,1,2,3,4,5,6,7,8,9"'     --shm-size=8g     -v $(pwd):/app/UAV-Language-Guided-Navigation     -v /export/openhome/vaziri/datasets:/app/datasets  -v /export/openhome/vaziri/datasets/AVDN/train_images:/app/UAV-Language-Guided-Navigation/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images   --rm -it     armanshadow/ualgn:latest

# Use PyTorch 1.11.0 with CUDA 11.3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create project directory
RUN mkdir -p UAV-Language-Guided-Navigation
RUN mkdir -p datasets

