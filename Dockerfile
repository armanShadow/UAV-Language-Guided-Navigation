# run the container:
# docker run --gpus all --shm-size=8g
# -v $(pwd)/AnsweringAgent/outputs:/app/AnsweringAgent/outputs
# -v $(pwd)/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images:/app/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images:ro
# --rm -it answering-agent-train

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

# Copy only the necessary code files
COPY AnsweringAgent/src/ AnsweringAgent/src/
COPY Aerial-Vision-and-Dialog-Navigation/src/ Aerial-Vision-and-Dialog-Navigation/src/
COPY Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/ Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/
COPY Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/ Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/
COPY Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/et_haa/ Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/et_haa/
COPY Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/lstm_haa/ Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/lstm_haa/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Create directory for train_images
RUN mkdir -p Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images

# Create output directories within AnsweringAgent
RUN mkdir -p AnsweringAgent/outputs/{checkpoints,logs,results}
 