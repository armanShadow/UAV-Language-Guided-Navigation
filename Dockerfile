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
COPY AnsweringAgent/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the codebase
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Create directory for checkpoints
RUN mkdir -p AnsweringAgent/checkpoints

# Set the default command
CMD ["python3", "AnsweringAgent/src/train.py"] 