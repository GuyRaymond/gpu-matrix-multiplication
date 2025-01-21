# Use the official CUDA image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy the source code into the container
COPY . .

# Build the project
RUN mkdir build && cd build && \
    cmake .. && \
    make

# Set the entry point to run the program
ENTRYPOINT ["./build/gpu_matrix_multiplication"]
