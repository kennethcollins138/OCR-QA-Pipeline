# Use NVIDIA CUDA image with cuDNN for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common git wget poppler-utils python3.10 python3-pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container
COPY . /app

# List files for debugging
RUN ls -al /app

# Expose the port Flask will run on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=nougat_ocr.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
