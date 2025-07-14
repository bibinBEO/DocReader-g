
# Stage 1: Builder - for dependencies and potentially model downloads
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ImageMagick for pdf2image
RUN apt-get update && apt-get install -y imagemagick && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/^\(disable ghostscript types\)/%\1/' /etc/ImageMagick-6/policy.xml

# Set working directory
WORKDIR /app

# Copy only necessary files for dependency installation
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production - lean image for deployment
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS production

ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
