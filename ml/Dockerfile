FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies in proper order to avoid build issues
RUN pip install --no-cache-dir numpy==1.26.2 scipy==1.11.3 && \
    pip install --no-cache-dir lap==0.5.12 && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads && chmod 777 uploads && \
    mkdir -p data/results && chmod 777 data/results && \
    mkdir -p credentials && chmod 755 credentials

# Copy configuration files first 
COPY cost_config_2.ini /app/

# Copy necessary Python modules
COPY api.py /app/
COPY main.py /app/
COPY core/ /app/core/
COPY config.yaml /app/
COPY download_models.py /app/
COPY run_servers.py /app/
# Download the YOLO model during build
RUN wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt || \
    python -c "from ultralytics import YOLO; model = YOLO('yolo11n')"

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-c", "import sys; sys.path.append('/app'); import api; import uvicorn; uvicorn.run(api.app, host='0.0.0.0', port=8000, log_level='info')"] 