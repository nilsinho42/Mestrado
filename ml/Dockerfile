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

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads && chmod 777 uploads && \
    mkdir -p data/results && chmod 777 data/results && \
    mkdir -p credentials && chmod 755 credentials

# Verify model exists after copying
RUN if [ -f "yolo11n.pt" ]; then \
      echo "YOLO model file found after COPY"; \
    else \
      echo "WARNING: YOLO model file not found after COPY"; \
      echo "Creating placeholder file for yolo11n.pt"; \
      echo "This is a placeholder. The real model will be downloaded at runtime." > yolo11n.pt; \
      echo "Placeholder file created"; \
    fi

# Add startup script with verification for YOLOv11n model
RUN echo '#!/bin/bash\n\
echo "ML Service starting..."\n\
echo "Current directory: $(pwd)"\n\
echo "Listing files in current directory:"\n\
ls -la\n\
echo "Checking for model file:"\n\
if [ -f "yolo11n.pt" ]; then\n\
  echo "Found yolo11n.pt at $(pwd)/yolo11n.pt"\n\
  # Check if it is a placeholder file\n\
  if grep -q "placeholder" yolo11n.pt; then\n\
    echo "Detected placeholder file, attempting to download real model..."\n\
    rm yolo11n.pt\n\
    wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt || \
    python -c "from ultralytics import YOLO; model = YOLO(\"yolo11n\")"\n\
    if [ -f "yolo11n.pt" ]; then\n\
      echo "Successfully downloaded real model file."\n\
    else\n\
      echo "ERROR: Failed to download model. Service may not function correctly."\n\
    fi\n\
  fi\n\
else\n\
  echo "ERROR: YOLOv11n model not found!"\n\
  echo "Attempting to download model..."\n\
  wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt || \
  python -c "from ultralytics import YOLO; model = YOLO(\"yolo11n\")"\n\
  if [ ! -f "yolo11n.pt" ]; then\n\
    echo "ERROR: Failed to download model. Service cannot start."\n\
    exit 1\n\
  fi\n\
fi\n\
# Check for Google Cloud credentials\n\
echo "Checking for Google Cloud credentials..."\n\
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then\n\
  echo "Found Google Cloud credentials at $GOOGLE_APPLICATION_CREDENTIALS"\n\
else\n\
  echo "WARNING: Google Cloud credentials not found at $GOOGLE_APPLICATION_CREDENTIALS!"\n\
  echo "GCP Vision API functionality will not work."\n\
fi\n\
# Start the application\n\
echo "Starting application..."\n\
python run_servers.py' > /start.sh && \
    chmod +x /start.sh

# Expose port
EXPOSE 8000

# Start the application
CMD ["/start.sh"] 