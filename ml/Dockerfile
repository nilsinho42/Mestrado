FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
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
    mkdir -p data/results && chmod 777 data/results

# Try to download the model using the download script
RUN python -c "import os; os.makedirs('data', exist_ok=True)" && \
    python download_models.py || true

# Verify model was downloaded or attempt manual download
RUN if [ ! -f "yolov11n.pt" ]; then \
      echo "Attempting manual download of YOLOv11n model..." && \
      wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt && \
      mv yolo11n.pt yolov11n.pt && \
      echo "Downloaded and renamed model to yolov11n.pt"; \
    fi

# Add startup script with verification for YOLOv11n model
RUN echo '#!/bin/bash\n\
echo "ML Service starting..."\n\
# Verify YOLOv11n model exists\n\
if [ ! -f "yolov11n.pt" ]; then\n\
  echo "ERROR: YOLOv11n model not found!"\n\
  echo "Please download YOLOv11n.pt and place it in the application root directory."\n\
  exit 1\n\
fi\n\
# Start the application\n\
python run_servers.py' > /start.sh && \
    chmod +x /start.sh

# Expose port
EXPOSE 8000

# Start the application
CMD ["/start.sh"] 