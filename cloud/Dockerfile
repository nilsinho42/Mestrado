FROM python:3.11

WORKDIR /app

# Install minimal runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Cython first as it's needed for building some packages
RUN pip install --no-cache-dir Cython

# Install critical dependencies first
RUN pip install --no-cache-dir torch>=2.7.0 torchvision>=0.22.0 gdown>=4.7.1 tensorboard>=2.12.0

# Now install all requirements except torchreid (we'll install from source)
RUN grep -v torchreid requirements.txt > requirements_no_torchreid.txt && \
    pip install --no-cache-dir -r requirements_no_torchreid.txt

# Install torchreid from source
RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git && \
    cd deep-person-reid && \
    pip install -e . && \
    cd .. && \
    # Verify torchreid module structure
    python -c "import torchreid; print(dir(torchreid)); print('torchreid utils:', dir(torchreid.utils) if hasattr(torchreid, 'utils') else 'NOT FOUND')"

# Verify DeepSort can use torchreid
RUN python -c "from deep_sort_realtime.deepsort_tracker import DeepSort; \
        tracker = DeepSort(embedder='torchreid', embedder_gpu=False, half=False, bgr=True);"

# Copy application files
COPY tracker_app.py .
COPY yolo11n.pt .

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "tracker_app:app", "--host", "0.0.0.0", "--port", "8080"] 