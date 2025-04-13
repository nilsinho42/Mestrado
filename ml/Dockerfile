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
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads && chmod 777 uploads

# Add startup script
RUN echo '#!/bin/bash\necho "ML Service starting..."\npython run_servers.py' > /start.sh && \
    chmod +x /start.sh

# Expose port
EXPOSE 8000

# Start the application
CMD ["/start.sh"] 