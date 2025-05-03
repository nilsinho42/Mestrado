FROM python:3.11

WORKDIR /app


# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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