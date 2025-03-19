from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
import time
import psutil
import os

class MetricsCollector:
    def __init__(self, service_name: str):
        self.registry = CollectorRegistry()
        self.service_name = service_name
        
        # Video processing metrics
        self.processing_duration = Histogram(
            'video_processing_duration_seconds',
            'Time spent processing videos',
            ['service', 'status'],
            registry=self.registry
        )
        
        self.processed_videos = Counter(
            'processed_videos_total',
            'Number of videos processed',
            ['service', 'status'],
            registry=self.registry
        )
        
        self.detection_count = Counter(
            'detections_total',
            'Number of detections made',
            ['service', 'type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Current memory usage',
            ['service'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'Current CPU usage',
            ['service'],
            registry=self.registry
        )
        
        # Model metrics
        self.inference_time = Histogram(
            'model_inference_time_seconds',
            'Time spent on model inference',
            ['service', 'model'],
            registry=self.registry
        )
        
        self.model_memory = Gauge(
            'model_memory_bytes',
            'Memory used by the model',
            ['service', 'model'],
            registry=self.registry
        )
        
        # API metrics
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['service', 'endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests',
            ['service', 'endpoint'],
            registry=self.registry
        )
        
        # Start system metrics collection
        self._start_system_metrics()

    def _start_system_metrics(self):
        """Update system metrics periodically."""
        process = psutil.Process()
        
        def update_metrics():
            self.memory_usage.labels(service=self.service_name).set(
                process.memory_info().rss
            )
            self.cpu_usage.labels(service=self.service_name).set(
                process.cpu_percent()
            )
        
        # Initial update
        update_metrics()

    def record_processing_duration(self, duration: float, status: str):
        """Record video processing duration."""
        self.processing_duration.labels(
            service=self.service_name,
            status=status
        ).observe(duration)
        
        self.processed_videos.labels(
            service=self.service_name,
            status=status
        ).inc()

    def record_detection(self, detection_type: str):
        """Record a detection."""
        self.detection_count.labels(
            service=self.service_name,
            type=detection_type
        ).inc()

    def record_inference_time(self, duration: float, model_name: str):
        """Record model inference time."""
        self.inference_time.labels(
            service=self.service_name,
            model=model_name
        ).observe(duration)

    def record_request_duration(self, duration: float, endpoint: str, method: str, status: str):
        """Record API request duration."""
        self.request_duration.labels(
            service=self.service_name,
            endpoint=endpoint,
            method=method,
            status=status
        ).observe(duration)

    def track_request(self, endpoint: str):
        """Track active request count."""
        self.active_requests.labels(
            service=self.service_name,
            endpoint=endpoint
        ).inc()
        
        return RequestTracker(
            self.active_requests,
            self.service_name,
            endpoint
        )

    def push_metrics(self):
        """Push metrics to Prometheus Pushgateway."""
        pushgateway = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
        try:
            push_to_gateway(
                pushgateway,
                job=self.service_name,
                registry=self.registry
            )
        except Exception as e:
            print(f"Failed to push metrics: {str(e)}")

class RequestTracker:
    def __init__(self, gauge, service_name: str, endpoint: str):
        self.gauge = gauge
        self.service_name = service_name
        self.endpoint = endpoint
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gauge.labels(
            service=self.service_name,
            endpoint=self.endpoint
        ).dec() 