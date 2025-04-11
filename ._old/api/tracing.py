from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import os
import logging

logger = logging.getLogger(__name__)

def setup_tracing(app, service_name):
    """
    Setup OpenTelemetry tracing with automatic instrumentation.
    
    This configures automatic tracing for:
    - FastAPI routes
    - SQLite database calls
    - HTTP requests
    
    Without a collector, traces will be generated but not exported,
    minimizing overhead while keeping the instrumentation in place.
    """
    # Configure the tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer_provider()

    # Set up a console exporter for debugging - logs to console only in debug mode
    if os.getenv("DEBUG", "false").lower() == "true":
        console_exporter = ConsoleSpanExporter()
        tracer.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("Enabled console span exporter for tracing")

    # Configure the OTLP exporter (only if endpoint is explicitly provided)
    otlp_endpoint = os.getenv("OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            logger.info(f"Attempting to connect to OpenTelemetry collector at {otlp_endpoint}")
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            tracer.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("Successfully configured OpenTelemetry collector")
        except Exception as e:
            logger.warning(f"Failed to configure OpenTelemetry collector: {str(e)}")
            logger.warning("Tracing will be enabled but traces won't be exported")

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)
    logger.info("FastAPI instrumentation enabled")

    # Instrument SQLite
    SQLite3Instrumentor().instrument(tracer_provider=tracer)
    logger.info("SQLite instrumentation enabled")

    # Instrument requests library
    RequestsInstrumentor().instrument(tracer_provider=tracer)
    logger.info("HTTP client instrumentation enabled")

    return trace.get_tracer(service_name)

# Note: The create_span function has been removed as it was not being used 