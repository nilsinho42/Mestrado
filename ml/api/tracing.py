from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import os

def setup_tracing(app, service_name):
    """Setup OpenTelemetry tracing."""
    # Configure the tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer_provider()

    # Configure the OTLP exporter
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)

    # Instrument SQLite
    SQLite3Instrumentor().instrument(tracer_provider=tracer)

    # Instrument requests library
    RequestsInstrumentor().instrument(tracer_provider=tracer)

    return trace.get_tracer(service_name)

def create_span(name: str, parent_span=None, attributes=None):
    """Create a new span with the given name and attributes."""
    tracer = trace.get_tracer(__name__)
    context = trace.get_current_span().get_span_context() if parent_span is None else parent_span.get_span_context()
    
    span = tracer.start_span(
        name,
        context=context,
        attributes=attributes or {}
    )
    return span 