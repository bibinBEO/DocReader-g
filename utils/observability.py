
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor

from prometheus_client import Counter, Histogram, generate_latest

# Prometheus Metrics
DOC_UPLOAD_COUNTER = Counter('docreader_documents_uploaded_total', 'Total number of documents uploaded')
DOC_PROCESSING_TIME = Histogram('docreader_document_processing_seconds', 'Time spent processing documents', buckets=[.1, .5, 1, 3, 5, 10, 30, 60])
DOC_STATUS_COUNTER = Counter('docreader_document_status_total', 'Total number of documents by final status', ['status'])

# OpenTelemetry Tracing
def configure_opentelemetry(app=None, celery_app=None):
    service_name = os.getenv("OTEL_SERVICE_NAME", "docreader-service")
    service_version = os.getenv("OTEL_SERVICE_VERSION", "0.1.0")

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    })

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    if app:
        FastAPIInstrumentor.instrument_app(app)
    if celery_app:
        CeleryInstrumentor().instrument(tracer_provider=provider, app=celery_app)

    print(f"OpenTelemetry configured for service: {service_name} v{service_version}")

