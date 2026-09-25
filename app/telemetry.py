"""OpenTelemetry wiring — the only module in this project that imports
`opentelemetry.*`. Everything here is gated on `OTEL_EXPORTER_OTLP_ENDPOINT`
being set, so on Vercel (where it's never set, and the opentelemetry packages
aren't even installed since they live in the `dev` dependency group) this
module no-ops before touching any of those imports.
"""

import os
from contextlib import contextmanager


class _NoOpSpan:
    def set_attribute(self, *args, **kwargs):
        pass


class _NoOpTracer:
    @contextmanager
    def start_as_current_span(self, *args, **kwargs):
        yield _NoOpSpan()


def setup_telemetry(app) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": "p3-hr-rag-api"})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    PsycopgInstrumentor().instrument()


def get_tracer():
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return _NoOpTracer()

    from opentelemetry import trace

    return trace.get_tracer("p3")
