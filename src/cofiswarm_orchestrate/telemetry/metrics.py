"""Prometheus metrics registry + /metrics HTTP exposure.

Series (matching the plan):
  agent_requests_total{agent_id,mode,status}
  agent_tokens_total{agent_id,direction}
  agent_latency_seconds{agent_id,phase}
  kv_cache_hits_total{agent_id}
  unified_memory_free_bytes  (M3 unified memory, set by pressure sampler)
  rag_query_seconds, rag_chunks_returned, rag_embed_seconds, rag_db_errors_total

Exposed via prometheus_client.start_http_server() — or use metrics_text() to
embed inside an existing HTTP server (e.g. the coordinator).
"""
from __future__ import annotations

import functools
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

logger = logging.getLogger(__name__)

# Module-level metric handles — single source of truth.
AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Mode invocations grouped by outcome.",
    ["agent_id", "mode", "status"],
)
AGENT_TOKENS = Counter(
    "agent_tokens_total",
    "Token throughput by agent and direction (prompt|completion).",
    ["agent_id", "direction"],
)
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "End-to-end latency per agent + phase.",
    ["agent_id", "phase"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
KV_CACHE_HITS = Counter(
    "kv_cache_hits_total",
    "KV-cache prefix hits per agent.",
    ["agent_id"],
)
UNIFIED_MEMORY_FREE = Gauge(
    "unified_memory_free_bytes",
    "Free unified memory on the host (M3). Updated by pressure sampler.",
)

# RAG metrics (Phase 4 — declared here so all telemetry lives in one place).
RAG_QUERY_SECONDS = Histogram(
    "rag_query_seconds",
    "End-to-end sqlite-vec query latency.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
RAG_CHUNKS_RETURNED = Histogram(
    "rag_chunks_returned",
    "Number of chunks returned per RAG query.",
    buckets=(1, 2, 3, 5, 10, 20),
)
RAG_EMBED_SECONDS = Histogram(
    "rag_embed_seconds",
    "Time to embed a batch of texts.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
RAG_DB_ERRORS = Counter(
    "rag_db_errors_total",
    "sqlite-vec store errors during RAG operations.",
    ["op"],
)


def start_metrics_server(port: int) -> None:
    """Bind a standalone /metrics HTTP server on the given port."""
    try:
        start_http_server(port)
        logger.info("metrics: listening on :%d/metrics", port)
    except OSError as exc:
        logger.error("metrics: failed to bind port %d: %s", port, exc)
        raise


def metrics_text(registry: CollectorRegistry = REGISTRY) -> tuple[bytes, str]:
    """Render the registry as a (body, content_type) pair for embedding."""
    return generate_latest(registry), CONTENT_TYPE_LATEST


@asynccontextmanager
async def instrument_mode(mode_id: str, agent_ids: list[str]) -> AsyncIterator[None]:
    """Wrap a mode execution. Increments agent_requests_total + tracks total latency."""
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        dur = time.perf_counter() - start
        for aid in agent_ids or ["_none"]:
            AGENT_REQUESTS.labels(agent_id=aid, mode=mode_id, status=status).inc()
            AGENT_LATENCY.labels(agent_id=aid, phase="total").observe(dur)


def instrument_generate(agent_id: str) -> Callable:
    """Decorator: measures a single generate_stream span + counts completion tokens."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            tokens = 0
            try:
                async for chunk in fn(*args, **kwargs):
                    if getattr(chunk, "text", ""):
                        tokens += 1
                    yield chunk
            finally:
                AGENT_LATENCY.labels(agent_id=agent_id, phase="generate").observe(
                    time.perf_counter() - start
                )
                AGENT_TOKENS.labels(agent_id=agent_id, direction="completion").inc(tokens)

        return wrapper

    return deco
