#!/usr/bin/env python3
"""
KB retrieval daemon — v2 index (ClinicDx V1).

Serves the who_knowledge_vec_v2.mv2 index over a simple HTTP API on port 4276.
All log output is newline-delimited JSON on stdout.

Endpoints:
  GET  /health           → {"ok": true, "index": "v2"}
  GET  /stats            → {"ok": true, "stats": {...}}
  POST /search           → {"ok": true, "hit": {...}, "hits": [...]}

Launch (inside container):
  python3 -m kb.daemon_v2
  python3 -m kb.daemon_v2 4276   # explicit port
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from pythonjsonlogger import jsonlogger

try:
    from kb.retrieval_core_v2 import KBRetriever
except ImportError:
    from retrieval_core_v2 import KBRetriever  # type: ignore[no-redef]

# ── Structured JSON logging ────────────────────────────────────────────────────

def _configure_logging() -> None:
    """Configure root logger to emit newline-delimited JSON on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "service"},
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


LOGGER = logging.getLogger("kb-daemon-v2")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = int(os.environ.get("KB_PORT", "4276"))
VEC_INDEX_V2 = os.environ.get(
    "KB_INDEX_PATH",
    "/kb_data/who_knowledge_vec_v2.mv2",
)

RETRIEVER: KBRetriever
CONFIG: Dict[str, Any]


def _json_response(
    handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]
) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class KBHandler(BaseHTTPRequestHandler):
    """HTTP request handler for KB search API."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {"ok": True, "index": "v2"})
            return
        if self.path == "/stats":
            try:
                _json_response(
                    self, 200, {"ok": True, "index": "v2", "stats": RETRIEVER.stats()}
                )
            except Exception as exc:
                LOGGER.exception("Stats error")
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/search":
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, 400, {"ok": False, "error": "invalid_json"})
            return

        query = (body.get("query") or "").strip()
        if not query:
            _json_response(self, 400, {"ok": False, "error": "query_required"})
            return

        k = int(body.get("k", CONFIG["k"]))
        snippet_chars = int(body.get("snippet_chars", CONFIG["snippet_chars"]))
        threshold = float(body.get("threshold", CONFIG["threshold"]))
        search_mode = str(body.get("search_mode", CONFIG["search_mode"]))
        safe_top1_guardrail = bool(
            body.get("safe_top1_guardrail", CONFIG["safe_top1_guardrail"])
        )

        LOGGER.info(
            "KB search",
            extra={
                "query": query[:120],
                "k": k,
                "mode": search_mode,
                "client": self.client_address[0],
            },
        )

        result = RETRIEVER.search(
            query=query,
            k=k,
            snippet_chars=snippet_chars,
            threshold=threshold,
            search_mode=search_mode,
            safe_top1_guardrail=safe_top1_guardrail,
        )

        top_hit = result.get("hit")
        LOGGER.info(
            "KB response",
            extra={
                "query":                  query[:120],
                "n_hits":                 len(result.get("hits") or []),
                "top_score":              round(top_hit["score"], 6) if top_hit else None,
                "top_content_type":       top_hit.get("content_type") if top_hit else None,
                "top_retrieval_priority": top_hit.get("retrieval_priority") if top_hit else None,
                "top1_swapped":           result.get("top1_swapped"),
                "quality_flags":          result.get("quality_flags") or None,
                "errors":                 result.get("errors") or None,
                "latency_ms":             round(result.get("latency_ms", 0), 1),
            },
        )

        _json_response(self, 200, {"ok": True, **result})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug("HTTP %s", fmt % args, extra={"client": self.address_string()})


def main() -> None:
    global RETRIEVER, CONFIG

    _configure_logging()

    CONFIG = {
        "k": int(os.environ.get("KB_K", "5")),
        "snippet_chars": int(os.environ.get("KB_SNIPPET_CHARS", "15000")),
        "threshold": float(os.environ.get("KB_THRESHOLD", "0.0")),
        "search_mode": os.environ.get("KB_SEARCH_MODE", "rrf"),
        "safe_top1_guardrail": os.environ.get("KB_SAFE_GUARDRAIL", "false").lower() == "true",
    }

    LOGGER.info("Loading KB index", extra={"index": VEC_INDEX_V2})
    RETRIEVER = KBRetriever(who_index=VEC_INDEX_V2)
    RETRIEVER.initialize(enable_vec=True)
    LOGGER.info("KB index loaded", extra={"index": VEC_INDEX_V2})

    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer((DEFAULT_HOST, port), KBHandler)
    LOGGER.info(
        "KB daemon v2 ready",
        extra={"host": DEFAULT_HOST, "port": port, "index": VEC_INDEX_V2},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
