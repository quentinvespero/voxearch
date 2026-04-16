"""
FastAPI server that wraps the CLI pipeline for the macOS GUI.

Run with:
    uv run uvicorn src.server:app --port 8765
"""

import asyncio
import json
import queue
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import AnyHttpUrl, BaseModel

from src import embedder
from src.config import DB_PATH
from src.database import sqlite_store, vector_store
from src import pipeline

app = FastAPI()


class _PipelineError:
    """Typed sentinel used to signal a pipeline exception across the queue."""
    def __init__(self, message: str) -> None:
        self.message = message


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness probe — Swift polls this after spawning the server process."""
    return {"status": "ok"}


# ── Ingest (SSE) ──────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: AnyHttpUrl
    language: str | None = None
    force: bool = False
    initial_prompt: str | None = None


class Source(BaseModel):
    id: int
    title: str
    url: str
    description: str | None = None
    status: str
    added_at: str


class SearchResult(BaseModel):
    id: int
    text: str
    start_time: float
    end_time: float
    source_title: str
    source_url: str
    # Semantic search only — None for keyword results
    score: float | None = None


@app.post("/ingest")
async def ingest_endpoint(body: IngestRequest):
    """
    Run the ingest pipeline and stream progress as Server-Sent Events.

    Event types:
      (default) — structured progress dict: {step, total, label, status, detail?}
                  or {status: "skipped"|"complete"}
      error     — pipeline raised an exception; data contains {"message": "..."}
    """
    msg_queue: queue.Queue[dict | _PipelineError | None] = queue.Queue()

    def on_progress(event: dict) -> None:
        # "batch" events are CLI-only (Rich progress bar) — don't forward to the GUI
        if event.get("status") == "batch":
            return
        msg_queue.put(event)

    def run() -> None:
        try:
            # TODO: support playlist ingestion — the server currently handles
            # only single-URL requests. Playlist selection and prefetched_metadata
            # passing would require a richer request body and a selection step.
            pipeline.ingest(
                str(body.url),
                body.language,
                body.force,
                body.initial_prompt,
                on_progress=on_progress,
            )
        except Exception as e:
            msg_queue.put(_PipelineError(str(e)))
        finally:
            # Always unblock the async reader, even if an exception was raised
            msg_queue.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(None, msg_queue.get)
            if msg is None:
                break
            if isinstance(msg, _PipelineError):
                yield f"event: error\ndata: {json.dumps({'message': msg.message})}\n\n"
            else:
                yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Sources ───────────────────────────────────────────────────────────────────

@app.get("/sources", response_model=list[Source])
async def sources_endpoint():
    """List all ingested sources, most recent first."""
    return sqlite_store.list_sources(DB_PATH)


# ── Search ────────────────────────────────────────────────────────────────────

@app.get("/search/keyword", response_model=list[SearchResult])
async def search_keyword_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """Full-text keyword search across all transcription segments."""
    return sqlite_store.search_keyword(DB_PATH, q, limit=limit)


@app.get("/search/semantic", response_model=list[SearchResult])
async def search_semantic_endpoint(q: str = Query(..., min_length=1), limit: int = 10):
    """Semantic similarity search using the same embedding model as ingest."""
    try:
        # Run in a thread — embed_texts is CPU-bound and would block the event loop
        query_vector = (await asyncio.to_thread(embedder.embed_texts, [q]))[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    return vector_store.search_semantic(query_vector, limit=limit)


# ── Sources (delete) ──────────────────────────────────────────────────────────

@app.delete("/sources/{source_id}")
async def delete_source_endpoint(source_id: int):
    """Delete a source and all its segments by ID (SQLite + Qdrant)."""
    deleted = sqlite_store.delete_source_by_id(DB_PATH, source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")
    vector_store.delete_by_source_id(source_id)
    return {"deleted": source_id}
