import os
from collections.abc import Callable

from src import downloader, embedder, transcriber
from src.config import AUDIO_DIR, DB_PATH, TRANSCRIPTION_MODEL
from src.database import sqlite_store, vector_store
from src.utils import normalize_url, is_hf_model_cached


INGEST_STEPS = 4


def ingest(
    url: str,
    language: str | None = None,
    force: bool = False,
    initial_prompt: str | None = None,
    on_progress: Callable[[dict], None] = lambda _: None,
) -> None:
    """
    Full ingest pipeline for a single audio URL:
      1. Download audio via yt-dlp
      2. Transcribe with mlx-whisper
      3. Store segments in SQLite (keyword search)
      4. Embed segments and store in Qdrant (semantic search)

    Args:
        url:            Any URL supported by yt-dlp (YouTube, SoundCloud, etc.)
        language:       ISO 639-1 language hint for Whisper (e.g. "fr", "en").
                        None = auto-detect (slightly slower).
        force:          Re-download and re-transcribe even if already processed.
        initial_prompt: Optional context hint for Whisper (e.g. "React, TypeScript").
                        See transcriber.transcribe() for details.
        on_progress:    Callback receiving structured progress dicts. Shape:
                          {"step": int, "total": int, "label": str, "status": "running"|"done", "detail": str}
                          {"status": "skipped", "detail": str}
                          {"status": "complete"}
                        Defaults to a no-op so the function is safe to call without a callback.
    """
    url = normalize_url(url)

    # Ensure storage directories exist before any file operations
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    sqlite_store.init_db(DB_PATH)

    # ── Deduplication check ───────────────────────────────────────────────────
    if not force:
        status = sqlite_store.get_source_status(DB_PATH, url)
        if status == "complete":
            on_progress({"status": "skipped", "detail": f"{url} already ingested. Use --force to re-process."})
            return

    if force:
        # Wipe existing data so the pipeline runs fresh.
        existing_id = sqlite_store.get_source_id_by_url(DB_PATH, url)
        if existing_id is not None:
            vector_store.delete_by_source_id(existing_id)
        sqlite_store.delete_source(DB_PATH, url)

    # ── 1. Download ──────────────────────────────────────────────────────────
    on_progress({"step": 1, "total": INGEST_STEPS, "label": "Downloading", "status": "running"})
    audio_info = downloader.download_audio(url, AUDIO_DIR, force=force)
    on_progress({"step": 1, "total": INGEST_STEPS, "label": "Downloading", "status": "done", "detail": audio_info["title"]})

    # ── 2. Transcribe ────────────────────────────────────────────────────────
    # Show "Downloading & transcribing" on first run so the user knows a model
    # download is in progress, not just audio processing.
    _transcribe_label = "Transcribing" if is_hf_model_cached(TRANSCRIPTION_MODEL) else "Downloading & transcribing"
    on_progress({"step": 2, "total": INGEST_STEPS, "label": _transcribe_label, "status": "running"})
    segments = transcriber.transcribe(audio_info["file_path"], language=language, initial_prompt=initial_prompt)
    on_progress({"step": 2, "total": INGEST_STEPS, "label": _transcribe_label, "status": "done", "detail": f"{len(segments)} segments"})

    # ── 3. SQLite ────────────────────────────────────────────────────────────
    on_progress({"step": 3, "total": INGEST_STEPS, "label": "Indexing (SQLite)", "status": "running"})
    source_id   = sqlite_store.insert_source(
        DB_PATH,
        audio_info["title"],
        url,
        audio_info["description"],
        upload_date=audio_info.get("upload_date"),
        season_number=audio_info.get("season_number"),
        episode_number=audio_info.get("episode_number"),
    )
    segment_ids = sqlite_store.insert_segments(DB_PATH, source_id, segments)
    on_progress({"step": 3, "total": INGEST_STEPS, "label": "Indexing (SQLite)", "status": "done", "detail": f"{len(segment_ids)} segments"})

    # ── 4. Qdrant ────────────────────────────────────────────────────────────
    on_progress({"step": 4, "total": INGEST_STEPS, "label": "Embedding (Qdrant)", "status": "running"})
    try:
        texts    = [s["text"] for s in segments]
        vectors  = embedder.embed_texts(texts)
        payloads = [
            {
                "source_id":    source_id,
                "source_title": audio_info["title"],
                "source_url":   url,
                "start_time":   s["start"],
                "end_time":     s["end"],
                "text":         s["text"],
            }
            for s in segments
        ]
        vector_store.insert_segments(segment_ids, vectors, payloads)
    except Exception:
        # Roll back SQLite data to keep stores in sync — source never reached
        # 'complete' status so it's safe to wipe and allow clean re-ingestion.
        try:
            sqlite_store.delete_source_by_id(DB_PATH, source_id)
        except Exception:
            pass  # rollback failed, but we still want to surface the original error
        raise
    on_progress({"step": 4, "total": INGEST_STEPS, "label": "Embedding (Qdrant)", "status": "done", "detail": f"{len(segment_ids)} embeddings stored"})

    sqlite_store.mark_source_complete(DB_PATH, source_id)
    on_progress({"status": "complete"})
