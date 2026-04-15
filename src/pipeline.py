import os
from collections.abc import Callable

from src import downloader, embedder, transcriber
from src.config import AUDIO_DIR, DB_PATH, TRANSCRIPTION_MODEL
from src.database import sqlite_store, vector_store
from src.utils import normalize_url, is_hf_model_cached


INGEST_STEPS = 4


def _run_qdrant_step(
    source_id: int,
    source_title: str,
    source_url: str,
    segment_ids: list[int],
    segments: list[dict],
    step: int,
    total: int,
    on_progress: Callable[[dict], None],
) -> None:
    """
    Embed segments and store them in Qdrant.
    Raises on failure — caller is responsible for cleanup.
    """
    on_progress({"step": step, "total": total, "label": "Embedding (Qdrant)", "status": "running"})
    texts = [s["text"] for s in segments]
    vectors = embedder.embed_texts(texts)
    payloads = [
        {
            "source_id": source_id,
            "source_title": source_title,
            "source_url": source_url,
            "start_time": s["start"],
            "end_time": s["end"],
            "text": s["text"],
        }
        for s in segments
    ]
    vector_store.insert_segments(segment_ids, vectors, payloads)
    on_progress({"step": step, "total": total, "label": "Embedding (Qdrant)", "status": "done", "detail": f"{len(segment_ids)} embeddings stored"})


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
      3. Store source + segments in SQLite atomically (keyword search)
      4. Embed segments and store in Qdrant (semantic search)

    Resume behaviour (without --force):
      - status = 'complete'  → skip entirely
      - status = 'pending'   → all segments are in SQLite; resume from step 4 only
      - no row               → run from scratch (yt-dlp reuses cached audio file)

    The invariant enforced by this pipeline:
      source row exists in SQLite ↔ all segments are present (atomic insert).

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

    # ── Deduplication / resume check ─────────────────────────────────────────
    if not force:
        status = sqlite_store.get_source_status(DB_PATH, url)

        if status == "complete":
            on_progress({"status": "skipped", "detail": f"{url} already ingested. Use --force to re-process."})
            return

        if status == "pending":
            # All segments are in SQLite — only Qdrant embedding is missing.
            # Read segments back from SQLite and resume from step 4.
            source_id = sqlite_store.get_source_id_by_url(DB_PATH, url)
            if source_id is None:
                # Should never happen: 'pending' status implies a row exists.
                raise RuntimeError(f"Inconsistent DB state: status='pending' but no row found for {url}")

            db_segments = sqlite_store.get_segments_by_source_id(DB_PATH, source_id)
            segment_ids = [s["id"] for s in db_segments]

            # Retrieve source title for Qdrant payloads
            source_row = sqlite_store.get_source_by_id(DB_PATH, source_id)
            if source_row is None:
                # Should never happen: source_id was just resolved from the same URL.
                raise RuntimeError(f"Inconsistent DB state: source id={source_id} vanished mid-pipeline for {url}")

            # Normalize DB segment keys (start_time/end_time) to the transcriber
            # format (start/end) so _run_qdrant_step only handles one shape.
            db_segments = [{"start": s["start_time"], "end": s["end_time"], "text": s["text"]} for s in db_segments]

            on_progress({"status": "resuming", "detail": "SQLite already indexed — resuming from Qdrant step"})

            # Any partial Qdrant state from a previous interrupted run is safe to
            # overwrite: insert_segments upserts by SQLite segment ID, so re-running
            # is idempotent.
            _run_qdrant_step(
                source_id, source_row["title"], url,
                segment_ids, db_segments,
                step=4, total=INGEST_STEPS,
                on_progress=on_progress,
            )

            sqlite_store.mark_source_complete(DB_PATH, source_id)
            on_progress({"status": "complete"})
            return

    # ── Force or no existing record: run from scratch ─────────────────────────
    # When force=True we didn't check status above, so look up the existing id.
    # When force=False and status=None there is no row — skip the DB round-trip.
    if force:
        existing_id = sqlite_store.get_source_id_by_url(DB_PATH, url)
        if existing_id is not None:
            vector_store.delete_by_source_id(existing_id)   # remove stale Qdrant vectors
            sqlite_store.delete_source_by_id(DB_PATH, existing_id)  # cascades to segments

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
    # Source row + all segments are inserted atomically. If this fails, no
    # partial state is left — the next run starts from scratch.
    on_progress({"step": 3, "total": INGEST_STEPS, "label": "Indexing (SQLite)", "status": "running"})
    source_id, segment_ids = sqlite_store.insert_source_with_segments(
        DB_PATH,
        audio_info["title"],
        url,
        audio_info.get("description"),
        upload_date=audio_info.get("upload_date"),
        season_number=audio_info.get("season_number"),
        episode_number=audio_info.get("episode_number"),
        segments=segments,
    )
    on_progress({"step": 3, "total": INGEST_STEPS, "label": "Indexing (SQLite)", "status": "done", "detail": f"{len(segment_ids)} segments"})

    # ── 4. Qdrant ────────────────────────────────────────────────────────────
    # Source is now 'pending'. If this step fails or is interrupted, the next
    # run will detect the 'pending' status and resume here (no rollback needed).
    _run_qdrant_step(
        source_id, audio_info["title"], url,
        segment_ids, segments,
        step=4, total=INGEST_STEPS,
        on_progress=on_progress,
    )

    sqlite_store.mark_source_complete(DB_PATH, source_id)
    on_progress({"status": "complete"})
