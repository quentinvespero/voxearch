"""
Tests for pipeline.py orchestrator.

All external I/O (downloader, transcriber, embedder, sqlite_store, vector_store,
filesystem) is mocked — no model weights, network, or real DB required.
"""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from src import pipeline
from src.config import DB_PATH


# ── Shared test data ──────────────────────────────────────────────────────────

_URL = "https://example.com/episode"

_AUDIO_INFO = {
    "title": "Test Episode",
    "file_path": "/tmp/test.mp3",
    "description": "A test podcast episode about Python",
    "upload_date": "2024-01-15",
    "season_number": None,
    "episode_number": 1,
}

_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "Hello world"},
    {"start": 5.0, "end": 10.0, "text": "Goodbye world"},
]

# Segment shape as returned by sqlite_store (start_time/end_time instead of start/end)
_DB_SEGMENTS = [
    {"id": 10, "start_time": 0.0, "end_time": 5.0, "text": "Hello world"},
    {"id": 11, "start_time": 5.0, "end_time": 10.0, "text": "Goodbye world"},
]

_VECTORS = [[0.1] * 384, [0.2] * 384]
_SOURCE_ID = 42
_SEGMENT_IDS = [10, 11]


# ── Mock factory ──────────────────────────────────────────────────────────────

def _make_mocks(**overrides) -> dict:
    """
    Create a fresh set of pipeline mocks with sensible defaults.
    Keyword overrides replace individual entries.
    """
    base = {
        "src.pipeline.os.makedirs": MagicMock(),
        "src.pipeline.sqlite_store.init_db": MagicMock(),
        "src.pipeline.sqlite_store.get_source_status": MagicMock(return_value=None),
        "src.pipeline.sqlite_store.get_source_id_by_url": MagicMock(return_value=None),
        "src.pipeline.sqlite_store.get_source_by_id": MagicMock(
            return_value={"title": "Test Episode", "url": _URL}
        ),
        "src.pipeline.sqlite_store.get_segments_by_source_id": MagicMock(return_value=_DB_SEGMENTS),
        "src.pipeline.sqlite_store.insert_source_with_segments": MagicMock(
            return_value=(_SOURCE_ID, _SEGMENT_IDS)
        ),
        "src.pipeline.sqlite_store.mark_source_complete": MagicMock(),
        "src.pipeline.sqlite_store.delete_source_by_id": MagicMock(),
        "src.pipeline.downloader.download_audio": MagicMock(return_value=_AUDIO_INFO),
        "src.pipeline.transcriber.transcribe": MagicMock(return_value=_SEGMENTS),
        "src.pipeline.embedder.embed_texts": MagicMock(return_value=_VECTORS),
        "src.pipeline.vector_store.insert_segments": MagicMock(),
        "src.pipeline.vector_store.delete_by_source_id": MagicMock(),
        "src.pipeline.is_hf_model_cached": MagicMock(return_value=True),
    }
    base.update(overrides)
    return base


def _run(mocks: dict | None = None, **ingest_kwargs) -> tuple[dict, list[dict]]:
    """
    Run pipeline.ingest(_URL) with all external I/O mocked.
    Returns (patched_mocks_dict, captured_progress_events).
    """
    if mocks is None:
        mocks = _make_mocks()
    progress: list[dict] = []
    with ExitStack() as stack:
        patched = {k: stack.enter_context(patch(k, v)) for k, v in mocks.items()}
        pipeline.ingest(_URL, on_progress=lambda e: progress.append(e), **ingest_kwargs)
    return patched, progress


# ── complete → skip ───────────────────────────────────────────────────────────

class TestCompleteSkip:
    @pytest.fixture(scope="class")
    def result(self):
        mocks = _make_mocks(**{
            "src.pipeline.sqlite_store.get_source_status": MagicMock(return_value="complete")
        })
        return _run(mocks)

    def test_emits_skipped_event(self, result):
        _, progress = result
        assert any(e.get("status") == "skipped" for e in progress)

    def test_does_not_download(self, result):
        patched, _ = result
        patched["src.pipeline.downloader.download_audio"].assert_not_called()

    def test_does_not_transcribe(self, result):
        patched, _ = result
        patched["src.pipeline.transcriber.transcribe"].assert_not_called()

    def test_does_not_embed(self, result):
        patched, _ = result
        patched["src.pipeline.embedder.embed_texts"].assert_not_called()

    def test_does_not_insert_to_sqlite(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.insert_source_with_segments"].assert_not_called()


# ── pending → resume from Qdrant ─────────────────────────────────────────────

class TestPendingResume:
    @pytest.fixture(scope="class")
    def result(self):
        mocks = _make_mocks(**{
            "src.pipeline.sqlite_store.get_source_status": MagicMock(return_value="pending"),
            "src.pipeline.sqlite_store.get_source_id_by_url": MagicMock(return_value=_SOURCE_ID),
        })
        return _run(mocks)

    def test_skips_download(self, result):
        patched, _ = result
        patched["src.pipeline.downloader.download_audio"].assert_not_called()

    def test_skips_transcription(self, result):
        patched, _ = result
        patched["src.pipeline.transcriber.transcribe"].assert_not_called()

    def test_skips_sqlite_insert(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.insert_source_with_segments"].assert_not_called()

    def test_runs_qdrant_embedding(self, result):
        patched, _ = result
        patched["src.pipeline.embedder.embed_texts"].assert_called_once()

    def test_marks_complete_after_qdrant(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.mark_source_complete"].assert_called_once_with(
            DB_PATH, _SOURCE_ID
        )

    def test_emits_complete_event(self, result):
        _, progress = result
        assert any(e.get("status") == "complete" for e in progress)


# ── fresh ingest (no existing row) ───────────────────────────────────────────

class TestFreshIngest:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_downloads_audio(self, result):
        patched, _ = result
        patched["src.pipeline.downloader.download_audio"].assert_called_once()

    def test_transcribes_audio(self, result):
        patched, _ = result
        patched["src.pipeline.transcriber.transcribe"].assert_called_once()

    def test_inserts_to_sqlite(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.insert_source_with_segments"].assert_called_once()

    def test_embeds_and_stores_in_qdrant(self, result):
        patched, _ = result
        patched["src.pipeline.embedder.embed_texts"].assert_called_once()
        patched["src.pipeline.vector_store.insert_segments"].assert_called_once()

    def test_marks_complete(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.mark_source_complete"].assert_called_once_with(
            DB_PATH, _SOURCE_ID
        )

    def test_emits_complete_event(self, result):
        _, progress = result
        assert any(e.get("status") == "complete" for e in progress)

    def test_progress_covers_all_four_steps(self, result):
        _, progress = result
        labels = {e.get("label") for e in progress}
        assert pipeline.LABEL_DOWNLOAD in labels
        assert pipeline.LABEL_TRANSCRIBE in labels
        assert pipeline.LABEL_SQLITE in labels
        assert pipeline.LABEL_EMBED in labels


# ── force re-ingest ───────────────────────────────────────────────────────────

class TestForceReingest:
    @pytest.fixture(scope="class")
    def result(self):
        # Simulate an existing record that should be wiped before re-ingesting
        mocks = _make_mocks(**{
            "src.pipeline.sqlite_store.get_source_id_by_url": MagicMock(return_value=_SOURCE_ID),
        })
        return _run(mocks, force=True)

    def test_does_not_check_source_status(self, result):
        # force=True bypasses the deduplication/resume check entirely
        patched, _ = result
        patched["src.pipeline.sqlite_store.get_source_status"].assert_not_called()

    def test_deletes_existing_from_qdrant(self, result):
        patched, _ = result
        patched["src.pipeline.vector_store.delete_by_source_id"].assert_called_once_with(_SOURCE_ID)

    def test_deletes_existing_from_sqlite(self, result):
        patched, _ = result
        patched["src.pipeline.sqlite_store.delete_source_by_id"].assert_called_once_with(
            DB_PATH, _SOURCE_ID
        )

    def test_runs_full_pipeline_after_deletion(self, result):
        patched, _ = result
        patched["src.pipeline.downloader.download_audio"].assert_called_once()
        patched["src.pipeline.transcriber.transcribe"].assert_called_once()
        patched["src.pipeline.sqlite_store.insert_source_with_segments"].assert_called_once()
        patched["src.pipeline.embedder.embed_texts"].assert_called_once()


def test_force_on_fresh_url_skips_deletion_and_runs_full_pipeline():
    # force=True but no existing row — deletion must be skipped entirely
    patched, _ = _run(force=True)  # default mock: get_source_id_by_url returns None
    patched["src.pipeline.vector_store.delete_by_source_id"].assert_not_called()
    patched["src.pipeline.sqlite_store.delete_source_by_id"].assert_not_called()
    patched["src.pipeline.downloader.download_audio"].assert_called_once()
    patched["src.pipeline.embedder.embed_texts"].assert_called_once()


# ── Qdrant failure ────────────────────────────────────────────────────────────

def _qdrant_error_mocks() -> dict:
    # Fresh mock each call — avoids shared state across test runs
    return {"src.pipeline.embedder.embed_texts": MagicMock(side_effect=RuntimeError("embed failed"))}


class TestQdrantFailure:
    """
    When the Qdrant step fails, the source row stays in 'pending' state in SQLite.
    The next ingest run detects 'pending' and resumes from Qdrant — no rollback needed.
    """
    @pytest.fixture(scope="class")
    def result(self):
        """Run ingest with a Qdrant error; swallows the exception for post-run assertions."""
        mocks = _make_mocks(**_qdrant_error_mocks())
        progress: list[dict] = []
        with ExitStack() as stack:
            patched = {k: stack.enter_context(patch(k, v)) for k, v in mocks.items()}
            try:
                pipeline.ingest(_URL, on_progress=lambda e: progress.append(e))
            except RuntimeError:
                pass
        return patched, progress

    def test_exception_propagates_to_caller(self):
        # Runs independently — verifies the exception actually surfaces to callers
        with pytest.raises(RuntimeError, match="embed failed"):
            _run(_make_mocks(**_qdrant_error_mocks()))

    def test_source_not_marked_complete(self, result):
        # Source must stay 'pending' so the next run can resume from Qdrant
        patched, _ = result
        patched["src.pipeline.sqlite_store.mark_source_complete"].assert_not_called()

    def test_sqlite_segments_not_deleted(self, result):
        # SQLite data must survive — it's the recovery state for the next run
        patched, _ = result
        patched["src.pipeline.sqlite_store.delete_source_by_id"].assert_not_called()


# ── auto_context ──────────────────────────────────────────────────────────────

class TestAutoContext:
    def test_title_and_description_in_prompt(self):
        # auto_context=True is the default — Whisper receives title + description as context
        patched, _ = _run(auto_context=True)
        prompt = patched["src.pipeline.transcriber.transcribe"].call_args.kwargs.get("initial_prompt") or ""
        assert "Test Episode" in prompt
        assert "A test podcast episode about Python" in prompt

    def test_auto_context_with_initial_prompt_appends_after_auto(self):
        # Both auto-context and user-supplied prompt must appear, auto-context first
        patched, _ = _run(auto_context=True, initial_prompt="React, TypeScript")
        prompt = patched["src.pipeline.transcriber.transcribe"].call_args.kwargs.get("initial_prompt") or ""
        assert "Test Episode" in prompt
        assert "React, TypeScript" in prompt
        assert prompt.index("Test Episode") < prompt.index("React, TypeScript")

    def test_auto_context_false_uses_only_initial_prompt(self):
        patched, _ = _run(auto_context=False, initial_prompt="React, TypeScript")
        prompt = patched["src.pipeline.transcriber.transcribe"].call_args.kwargs.get("initial_prompt")
        assert prompt == "React, TypeScript"
        assert "Test Episode" not in prompt

    def test_auto_context_false_with_no_initial_prompt(self):
        patched, _ = _run(auto_context=False)
        prompt = patched["src.pipeline.transcriber.transcribe"].call_args.kwargs.get("initial_prompt")
        assert prompt is None


# ── prefetched_metadata ───────────────────────────────────────────────────────

class TestPrefetchedMetadata:
    def _capture_insert(self, captured: dict):
        """Side-effect for insert_source_with_segments that records title and description by name."""
        def _side_effect(db_path, title, url, description, **kwargs):
            captured["title"] = title
            captured["description"] = description
            return (_SOURCE_ID, _SEGMENT_IDS)
        return _side_effect

    def test_prefetched_title_overrides_yt_dlp_title(self):
        # RSS/playlist title takes priority over whatever yt-dlp resolved
        captured: dict = {}
        mocks = _make_mocks(**{
            "src.pipeline.sqlite_store.insert_source_with_segments": MagicMock(
                side_effect=self._capture_insert(captured)
            )
        })
        _run(mocks, prefetched_metadata={"title": "RSS Title", "description": None})
        assert captured["title"] == "RSS Title"

    def test_prefetched_description_fills_missing_field(self):
        # yt-dlp returned no description; prefetched_metadata fills it in
        captured: dict = {}
        mocks = _make_mocks(**{
            "src.pipeline.downloader.download_audio": MagicMock(return_value={
                **_AUDIO_INFO,
                "description": None,
            }),
            "src.pipeline.sqlite_store.insert_source_with_segments": MagicMock(
                side_effect=self._capture_insert(captured)
            ),
        })
        _run(mocks, prefetched_metadata={"title": None, "description": "Prefetched desc"})
        assert captured["description"] == "Prefetched desc"

    def test_yt_dlp_description_not_overridden(self):
        # yt-dlp already has a description; prefetched_metadata must not overwrite it
        captured: dict = {}
        mocks = _make_mocks(**{
            "src.pipeline.sqlite_store.insert_source_with_segments": MagicMock(
                side_effect=self._capture_insert(captured)
            )
        })
        _run(mocks, prefetched_metadata={"title": None, "description": "Should be ignored"})
        assert captured["description"] == _AUDIO_INFO["description"]
