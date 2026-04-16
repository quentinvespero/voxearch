import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.server import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    # Use as context manager so FastAPI lifespan handlers (startup/shutdown) run correctly
    with TestClient(app) as c:
        yield c


@pytest.fixture
def error_client():
    # raise_server_exceptions=False converts unhandled exceptions into 500 responses
    # instead of re-raising them in the test process — needed to assert on 500 status codes
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Shared mock data ──────────────────────────────────────────────────────────

_MOCK_SOURCES = [
    {
        "id": 1,
        "title": "Episode 1",
        "url": "https://example.com/ep1",
        "description": "First episode",
        "status": "complete",
        "added_at": "2024-01-15 10:00:00",
        "upload_date": None,
        "season_number": None,
        "episode_number": None,
    }
]

_MOCK_SEARCH_RESULTS = [
    {
        "id": 42,
        "text": "Hello world",
        "start_time": 0.0,
        "end_time": 5.0,
        "source_title": "Episode 1",
        "source_url": "https://example.com/ep1",
        "score": None,
    }
]

_MOCK_SEMANTIC_RESULTS = [
    {
        "id": 42,
        "text": "Hello world",
        "start_time": 0.0,
        "end_time": 5.0,
        "source_title": "Episode 1",
        "source_url": "https://example.com/ep1",
        "score": 0.95,
    }
]


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── GET /sources ──────────────────────────────────────────────────────────────

class TestSourcesEndpoint:
    def test_returns_empty_list(self, client):
        with patch("src.server.sqlite_store.list_sources", return_value=[]):
            response = client.get("/sources")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_sources(self, client):
        with patch("src.server.sqlite_store.list_sources", return_value=_MOCK_SOURCES):
            response = client.get("/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Episode 1"
        assert data[0]["status"] == "complete"


# ── DELETE /sources/{id} ──────────────────────────────────────────────────────

class TestDeleteSourceEndpoint:
    def test_success_returns_deleted_id(self, client):
        with (
            patch("src.server.sqlite_store.delete_source_by_id", return_value=True),
            patch("src.server.vector_store.delete_by_source_id"),
        ):
            response = client.delete("/sources/1")
        assert response.status_code == 200
        assert response.json() == {"deleted": 1}

    def test_not_found_returns_404(self, client):
        with patch("src.server.sqlite_store.delete_source_by_id", return_value=False):
            response = client.delete("/sources/9999")
        assert response.status_code == 404

    def test_also_deletes_from_vector_store(self, client):
        mock_vec_delete = MagicMock()
        with (
            patch("src.server.sqlite_store.delete_source_by_id", return_value=True),
            patch("src.server.vector_store.delete_by_source_id", mock_vec_delete),
        ):
            client.delete("/sources/5")
        mock_vec_delete.assert_called_once_with(5)


# ── GET /search/keyword ───────────────────────────────────────────────────────

class TestSearchKeywordEndpoint:
    def test_returns_results(self, client):
        with patch(
            "src.server.sqlite_store.search_keyword", return_value=_MOCK_SEARCH_RESULTS
        ):
            response = client.get("/search/keyword?q=hello")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "Hello world"

    def test_missing_q_returns_422(self, client):
        response = client.get("/search/keyword")
        assert response.status_code == 422

    def test_empty_q_returns_422(self, client):
        # min_length=1 on the Query parameter
        response = client.get("/search/keyword?q=")
        assert response.status_code == 422

    def test_passes_limit_param(self, client):
        mock_search = MagicMock(return_value=[])
        with patch("src.server.sqlite_store.search_keyword", mock_search):
            client.get("/search/keyword?q=test&limit=5")
        assert mock_search.call_args.kwargs["limit"] == 5


# ── GET /search/semantic ──────────────────────────────────────────────────────

class TestSearchSemanticEndpoint:
    def test_returns_results(self, client):
        mock_vector = [0.1] * 384
        with (
            patch(
                "src.server.embedder.embed_texts", return_value=[mock_vector]
            ),
            patch(
                "src.server.vector_store.search_semantic",
                return_value=_MOCK_SEMANTIC_RESULTS,
            ),
        ):
            response = client.get("/search/semantic?q=hello")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["score"] == pytest.approx(0.95)

    def test_missing_q_returns_422(self, client):
        response = client.get("/search/semantic")
        assert response.status_code == 422

    def test_embedding_failure_returns_500(self, client):
        # server.py wraps embed_texts in try/except and raises HTTPException(500)
        with patch(
            "src.server.embedder.embed_texts", side_effect=RuntimeError("model not loaded")
        ):
            response = client.get("/search/semantic?q=hello")
        assert response.status_code == 500
        assert "Embedding failed" in response.json()["detail"]


# ── POST /ingest (SSE) ────────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE response text into a list of (event_type, data) dicts."""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = block.strip().splitlines()
        event_type = "message"
        data = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if data is not None:
            events.append({"type": event_type, "data": data})
    return events


class TestIngestEndpoint:
    _PAYLOAD = {"url": "https://example.com/ep1"}

    def _mock_pipeline(self, on_progress_events: list[dict], *, raises: Exception | None = None):
        """Return a side_effect function for pipeline.ingest that drives on_progress."""
        def _ingest(url, language, force, initial_prompt, on_progress, auto_context, **kwargs):
            for ev in on_progress_events:
                on_progress(ev)
            if raises is not None:
                raise raises
        return _ingest

    def test_missing_url_returns_422(self, client):
        response = client.post("/ingest", json={})
        assert response.status_code == 422

    def test_returns_200_with_sse_content_type(self, client):
        with patch("src.server.pipeline.ingest", self._mock_pipeline([])):
            response = client.post("/ingest", json=self._PAYLOAD)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_progress_events_forwarded_to_stream(self, client):
        events = [
            {"step": 1, "total": 4, "label": "Downloading", "status": "progress"},
            {"step": 4, "total": 4, "label": "Done",        "status": "complete"},
        ]
        with patch("src.server.pipeline.ingest", self._mock_pipeline(events)):
            response = client.post("/ingest", json=self._PAYLOAD)
        parsed = _parse_sse(response.text)
        assert len(parsed) == 2
        assert parsed[0]["data"]["label"] == "Downloading"
        assert parsed[1]["data"]["status"] == "complete"

    def test_batch_events_are_filtered_out(self, client):
        # "batch" events are CLI-only and must not reach the SSE stream
        events = [
            {"status": "batch", "detail": "cli-only"},
            {"step": 1, "total": 1, "label": "Done", "status": "complete"},
        ]
        with patch("src.server.pipeline.ingest", self._mock_pipeline(events)):
            response = client.post("/ingest", json=self._PAYLOAD)
        parsed = _parse_sse(response.text)
        assert all(e["data"].get("status") != "batch" for e in parsed)
        assert len(parsed) == 1

    def test_pipeline_exception_emits_error_event(self, client):
        with patch(
            "src.server.pipeline.ingest",
            self._mock_pipeline([], raises=RuntimeError("download failed")),
        ):
            response = client.post("/ingest", json=self._PAYLOAD)
        parsed = _parse_sse(response.text)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "error"
        assert "download failed" in parsed[0]["data"]["message"]

    def test_language_param_forwarded_to_pipeline(self, client):
        mock_ingest = MagicMock()
        with patch("src.server.pipeline.ingest", mock_ingest):
            client.post("/ingest", json={"url": "https://example.com/ep1", "language": "fr"})
        mock_ingest.assert_called_once()
        # language is the second positional arg (index 1)
        assert mock_ingest.call_args.args[1] == "fr"

    def test_force_param_forwarded_to_pipeline(self, client):
        mock_ingest = MagicMock()
        with patch("src.server.pipeline.ingest", mock_ingest):
            client.post("/ingest", json={"url": "https://example.com/ep1", "force": True})
        mock_ingest.assert_called_once()
        # force is the third positional arg (index 2)
        assert mock_ingest.call_args.args[2] is True

    def test_default_params_forwarded_to_pipeline(self, client):
        # When optional params are omitted, defaults from IngestRequest must reach the pipeline
        mock_ingest = MagicMock()
        with patch("src.server.pipeline.ingest", mock_ingest):
            client.post("/ingest", json={"url": "https://example.com/ep1"})
        mock_ingest.assert_called_once()
        args = mock_ingest.call_args.args
        kwargs = mock_ingest.call_args.kwargs
        assert args[1] is None       # language defaults to None
        assert args[2] is False      # force defaults to False
        assert args[3] is None       # initial_prompt defaults to None
        assert kwargs["auto_context"] is True  # auto_context defaults to True


# ── GET /search/keyword — DB error ────────────────────────────────────────────

class TestSearchKeywordDbError:
    def test_db_error_returns_500(self, error_client):
        with patch(
            "src.server.sqlite_store.search_keyword",
            side_effect=Exception("DB unavailable"),
        ):
            response = error_client.get("/search/keyword?q=test")
        assert response.status_code == 500


# ── GET /search/semantic — DB error ───────────────────────────────────────────

class TestSearchSemanticDbError:
    def test_vector_store_error_returns_500(self, error_client):
        # Embedding succeeds; vector_store.search_semantic fails (unhandled → 500)
        mock_vector = [0.1] * 384
        with (
            patch("src.server.embedder.embed_texts", return_value=[mock_vector]),
            patch(
                "src.server.vector_store.search_semantic",
                side_effect=Exception("Qdrant unavailable"),
            ),
        ):
            response = error_client.get("/search/semantic?q=test")
        assert response.status_code == 500
