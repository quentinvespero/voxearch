import sqlite3

import pytest

from src.database.sqlite_store import (
    init_db,
    get_source_status,
    insert_source_with_segments,
    mark_source_complete,
    get_source_id_by_url,
    get_source_by_id,
    delete_source_by_id,
    get_segments_by_source_id,
    list_sources,
    search_keyword,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path) -> str:
    """Initialised temp SQLite DB; returns the db_path string."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


_SAMPLE_SEGMENTS = [
    {"start": 0.0,  "end": 5.0,  "text": "Hello world"},
    {"start": 5.0,  "end": 10.0, "text": "This is a podcast episode"},
    {"start": 10.0, "end": 15.0, "text": "About machine learning"},
]


@pytest.fixture
def db_with_source(db) -> tuple[str, int]:
    """DB with one source + 3 segments already inserted. Returns (db_path, source_id)."""
    source_id, _ = insert_source_with_segments(
        db,
        title="Test Episode",
        url="https://example.com/episode1",
        description="A test episode",
        upload_date="2024-01-15",
        season_number=1,
        episode_number=3,
        segments=_SAMPLE_SEGMENTS,
    )
    return db, source_id


# ── init_db ───────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_sources_table(self, db):
        conn = sqlite3.connect(db)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
        ).fetchone()
        conn.close()
        assert result is not None

    def test_creates_segments_table(self, db):
        conn = sqlite3.connect(db)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='segments'"
        ).fetchone()
        conn.close()
        assert result is not None

    def test_creates_fts_table(self, db):
        conn = sqlite3.connect(db)
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='segments_fts'"
        ).fetchone()
        conn.close()
        assert result is not None

    def test_idempotent(self, db):
        # Calling init_db a second time should not raise
        init_db(db)


# ── insert_source_with_segments ───────────────────────────────────────────────

class TestInsertSourceWithSegments:
    def test_returns_source_id_and_segment_ids(self, db):
        source_id, seg_ids = insert_source_with_segments(
            db, "Title", "https://ex.com/1", None, None, None, None, _SAMPLE_SEGMENTS
        )
        assert isinstance(source_id, int)
        assert len(seg_ids) == len(_SAMPLE_SEGMENTS)
        assert all(isinstance(i, int) for i in seg_ids)

    def test_segments_stored_in_db(self, db_with_source):
        db, source_id = db_with_source
        segs = get_segments_by_source_id(db, source_id)
        assert len(segs) == 3

    def test_duplicate_url_raises(self, db):
        insert_source_with_segments(
            db, "Title", "https://ex.com/dup", None, None, None, None, []
        )
        with pytest.raises(RuntimeError, match="Unexpected duplicate URL"):
            insert_source_with_segments(
                db, "Title2", "https://ex.com/dup", None, None, None, None, []
            )

    def test_initial_status_is_pending(self, db_with_source):
        db, _ = db_with_source
        status = get_source_status(db, "https://example.com/episode1")
        assert status == "pending"


# ── get_source_status ─────────────────────────────────────────────────────────

class TestGetSourceStatus:
    def test_unknown_url_returns_none(self, db):
        assert get_source_status(db, "https://never-inserted.com") is None

    def test_pending_after_insert(self, db_with_source):
        db, _ = db_with_source
        assert get_source_status(db, "https://example.com/episode1") == "pending"

    def test_complete_after_mark(self, db_with_source):
        db, source_id = db_with_source
        mark_source_complete(db, source_id)
        assert get_source_status(db, "https://example.com/episode1") == "complete"


# ── mark_source_complete ──────────────────────────────────────────────────────

class TestMarkSourceComplete:
    def test_flips_status_to_complete(self, db_with_source):
        db, source_id = db_with_source
        mark_source_complete(db, source_id)
        row = get_source_by_id(db, source_id)
        assert row["status"] == "complete"


# ── get_source_id_by_url ──────────────────────────────────────────────────────

class TestGetSourceIdByUrl:
    def test_returns_id_for_known_url(self, db_with_source):
        db, source_id = db_with_source
        result = get_source_id_by_url(db, "https://example.com/episode1")
        assert result == source_id

    def test_returns_none_for_unknown_url(self, db):
        assert get_source_id_by_url(db, "https://not-here.com") is None


# ── get_source_by_id ──────────────────────────────────────────────────────────

class TestGetSourceById:
    def test_returns_dict_with_expected_fields(self, db_with_source):
        db, source_id = db_with_source
        row = get_source_by_id(db, source_id)
        assert row is not None
        assert row["id"] == source_id
        assert row["title"] == "Test Episode"
        assert row["url"] == "https://example.com/episode1"
        assert row["description"] == "A test episode"
        assert row["season_number"] == 1
        assert row["episode_number"] == 3

    def test_returns_none_for_missing_id(self, db):
        assert get_source_by_id(db, 9999) is None


# ── delete_source_by_id ───────────────────────────────────────────────────────

class TestDeleteSourceById:
    def test_returns_true_on_success(self, db_with_source):
        db, source_id = db_with_source
        assert delete_source_by_id(db, source_id) is True

    def test_returns_false_for_missing_id(self, db):
        assert delete_source_by_id(db, 9999) is False

    def test_source_gone_after_delete(self, db_with_source):
        db, source_id = db_with_source
        delete_source_by_id(db, source_id)
        assert get_source_by_id(db, source_id) is None

    def test_cascades_to_segments(self, db_with_source):
        db, source_id = db_with_source
        delete_source_by_id(db, source_id)
        assert get_segments_by_source_id(db, source_id) == []


# ── list_sources ──────────────────────────────────────────────────────────────

class TestListSources:
    def test_empty_when_no_sources(self, db):
        assert list_sources(db) == []

    def test_returns_all_sources(self, db):
        insert_source_with_segments(db, "A", "https://ex.com/a", None, None, None, None, [])
        insert_source_with_segments(db, "B", "https://ex.com/b", None, None, None, None, [])
        assert len(list_sources(db)) == 2

    def test_most_recent_first(self, db):
        # Use the public API for inserts (schema-safe), then patch added_at directly
        # because CURRENT_TIMESTAMP has 1-second resolution and both inserts would
        # otherwise share the same timestamp within a fast test run.
        id_older, _ = insert_source_with_segments(db, "Older", "https://ex.com/older", None, None, None, None, [])
        id_newer, _ = insert_source_with_segments(db, "Newer", "https://ex.com/newer", None, None, None, None, [])

        conn = sqlite3.connect(db)
        conn.execute("UPDATE sources SET added_at = ? WHERE id = ?", ("2024-01-01 10:00:00", id_older))
        conn.execute("UPDATE sources SET added_at = ? WHERE id = ?", ("2024-01-01 11:00:00", id_newer))
        conn.commit()
        conn.close()

        sources = list_sources(db)
        assert sources[0]["title"] == "Newer"
        assert sources[1]["title"] == "Older"


# ── get_segments_by_source_id ─────────────────────────────────────────────────

class TestGetSegmentsBySourceId:
    def test_returns_segments_ordered_by_start_time(self, db_with_source):
        db, source_id = db_with_source
        segs = get_segments_by_source_id(db, source_id)
        assert len(segs) == 3
        assert segs[0]["start_time"] == 0.0
        assert segs[1]["start_time"] == 5.0
        assert segs[2]["start_time"] == 10.0

    def test_returns_empty_for_missing_source(self, db):
        assert get_segments_by_source_id(db, 9999) == []

    def test_segment_fields_present(self, db_with_source):
        db, source_id = db_with_source
        seg = get_segments_by_source_id(db, source_id)[0]
        assert "id" in seg
        assert "start_time" in seg
        assert "end_time" in seg
        assert "text" in seg


# ── search_keyword ────────────────────────────────────────────────────────────

class TestSearchKeyword:
    def test_returns_matching_segment(self, db_with_source):
        db, _ = db_with_source
        results = search_keyword(db, "podcast")
        assert len(results) == 1
        assert "podcast" in results[0]["text"].lower()

    def test_returns_empty_for_no_match(self, db_with_source):
        db, _ = db_with_source
        assert search_keyword(db, "xyznomatch") == []

    def test_respects_limit(self, db_with_source):
        db, _ = db_with_source
        # "machine" only matches 1, but let's add a broader query then limit to 1
        results = search_keyword(db, "Hello OR podcast OR machine", limit=1)
        assert len(results) <= 1

    def test_result_has_source_info(self, db_with_source):
        db, _ = db_with_source
        results = search_keyword(db, "podcast")
        assert results[0]["source_title"] == "Test Episode"
        assert results[0]["source_url"] == "https://example.com/episode1"

    def test_sql_injection_attempt_does_not_corrupt_data(self, db_with_source):
        # The query is passed as a parameterized FTS expression, not interpolated into SQL,
        # so classic SQL injection cannot reach the query engine. An FTS parse error is
        # acceptable; corrupted or deleted data is not.
        db, _ = db_with_source
        try:
            search_keyword(db, "'; DROP TABLE sources; --")
        except Exception:
            pass  # FTS syntax error is fine — what matters is that the data survives
        assert list_sources(db) != [], "sources table must survive a malicious query string"
