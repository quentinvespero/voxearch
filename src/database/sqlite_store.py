import sqlite3
from contextlib import contextmanager


@contextmanager
def _connect(db_path: str):
    """Context manager that opens a connection and commits on success."""
    conn = sqlite3.connect(db_path)
    # Return rows as dict-like objects (access by column name)
    conn.row_factory = sqlite3.Row
    # Enable foreign-key enforcement (off by default in SQLite)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """
    Create tables and FTS index if they don't already exist.

    Schema:
      sources  — one row per audio source (YouTube video, podcast episode…)
      segments — one row per Whisper segment, linked to a source
      segments_fts — FTS5 virtual table mirroring segments.text for keyword search

    Triggers keep segments_fts in sync with segments automatically.
    """
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id             INTEGER   PRIMARY KEY AUTOINCREMENT,
                title          TEXT      NOT NULL,
                url            TEXT      UNIQUE NOT NULL,
                description    TEXT,
                status         TEXT      NOT NULL DEFAULT 'pending',
                added_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                upload_date    TEXT,
                season_number  INTEGER,
                episode_number INTEGER
            );

            CREATE TABLE IF NOT EXISTS segments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id  INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                start_time REAL    NOT NULL,
                end_time   REAL    NOT NULL,
                text       TEXT    NOT NULL
            );

            -- FTS5 content table: only indexes 'text', delegate storage to segments
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text,
                content=segments,
                content_rowid=id
            );

            -- Keep the FTS index in sync with the segments table
            CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text)
                    VALUES ('delete', old.id, old.text);
                INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
            END;
        """)

        # Migration: add status column to existing databases that predate it
        try:
            conn.execute(
                "ALTER TABLE sources ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"
            )
            # Pre-migration rows were fully ingested — mark them complete so they
            # aren't treated as interrupted and unnecessarily re-embedded in Qdrant.
            conn.execute("UPDATE sources SET status = 'complete'")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Migration: add description column to existing databases that predate it
        try:
            conn.execute("ALTER TABLE sources ADD COLUMN description TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Migration: add upload metadata columns
        # col_def values are hardcoded literals — not user input, safe to interpolate
        # (SQLite doesn't support ? placeholders in DDL statements)
        for col_def in [
            "upload_date    TEXT",
            "season_number  INTEGER",
            "episode_number INTEGER",
        ]:
            try:
                conn.execute(f"ALTER TABLE sources ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists


def get_source_status(db_path: str, url: str) -> str | None:
    """
    Return the ingest status of a source by URL, or None if not found.
    Possible values: 'pending', 'complete'.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM sources WHERE url = ?", (url,)
        ).fetchone()
    return row["status"] if row else None


def mark_source_complete(db_path: str, source_id: int) -> None:
    """Mark a source as fully ingested (all pipeline stages done)."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE sources SET status = 'complete' WHERE id = ?", (source_id,)
        )


def get_source_id_by_url(db_path: str, url: str) -> int | None:
    """Return the source id for a given URL, or None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
    return row["id"] if row else None


def get_source_by_id(db_path: str, source_id: int) -> dict | None:
    """Return a single source row by id, or None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    return dict(row) if row else None


def delete_source(db_path: str, url: str) -> None:
    """
    Delete a source and all its segments by URL.
    Segments are removed automatically via ON DELETE CASCADE.
    Note: corresponding Qdrant points are NOT cleaned up here.
    """
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM sources WHERE url = ?", (url,))


def delete_source_by_id(db_path: str, source_id: int) -> bool:
    """
    Delete a source and all its segments by ID.
    Returns True if a row was deleted, False if the ID was not found.
    Note: corresponding Qdrant points are NOT cleaned up here.
    """
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cursor.rowcount > 0


def insert_source_with_segments(
    db_path: str,
    title: str,
    url: str,
    description: str | None,
    upload_date: str | None,
    season_number: int | None,
    episode_number: int | None,
    segments: list[dict],
) -> tuple[int, list[int]]:
    """
    Atomically insert a source row and all its segments in a single transaction.
    If either insert fails, both are rolled back — guaranteeing the invariant:
    source row exists ↔ all segments are present.
    Returns (source_id, segment_ids).
    """
    with _connect(db_path) as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO sources
                    (title, url, description, upload_date, season_number, episode_number)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, url, description, upload_date, season_number, episode_number),
            )
        except sqlite3.IntegrityError as exc:
            # This should never happen in normal flow — the pipeline always
            # cleans up any existing row before calling this function.
            raise RuntimeError(
                f"Unexpected duplicate URL in DB: {url!r}. "
                "This indicates a bug in the pipeline's deduplication logic."
            ) from exc
        source_id = cursor.lastrowid
        segment_ids = []
        for seg in segments:
            seg_cursor = conn.execute(
                "INSERT INTO segments (source_id, start_time, end_time, text)"
                " VALUES (?, ?, ?, ?)",
                (source_id, seg["start"], seg["end"], seg["text"]),
            )
            segment_ids.append(seg_cursor.lastrowid)
        return source_id, segment_ids


def get_segments_by_source_id(db_path: str, source_id: int) -> list[dict]:
    """Return all segments for a source ordered by start_time."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, start_time, end_time, text FROM segments"
            " WHERE source_id = ? ORDER BY start_time",
            (source_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_sources(db_path: str) -> list[dict]:
    """Return all sources ordered by most recently added first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, url, description, status, added_at,"
            " upload_date, season_number, episode_number"
            " FROM sources ORDER BY added_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def search_keyword(db_path: str, query: str, limit: int = 10) -> list[dict]:
    """
    Full-text search across all segments.

    Results are ranked by FTS relevance (best match first).
    Each result includes the source title, URL, timestamps, and matched text.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.start_time,
                s.end_time,
                s.text,
                src.title AS source_title,
                src.url   AS source_url
            FROM segments_fts
            JOIN segments s   ON segments_fts.rowid = s.id
            JOIN sources  src ON s.source_id = src.id
            WHERE segments_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(row) for row in rows]
