"""SQLite store for ingested channels and videos.

Two tables, both idempotent on insert:
- channels: one row per seed channel
- videos:   one row per ingested clip (PK = YouTube video id)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from medea.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT NOT NULL UNIQUE,
    handle         TEXT,
    yt_channel_id  TEXT UNIQUE,
    label          INTEGER NOT NULL,  -- 1 = offender, 0 = control
    first_seen_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,   -- YouTube video id
    channel_id    INTEGER NOT NULL REFERENCES channels(id),
    title         TEXT,
    description   TEXT,
    upload_date   TEXT,               -- yyyymmdd
    duration      INTEGER,            -- seconds
    view_count    INTEGER,
    clip_path     TEXT,               -- relative to project root
    label         INTEGER NOT NULL,
    ingested_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_label   ON videos(label);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_channel(
    conn: sqlite3.Connection,
    url: str,
    label: int,
    handle: str | None = None,
    yt_channel_id: str | None = None,
) -> int:
    """Insert channel if new, else return existing id. Returns channels.id."""
    cur = conn.execute(
        """
        INSERT INTO channels (url, handle, yt_channel_id, label)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            handle = COALESCE(excluded.handle, channels.handle),
            yt_channel_id = COALESCE(excluded.yt_channel_id, channels.yt_channel_id)
        RETURNING id
        """,
        (url, handle, yt_channel_id, label),
    )
    row = cur.fetchone()
    return int(row["id"])


def video_exists(conn: sqlite3.Connection, video_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,))
    return cur.fetchone() is not None


def insert_video(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    channel_id: int,
    label: int,
    title: str | None,
    description: str | None,
    upload_date: str | None,
    duration: int | None,
    view_count: int | None,
    clip_path: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO videos
            (id, channel_id, label, title, description, upload_date,
             duration, view_count, clip_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            channel_id,
            label,
            title,
            description,
            upload_date,
            duration,
            view_count,
            clip_path,
        ),
    )


def count_videos(conn: sqlite3.Connection, label: int | None = None) -> int:
    if label is None:
        cur = conn.execute("SELECT COUNT(*) AS n FROM videos")
    else:
        cur = conn.execute("SELECT COUNT(*) AS n FROM videos WHERE label = ?", (label,))
    return int(cur.fetchone()["n"])
