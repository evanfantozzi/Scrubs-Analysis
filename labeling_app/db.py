"""
DuckDB schema and helpers for the wiki transcript labeling app.
- labels: one row per scene_id (funny, sad 1–5, updated_at).

"""
import json
import fcntl
import os
import threading
import duckdb
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
ROOT = Path(os.environ.get("PROJECT_ROOT"))
OUTPUT_PATH = ROOT / "data" / "labels.json"

# -----------------------------------------------------------------------------
# DB path and connection
# -----------------------------------------------------------------------------
load_dotenv()
ROOT = Path(os.environ.get("PROJECT_ROOT"))
DB_PATH = ROOT / "labeling_app" / "labels_wiki.duckdb"
LOCK_PATH = ROOT / "labeling_app" / "labels_wiki.duckdb.schema.lock"
_schema_lock = threading.Lock()


def get_conn():
    """Return a DuckDB connection to the labels DB. Schema is ensured via init_schema (under lock)."""
    path = str(DB_PATH)
    conn = duckdb.connect(path)
    with _schema_lock:
        lock_file = open(LOCK_PATH, "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            init_schema(conn)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    return conn


def init_schema(conn):
    """Create tables if missing. Migrate from old (scene_id, username) schema when present."""
    try:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'labels'"
        ).fetchall()
        col_names = [r[0] for r in cols]
    except Exception:
        col_names = []

    if "username" in col_names:
        # Migrate: one row per scene_id (keep first by updated_at)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS labels_new (
                scene_id VARCHAR PRIMARY KEY,
                funny INT NOT NULL CHECK (funny >= 1 AND funny <= 5),
                sad INT NOT NULL CHECK (sad >= 1 AND sad <= 5),
                updated_at TIMESTAMP
            )
        """)
        conn.execute("""
            INSERT INTO labels_new (scene_id, funny, sad, updated_at)
            SELECT scene_id, funny, sad, updated_at FROM (
                SELECT scene_id, funny, sad, updated_at,
                    ROW_NUMBER() OVER (PARTITION BY scene_id ORDER BY updated_at DESC NULLS LAST) AS rn
                FROM labels
            ) t WHERE rn = 1
        """)
        conn.execute("DROP TABLE labels")
        conn.execute("ALTER TABLE labels_new RENAME TO labels")
    elif not col_names:
        conn.execute("DROP TABLE IF EXISTS labels")
        conn.execute("""
            CREATE TABLE labels (
                scene_id VARCHAR PRIMARY KEY,
                funny INT NOT NULL CHECK (funny >= 1 AND funny <= 5),
                sad INT NOT NULL CHECK (sad >= 1 AND sad <= 5),
                updated_at TIMESTAMP
            )
        """)
    conn.execute("DROP TABLE IF EXISTS users")
    conn.commit()


# -----------------------------------------------------------------------------
# Labels (funny/sad per scene)
# -----------------------------------------------------------------------------
def get_label(conn, scene_id: str):
    """Return label for this scene (funny, sad, updated_at), or None."""
    r = conn.execute(
        "SELECT scene_id, funny, sad, updated_at FROM labels WHERE scene_id = ?",
        [scene_id],
    ).fetchone()
    if not r:
        return None
    return {"scene_id": r[0], "funny": r[1], "sad": r[2], "updated_at": r[3].isoformat() if r[3] else None}


def get_all_labels_for_scene(conn, scene_id: str):
    """Return label rows for a scene: funny, sad, updated_at (single row in current schema)."""
    rows = conn.execute(
        "SELECT funny, sad, updated_at FROM labels WHERE scene_id = ? ORDER BY updated_at",
        [scene_id],
    ).fetchall()
    return [
        {"funny": r[0], "sad": r[1], "updated_at": r[2].isoformat() if r[2] else None}
        for r in rows
    ]


def get_all_labels(conn):
    """Return all label rows: scene_id, funny, sad, updated_at."""
    rows = conn.execute(
        "SELECT scene_id, funny, sad, updated_at FROM labels ORDER BY updated_at"
    ).fetchall()
    return [
        {"scene_id": r[0], "funny": r[1], "sad": r[2], "updated_at": r[3].isoformat() if r[3] else None}
        for r in rows
    ]


def set_label(conn, scene_id: str, funny: int, sad: int):
    """Insert or update label for a scene."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO labels (scene_id, funny, sad, updated_at) VALUES (?, ?, ?, ?)
        ON CONFLICT (scene_id) DO UPDATE SET funny = excluded.funny, sad = excluded.sad, updated_at = excluded.updated_at
        """,
        [scene_id, funny, sad, now],
    )
    conn.commit()


def export_labels_to_json():
    """
    Write all labels (scene_id, funny, sad, updated_at) to a JSON file.
    Format matches bert/labels_new_data.json for training.
    """
    conn = get_conn()
    try:
        rows = get_all_labels(conn)
    finally:
        conn.close()
    out = [
        {"scene_id": r["scene_id"], "funny": r["funny"], "sad": r["sad"], "updated_at": r["updated_at"]}
        for r in rows
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(out)