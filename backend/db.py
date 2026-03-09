# backend/db.py

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_conn():
    """
    Open a new PostgreSQL connection.

    Connection parameters are taken from environment variables:

        PGHOST      (default: "localhost")
        PGPORT      (default: "5432")
        PGDATABASE  (default: "rebar_db")
        PGUSER      (default: "rebar_user")
        PGPASSWORD  (default: "")

    Returns a connection where cursor() yields RealDictCursor,
    so rows behave like dicts: row["column_name"].
    """
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "rebar_db")
    user = os.getenv("PGUSER", "rebar_user")
    password = os.getenv("PGPASSWORD", "")

    # Optional: Add debug print to verify variables are loaded
    # print(f"Connecting to {host}:{port}/{dbname} as {user}")

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        cursor_factory=RealDictCursor,
    )
    return conn


def init_db():
    """
    Initialize PostgreSQL schema.

    - Creates 'users' and 'detections' tables if they do not exist.
    - Ensures index on detections(user_id, timestamp DESC).
    - Ensures 'bundle_info' column exists on detections (for old DBs).

    This mirrors the SQLite schema you had before, adapted to Postgres types.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            pwd_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    # Detections table
    # id is TEXT because your code generates UUID strings in Python.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            stream_url TEXT,
            snapshot_url TEXT,
            image_path TEXT NOT NULL,
            thumb_path TEXT NOT NULL,
            count INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            bundle_info TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Index on (user_id, timestamp DESC), same as before
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_det_user_time "
        "ON detections(user_id, timestamp DESC)"
    )

    # For existing DBs that might not yet have bundle_info:
    # (CREATE TABLE above already includes it for fresh DBs)
    cur.execute(
        """
        ALTER TABLE detections
        ADD COLUMN IF NOT EXISTS bundle_info TEXT
        """
    )

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")
