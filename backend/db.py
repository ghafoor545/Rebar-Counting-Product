# backend/db.py

import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_conn():
    """
    Open a database connection.

    Supports two modes, configured with DB_DRIVER:

      DB_DRIVER=postgres (default): requires PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD
      DB_DRIVER=sqlite: local sqlite file (DB_PATH or data/app.db)

    PostgreSQL returns psycopg2 connection with RealDictCursor (dict rows).
    SQLite returns an adapter that accepts %s parameter style (for compatibility).
    """
    db_driver = os.getenv("DB_DRIVER", "postgres").strip().lower()

    if db_driver == "sqlite":
        db_path = os.getenv("DB_PATH", "data/app.db")
        db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        sqlite_conn.row_factory = sqlite3.Row
        return SQLiteConnectionAdapter(sqlite_conn)

    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "rebar_db")
    user = os.getenv("PGUSER", "rebar_user")
    password = os.getenv("PGPASSWORD", "")

    if password == "":
        raise RuntimeError(
            "PostgreSQL password is missing. Set PGPASSWORD in .env "
            "or set DB_DRIVER=sqlite for local development."
        )

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            cursor_factory=RealDictCursor,
        )
        return conn
    except psycopg2.OperationalError as e:
        raise RuntimeError(
            "Failed to connect to PostgreSQL.\n"
            f"  Server : {host}:{port}\n"
            f"  DB     : {dbname}\n"
            f"  User   : {user}\n"
            "  Please verify PGPASSWORD and existing DB/user (or switch DB_DRIVER=sqlite).\n"
            f"Original error: {e}"
        ) from e


class SQLiteCursorAdapter:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        q = query.replace("%s", "?")
        return self._cursor.execute(q, params or ())

    def executemany(self, query, seq_of_params):
        q = query.replace("%s", "?")
        return self._cursor.executemany(q, seq_of_params)

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [dict(r) for r in rows]

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class SQLiteConnectionAdapter:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return SQLiteCursorAdapter(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


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

    db_driver = os.getenv("DB_DRIVER", "postgres").strip().lower()
    if db_driver == "sqlite":
        users_id_column = "INTEGER PRIMARY KEY AUTOINCREMENT"
        index_sql = "CREATE INDEX IF NOT EXISTS idx_det_user_time ON detections(user_id, timestamp)"
    else:
        users_id_column = "SERIAL PRIMARY KEY"
        index_sql = "CREATE INDEX IF NOT EXISTS idx_det_user_time ON detections(user_id, timestamp DESC)"

    # Users table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {users_id_column},
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

    # Index on (user_id, timestamp) for sqlite, DESC allowed in postgres.
    cur.execute(index_sql)

    # For existing DBs that might not yet have bundle_info:
    # (CREATE TABLE above already includes it for fresh DBs)
    try:
        cur.execute(
            """
            ALTER TABLE detections
            ADD COLUMN IF NOT EXISTS bundle_info TEXT
            """
        )
    except Exception:
        # Some sqlite versions may not support ADD COLUMN IF NOT EXISTS; okay to skip.
        pass

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")
