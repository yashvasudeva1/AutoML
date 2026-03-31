import os
import json
import sqlite3
import importlib
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")
DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
USE_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")

_POOL = None


def _normalized_database_url() -> str:
    raw = DATABASE_URL
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]

    # psycopg2 rejects provider-specific query args like pgbouncer=true.
    parsed = urlparse(raw)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    allowed_pairs = [(k, v) for (k, v) in query_pairs if k.lower() != "pgbouncer"]
    clean_query = urlencode(allowed_pairs)
    return urlunparse(parsed._replace(query=clean_query))


def _get_pg_pool():
    global _POOL
    if _POOL is not None:
        return _POOL

    try:
        psycopg2_pool = importlib.import_module("psycopg2.pool")
        SimpleConnectionPool = getattr(psycopg2_pool, "SimpleConnectionPool")
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set for PostgreSQL, but psycopg2 is not installed. "
            "Install requirements and retry."
        ) from exc

    max_conn = int(os.environ.get("DB_POOL_SIZE", "10"))
    _POOL = SimpleConnectionPool(1, max_conn, dsn=_normalized_database_url())
    return _POOL


def _get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _pg_row_to_dict(cursor, row):
    if not row:
        return None
    cols = [desc[0] for desc in cursor.description]
    return {cols[i]: row[i] for i in range(len(cols))}


def init_db():
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        email TEXT UNIQUE,
                        password TEXT
                    )
                    """
                )
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        filename TEXT,
                        filepath TEXT,
                        summary TEXT,
                        col_info TEXT,
                        data_blob BYTEA,
                        uploaded_at TIMESTAMPTZ DEFAULT NOW(),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                    """
                )
                c.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'datasets' AND column_name = 'data_blob'
                    """
                )
                if c.fetchone() is None:
                    c.execute("ALTER TABLE datasets ADD COLUMN data_blob BYTEA")

            conn.commit()
        finally:
            pool.putconn(conn)
        return

    conn = _get_sqlite_conn()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            filename TEXT,
            filepath TEXT,
            summary TEXT,
            col_info TEXT,
            data_blob BLOB,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )

    cols = [row[1] for row in c.execute("PRAGMA table_info(datasets)").fetchall()]
    if "data_blob" not in cols:
        c.execute("ALTER TABLE datasets ADD COLUMN data_blob BLOB")

    conn.commit()
    conn.close()


init_db()


def create_user(user_id, name, email, password):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO users (id, name, email, password) VALUES (%s, %s, %s, %s)",
                    (user_id, name, email, password),
                )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False
        finally:
            pool.putconn(conn)

    conn = _get_sqlite_conn()
    try:
        conn.execute(
            "INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)",
            (user_id, name, email, password),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def update_user(user_id, name=None, password=None):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                if name:
                    c.execute("UPDATE users SET name = %s WHERE id = %s", (name, user_id))
                if password:
                    c.execute("UPDATE users SET password = %s WHERE id = %s", (password, user_id))
            conn.commit()
        finally:
            pool.putconn(conn)
        return

    conn = _get_sqlite_conn()
    if name:
        conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))
    if password:
        conn.execute("UPDATE users SET password = ? WHERE id = ?", (password, user_id))
    conn.commit()
    conn.close()


def get_user_by_email(email):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute("SELECT * FROM users WHERE email = %s", (email,))
                row = c.fetchone()
                return _pg_row_to_dict(c, row)
        finally:
            pool.putconn(conn)

    conn = _get_sqlite_conn()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = c.fetchone()
                return _pg_row_to_dict(c, row)
        finally:
            pool.putconn(conn)

    conn = _get_sqlite_conn()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def create_dataset(dataset_id, user_id, filename, filepath, summary, col_info, data_blob=None):
    summary_s = json.dumps(summary)
    col_info_s = json.dumps(col_info)

    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute(
                    """
                    INSERT INTO datasets (id, user_id, filename, filepath, summary, col_info, data_blob)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (dataset_id, user_id, filename, filepath, summary_s, col_info_s, data_blob),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
        return

    conn = _get_sqlite_conn()
    conn.execute(
        """
        INSERT INTO datasets (id, user_id, filename, filepath, summary, col_info, data_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, user_id, filename, filepath, summary_s, col_info_s, data_blob),
    )
    conn.commit()
    conn.close()


def get_dataset(dataset_id):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
                row = c.fetchone()
                if not row:
                    return None
                d = _pg_row_to_dict(c, row)
                d["summary"] = json.loads(d["summary"])
                d["col_info"] = json.loads(d["col_info"])
                return d
        finally:
            pool.putconn(conn)

    conn = _get_sqlite_conn()
    ds = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if ds:
        d = dict(ds)
        d["summary"] = json.loads(d["summary"])
        d["col_info"] = json.loads(d["col_info"])
        return d
    return None


def update_dataset_col_info(dataset_id, col_info):
    col_info_s = json.dumps(col_info)
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute("UPDATE datasets SET col_info = %s WHERE id = %s", (col_info_s, dataset_id))
            conn.commit()
        finally:
            pool.putconn(conn)
        return

    conn = _get_sqlite_conn()
    conn.execute("UPDATE datasets SET col_info = ? WHERE id = ?", (col_info_s, dataset_id))
    conn.commit()
    conn.close()


def update_dataset_blob(dataset_id, data_blob):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute("UPDATE datasets SET data_blob = %s WHERE id = %s", (data_blob, dataset_id))
            conn.commit()
        finally:
            pool.putconn(conn)
        return

    conn = _get_sqlite_conn()
    conn.execute("UPDATE datasets SET data_blob = ? WHERE id = ?", (data_blob, dataset_id))
    conn.commit()
    conn.close()


def get_user_datasets(user_id):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as c:
                c.execute(
                    "SELECT id, filename, uploaded_at FROM datasets WHERE user_id = %s ORDER BY uploaded_at DESC",
                    (user_id,),
                )
                rows = c.fetchall() or []
                cols = [desc[0] for desc in c.description]
                return [{cols[i]: row[i] for i in range(len(cols))} for row in rows]
        finally:
            pool.putconn(conn)

    conn = _get_sqlite_conn()
    datasets = conn.execute(
        "SELECT id, filename, uploaded_at FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(d) for d in datasets]


def get_user_dataset_groups(user_id):
    rows = get_user_datasets(user_id)

    groups = {}
    order = []
    for item in rows:
        filename = item["filename"]
        if filename not in groups:
            groups[filename] = {
                "dataset_id": item["id"],
                "filename": filename,
                "latest_uploaded_at": item["uploaded_at"],
                "history": [],
            }
            order.append(filename)

        groups[filename]["history"].append(
            {
                "dataset_id": item["id"],
                "uploaded_at": item["uploaded_at"],
            }
        )

    result = []
    for filename in order:
        group = groups[filename]
        total_versions = len(group["history"])
        group["history"] = group["history"][:25]
        group["versions"] = total_versions
        result.append(group)

    return result
