import os
import psycopg2


def get_connection():
    """connect to postgres using DATABASE_URL from env"""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(database_url)


def init_db():
    """create tables if they don't exist"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key_hash VARCHAR(64) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    api_key_id UUID REFERENCES api_keys(id),
                    endpoint VARCHAR(50),
                    n_vectors INTEGER,
                    dim INTEGER,
                    bit_width INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
    finally:
        conn.close()
