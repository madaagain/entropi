import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# charge le .env automatiquement depuis la racine du repo
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path, override=False)


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
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    plan VARCHAR(50) DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key_hash VARCHAR(64) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    user_id UUID REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            # add user_id column if table already exists without it
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'api_keys' AND column_name = 'user_id'
                    ) THEN
                        ALTER TABLE api_keys ADD COLUMN user_id UUID REFERENCES users(id);
                    END IF;
                END $$;
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
