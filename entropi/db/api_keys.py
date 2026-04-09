import hashlib
import secrets
import threading

from entropi.db.database import get_connection


def hash_key(api_key: str) -> str:
    """sha256 hash of the api key"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def validate_api_key(api_key: str) -> bool:
    """check if the key exists in db and is active"""
    key_hash = hash_key(api_key)
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM api_keys WHERE key_hash = %s AND is_active = TRUE",
                    (key_hash,),
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    except Exception:
        return False


def create_api_key(name: str) -> str:
    """generate a new api key, store hash in db, return plaintext key once"""
    raw = secrets.token_hex(16)
    api_key = f"ent_live_{raw}"
    key_hash = hash_key(api_key)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, name) VALUES (%s, %s)",
                (key_hash, name),
            )
            conn.commit()
    finally:
        conn.close()

    return api_key


def log_usage(api_key: str, endpoint: str, n_vectors: int, dim: int, bit_width: int):
    """log api call in background so it doesn't slow down the response"""
    def _log():
        key_hash = hash_key(api_key)
        try:
            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO usage_logs (api_key_id, endpoint, n_vectors, dim, bit_width)
                        SELECT id, %s, %s, %s, %s FROM api_keys WHERE key_hash = %s
                        """,
                        (endpoint, n_vectors, dim, bit_width, key_hash),
                    )
                    conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    threading.Thread(target=_log, daemon=True).start()
