import hashlib
import secrets
import threading

from entropi.db.database import get_connection


def hash_key(api_key: str) -> str:
    """sha256 hash of the api key"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def validate_api_key(api_key: str) -> dict | None:
    """check if the key exists in db and is active.
    returns the key row dict if valid, None otherwise."""
    key_hash = hash_key(api_key)
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ak.id, ak.user_id, ak.name, ak.is_active
                    FROM api_keys ak
                    LEFT JOIN users u ON ak.user_id = u.id
                    WHERE ak.key_hash = %s AND ak.is_active = TRUE
                    AND (ak.user_id IS NULL OR u.is_active = TRUE)
                    """,
                    (key_hash,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": str(row[0]),
                    "user_id": str(row[1]) if row[1] else None,
                    "name": row[2],
                    "is_active": row[3],
                }
        finally:
            conn.close()
    except Exception:
        return None


def create_api_key(name: str, user_id: str = None) -> str:
    """generate a new api key, store hash in db, return plaintext key once"""
    raw = secrets.token_hex(16)
    api_key = f"ent_live_{raw}"
    key_hash = hash_key(api_key)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (key_hash, name, user_id) VALUES (%s, %s, %s)",
                (key_hash, name, user_id),
            )
            conn.commit()
    finally:
        conn.close()

    return api_key


def get_keys_by_user(user_id: str) -> list[dict]:
    """list all keys for a user"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, created_at, is_active
                FROM api_keys WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "created_at": row[2].isoformat(),
                    "is_active": row[3],
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()


def delete_key(key_id: str, user_id: str) -> bool:
    """deactivate a key, only if it belongs to the user"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_keys SET is_active = FALSE
                WHERE id = %s AND user_id = %s AND is_active = TRUE
                """,
                (key_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


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
