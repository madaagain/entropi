import bcrypt

from entropi.db.database import get_connection


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_user(email: str, password: str) -> dict:
    """create a new user, returns user dict (without password_hash)"""
    pw_hash = hash_password(password)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (%s, %s)
                RETURNING id, email, plan, created_at, is_active
                """,
                (email, pw_hash),
            )
            row = cur.fetchone()
            conn.commit()
            return {
                "id": str(row[0]),
                "email": row[1],
                "plan": row[2],
                "created_at": row[3].isoformat(),
                "is_active": row[4],
            }
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    """returns user dict with password_hash, or None"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash, plan, created_at, is_active FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "email": row[1],
                "password_hash": row[2],
                "plan": row[3],
                "created_at": row[4].isoformat(),
                "is_active": row[5],
            }
    finally:
        conn.close()


def get_user_by_id(user_id: str) -> dict | None:
    """returns user dict without password_hash, or None"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, plan, created_at, is_active FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "email": row[1],
                "plan": row[2],
                "created_at": row[3].isoformat(),
                "is_active": row[4],
            }
    finally:
        conn.close()
