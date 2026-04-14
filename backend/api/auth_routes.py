import os
import json
import hmac
import hashlib
import sqlite3  # kept for reference, though we use psycopg2 underneath
from typing import Optional, Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.core.config import APP_SECRET, SESSION_FILE
from backend.db import get_conn
from backend.utils.utils import utc_now_iso

router = APIRouter(prefix="/auth", tags=["auth"])


# ------------------------
# Password hashing helpers
# ------------------------
def hash_password(password: str, salt: Optional[bytes] = None):
    """
    Hash a password using PBKDF2-HMAC (SHA-256, 150k iterations) with random salt.
    Returns (pwd_hex, salt_hex).
    """
    if salt is None:
        salt = os.urandom(16)
    ph = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000)
    return ph.hex(), salt.hex()


def verify_password(password: str, pwd_hex: str, salt_hex: str) -> bool:
    """
    Verify a password against existing hash + salt (hex-encoded).
    """
    nh = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 150_000
    ).hex()
    return hmac.compare_digest(nh, pwd_hex)


# ------------------------
# DB helpers
# ------------------------
def create_user(username: str, email: str, password: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        ph, sh = hash_password(password)

        cur.execute(
            """
            INSERT INTO users (username, email, pwd_hash, salt, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (username, email, ph, sh, utc_now_iso()),
        )
        new_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()
        return True, f"Account created with ID {new_id}."
    except Exception as e:
        # Catch duplicates (IntegrityError in psycopg2 -> "duplicate key" / "unique constraint")
        err_msg = str(e).lower()
        if "unique constraint" in err_msg or "duplicate key" in err_msg:
            return False, "Username or Email already exists."
        return False, f"Error creating user: {e}"


def get_user_by_login(identifier: str):
    """
    Fetch a user by either username OR email.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=%s OR email=%s LIMIT 1",
        (identifier, identifier),
    )
    row = cur.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


# ------------------------
# Simple file-based session (optional)
# ------------------------
def save_persistent_session(user_id: int, pwd_hex: str):
    """
    Save a simple persistent session token in a JSON file.
    """
    token = hashlib.sha256(f"{user_id}:{pwd_hex}:{APP_SECRET}".encode()).hexdigest()
    with open(SESSION_FILE, "w") as f:
        json.dump({"user_id": user_id, "token": token}, f)


def load_persistent_session():
    """
    Load and validate the persistent session from file.
    Returns user_id if valid, else None.
    """
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        user_id = data.get("user_id")
        token = data.get("token")

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, pwd_hash FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        expected = hashlib.sha256(
            f"{row['id']}:{row['pwd_hash']}:{APP_SECRET}".encode()
        ).hexdigest()
        if hmac.compare_digest(token, expected):
            return row["id"]
        return None
    except Exception:
        return None


def clear_persistent_session():
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
    except Exception:
        pass


# ------------------------
# Pydantic models
# ------------------------
class PasswordHashRequest(BaseModel):
    password: str


class VerifyPasswordRequest(BaseModel):
    password: str
    pwd_hex: str
    salt_hex: str


class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    identifier: str  # username or email
    password: str


class SaveSessionRequest(BaseModel):
    user_id: int
    pwd_hex: str


# ------------------------
# Helper
# ------------------------
def _row_to_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    try:
        return dict(row)
    except Exception:
        return {"value": row}


# ------------------------
# FastAPI routes
# ------------------------
@router.post("/hash-password")
def api_hash_password(payload: PasswordHashRequest):
    pwd_hash, salt = hash_password(payload.password)
    return {"pwd_hash": pwd_hash, "salt": salt}


@router.post("/verify-password")
def api_verify_password(payload: VerifyPasswordRequest):
    valid = verify_password(payload.password, payload.pwd_hex, payload.salt_hex)
    return {"valid": valid}


@router.post("/users", status_code=status.HTTP_201_CREATED)
def api_create_user(payload: CreateUserRequest):
    ok, msg = create_user(payload.username, payload.email, payload.password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"success": True, "message": msg}


@router.post("/login")
def api_login(payload: UserLoginRequest):
    """
    Login with username OR email + password.
    Returns a SAFE user object (no password hash / salt).
    """
    user = get_user_by_login(payload.identifier)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )

    pwd_hash = user["pwd_hash"]
    salt = user["salt"]

    if not verify_password(payload.password, pwd_hash, salt):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )

    # SAFE USER OBJECT (NO PASSWORD DATA)
    safe_user = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"],
    }

    return {"user": safe_user}


@router.get("/users/by-login")
def api_get_user_by_login(identifier: str):
    row = get_user_by_login(identifier)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return _row_to_dict(row)


@router.get("/users/{user_id}")
def api_get_user_by_id(user_id: int):
    row = get_user_by_id(user_id)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return _row_to_dict(row)


@router.post("/session/save")
def api_save_persistent_session(payload: SaveSessionRequest):
    save_persistent_session(payload.user_id, payload.pwd_hex)
    return {"success": True}


@router.get("/session/load")
def api_load_persistent_session():
    user_id = load_persistent_session()
    return {"user_id": user_id}


@router.delete("/session/clear")
def api_clear_persistent_session():
    clear_persistent_session()
    return {"success": True}