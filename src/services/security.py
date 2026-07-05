# src/services/security.py
"""Password hashing helpers (bcrypt)."""

import bcrypt

# bcrypt only uses the first 72 bytes; truncate to avoid a ValueError on long input.
_MAX = 72


def hash_password(password: str) -> str:
    pw = (password or "").encode("utf-8")[:_MAX]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(
            (password or "").encode("utf-8")[:_MAX],
            (hashed or "").encode("utf-8"),
        )
    except Exception:
        return False


def is_hashed(value) -> bool:
    """True if a stored password already looks like a bcrypt hash."""
    return isinstance(value, str) and value.startswith(("$2a$", "$2b$", "$2y$")) and len(value) >= 55