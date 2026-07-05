from typing import Optional, Tuple, List
import pandas as pd
from sqlalchemy import text

from src.db.engine import engine
from src.services.security import hash_password, verify_password, is_hashed


# ==================================================
# User Authentication
# ==================================================

def get_user_by_username(username: str) -> Optional[dict]:
    """
    Fetch a single user by username.
    """
    query = """
    SELECT *
    FROM users
    WHERE username = :username
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"username": username})
    return df.iloc[0].to_dict() if not df.empty else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    """
    Fetch a single user by ID.
    """
    query = """
    SELECT *
    FROM users
    WHERE user_id = :user_id
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"user_id": user_id})
    return df.iloc[0].to_dict() if not df.empty else None


def authenticate_user(username: str, password: str) -> Tuple[bool, str, dict]:
    """
    Authenticate a user.
    
    Returns:
        Tuple of (success, message, user_data)
    """
    if not username or not password:
        return False, "Username and password are required.", {}
    
    user = get_user_by_username(username)

    if user is None:
        return False, "Username not found.", {}

    stored = user.get("password")
    if is_hashed(stored):
        if not verify_password(password, stored):
            return False, "Incorrect password.", {}
    else:
        # Legacy plaintext password: compare directly, then upgrade to a hash.
        if str(stored) != str(password):
            return False, "Incorrect password.", {}
        _set_password_hash(username, password)

    return True, "Login successful!", {
        "user_id": user["user_id"],
        "username": user["username"],
        "preference": user.get("preference"),
        "allergies": user.get("allergies"),
        "diet_type": user.get("diet_type"),
        "health_goal": user.get("health_goal"),
    }


def register_user(
    username: str,
    password: str,
    preference: str,
    allergies: str = "",
    diet_type: str = "",
    health_goal: str = "",
) -> Tuple[bool, str]:
    """
    Register a new user with an optional profile (allergies, diet, health goal).

    Returns:
        Tuple of (success, message)
    """
    if not username or not password:
        return False, "Username and password are required."

    # Check if username already exists
    existing_user = get_user_by_username(username)
    if existing_user is not None:
        return False, "Username already exists."

    # Insert new user
    query = """
    INSERT INTO users (username, password, preference, allergies, diet_type, health_goal)
    VALUES (:username, :password, :preference, :allergies, :diet_type, :health_goal)
    """

    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                "username": username,
                "password": hash_password(password),
                "preference": preference or "",
                "allergies": allergies or "",
                "diet_type": diet_type or "",
                "health_goal": health_goal or "",
            })
            conn.commit()
        return True, "Registration successful! Please log in."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"


def _set_password_hash(username: str, plaintext: str) -> None:
    """Store the bcrypt hash of a password (used to upgrade legacy plaintext)."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE users SET password = :p WHERE username = :u"),
                {"p": hash_password(plaintext), "u": username},
            )
            conn.commit()
    except Exception:
        pass  # non-fatal: login still succeeds this session


def update_user_profile(
    username: str,
    allergies: str = None,
    diet_type: str = None,
    health_goal: str = None,
) -> Tuple[bool, str]:
    """Update profile fields that are provided (None = leave unchanged)."""
    user = get_user_by_username(username)
    if user is None:
        return False, "User not found."

    updates = {k: v for k, v in {
        "allergies": allergies,
        "diet_type": diet_type,
        "health_goal": health_goal,
    }.items() if v is not None}

    if not updates:
        return True, "Nothing to update."

    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["username"] = username
    try:
        with engine.connect() as conn:
            conn.execute(text(f"UPDATE users SET {set_clause} WHERE username = :username"), updates)
            conn.commit()
        return True, "Profile updated successfully!"
    except Exception as e:
        return False, f"Update failed: {str(e)}"


def update_user_preference(username: str, preference: str) -> Tuple[bool, str]:
    """
    Update a user's food preference.
    
    Returns:
        Tuple of (success, message)
    """
    user = get_user_by_username(username)
    if user is None:
        return False, "User not found."
    
    query = """
    UPDATE users
    SET preference = :preference
    WHERE username = :username
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                "username": username,
                "preference": preference
            })
            conn.commit()
        return True, "Preference updated successfully!"
    except Exception as e:
        return False, f"Update failed: {str(e)}"


def update_user_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """
    Update a user's password.
    
    Returns:
        Tuple of (success, message)
    """
    # First authenticate with old password
    success, message, _ = authenticate_user(username, old_password)
    if not success:
        return False, "Current password is incorrect."
    
    if not new_password:
        return False, "New password is required."
    
    query = """
    UPDATE users
    SET password = :password
    WHERE username = :username
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {
                "username": username,
                "password": hash_password(new_password)
            })
            conn.commit()
        return True, "Password updated successfully!"
    except Exception as e:
        return False, f"Update failed: {str(e)}"


def delete_user(username: str) -> Tuple[bool, str]:
    """
    Delete a user by username.
    
    Returns:
        Tuple of (success, message)
    """
    user = get_user_by_username(username)
    if user is None:
        return False, "User not found."
    
    query = """
    DELETE FROM users
    WHERE username = :username
    """
    
    try:
        with engine.connect() as conn:
            conn.execute(text(query), {"username": username})
            conn.commit()
        return True, "User deleted successfully!"
    except Exception as e:
        return False, f"Delete failed: {str(e)}"


def get_all_users() -> List[dict]:
    """
    Fetch all users (for admin purposes).
    Returns list of dicts without passwords.
    """
    query = """
    SELECT user_id, username, preference
    FROM users
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df.to_dict(orient="records")