from sqlalchemy import create_engine, text

from src.config.settings import DB_PATH


# --------------------------------------------------
# Database engine (single source of truth)
# --------------------------------------------------

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    echo=False,
    # Wait for the write lock instead of failing immediately with
    # "database is locked" when another connection is writing.
    connect_args={"timeout": 30},
)


# --------------------------------------------------
# Initialize users table from CSV
# --------------------------------------------------

def init_users_table():
    """
    Ensure the users table exists and is migrated. Users are created by
    registration and stored in the database (the single source of truth).
    """
    with engine.connect() as conn:
        # Create users table if not exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                preference TEXT
            )
        """))
        conn.commit()

        # Idempotent migration: add profile columns if an older table is missing them.
        existing_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(users)"))}
        for col in ("allergies", "diet_type", "health_goal"):
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} TEXT"))
        conn.commit()

        # Security: hash any plaintext passwords at rest (idempotent).
        from src.services.security import hash_password, is_hashed
        migrated = 0
        for uname, pw in conn.execute(text("SELECT username, password FROM users")).fetchall():
            if not is_hashed(pw):
                conn.execute(text("UPDATE users SET password = :p WHERE username = :u"),
                             {"p": hash_password(pw or ""), "u": uname})
                migrated += 1
        if migrated:
            conn.commit()
            print(f"🔒 Hashed {migrated} plaintext password(s).")


# Initialize users table on module import
init_users_table()