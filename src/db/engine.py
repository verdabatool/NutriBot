from sqlalchemy import create_engine
from src.config.settings import DB_PATH


# --------------------------------------------------
# Database engine (single source of truth)
# --------------------------------------------------

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    echo=False
)
