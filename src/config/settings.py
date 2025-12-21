from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
VECTOR_DIR = DATA_DIR / "vectorstore"

DB_PATH = DB_DIR / "recipes.db"
VECTOR_INDEX_PATH = VECTOR_DIR / "faiss_recipes_index"


