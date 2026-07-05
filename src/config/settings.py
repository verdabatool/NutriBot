from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# NOTE: the folder on disk is "Data" (capital D). This must match exactly so
# the app also works on case-sensitive filesystems (e.g. Linux deployments);
# otherwise SQLite silently creates an empty DB and the FAISS index fails to load.
DATA_DIR = BASE_DIR / "Data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
VECTOR_DIR = DATA_DIR / "vectorstore"

DB_PATH = DB_DIR / "recipes.db"
VECTOR_INDEX_PATH = VECTOR_DIR / "faiss_recipes_index"


