from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.db.engine import engine
from src.config.settings import VECTOR_INDEX_PATH

# --------------------------------------------------
# Load documents from DB
# --------------------------------------------------
def load_recipe_documents():
    print("📖 Loading recipe documents from DB")

    df = pd.read_sql(
        "SELECT recipe_id, document FROM recipes",
        engine
    )

    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["document"],
                metadata={"recipe_id": int(row["recipe_id"])}
            )
        )
    return docs

# --------------------------------------------------
# Build FAISS vectorstore
# --------------------------------------------------
def build_vectorstore():
    print("🚀 Starting FAISS index build")

    docs = load_recipe_documents()
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🧠 Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True
        }
    )

    print("📐 Embedding documents")
    vectors = []
    batch_size = 64

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        vectors.extend(batch_vectors)

    vectors = np.array(vectors, dtype="float32")

    print("📦 Building FAISS index")

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas
    )

    VECTOR_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving index to {VECTOR_INDEX_PATH}")
    vectorstore.save_local(str(VECTOR_INDEX_PATH))

    print(f"🎉 Indexed {len(texts)} recipes")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    build_vectorstore()
