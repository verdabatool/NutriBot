from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config.settings import VECTOR_INDEX_PATH
from src.db.recipes import get_recipes_by_ids, exclude_ingredients


@dataclass(frozen=True)
class RetrievalResult:
    recipe_ids: List[int]
    recipes: List[dict]


# --------------------------------------------------
# Load embeddings + FAISS index (module-level cache)
# --------------------------------------------------

_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

_VECTORSTORE = FAISS.load_local(
    str(VECTOR_INDEX_PATH),
    _EMBEDDINGS,
    allow_dangerous_deserialization=True,
)


# --------------------------------------------------
# Public API
# --------------------------------------------------

def retrieve_recipe_ids(query: str, k: int = 10) -> List[int]:
    """
    Semantic retrieval: returns recipe_ids only.
    """
    docs = _VECTORSTORE.similarity_search(query, k=k)
    recipe_ids = [int(d.metadata["recipe_id"]) for d in docs if "recipe_id" in d.metadata]
    return recipe_ids


def retrieve_recipes(
    query: str,
    k: int = 5,
    exclude: Optional[List[str]] = None,
    oversample: int = 3,
) -> RetrievalResult:
    """
    Retrieve recipes semantically, then enforce deterministic filters via SQL.

    Args:
        query: user query text
        k: final number of recipes to return
        exclude: ingredients to exclude (e.g., allergies)
        oversample: retrieve k*oversample candidates from FAISS before filtering

    Returns:
        RetrievalResult(recipe_ids, recipes)
    """
    k = max(1, int(k))
    oversample = max(1, int(oversample))

    # 1) FAISS candidate recall
    candidate_ids = retrieve_recipe_ids(query, k=k * oversample)
    if not candidate_ids:
        return RetrievalResult(recipe_ids=[], recipes=[])

    # 2) Fetch full rows from DB (preserves FAISS order)
    df = get_recipes_by_ids(candidate_ids)

    # 3) Enforce hard exclusions deterministically
    if exclude:
        df = exclude_ingredients(df, exclude)

    if df.empty:
        return RetrievalResult(recipe_ids=[], recipes=[])

    # 4) Limit to top-k after filtering
    df = df.head(k)
    recipes = df.to_dict(orient="records")
    recipe_ids = [int(r["recipe_id"]) for r in recipes]

    return RetrievalResult(recipe_ids=recipe_ids, recipes=recipes)
