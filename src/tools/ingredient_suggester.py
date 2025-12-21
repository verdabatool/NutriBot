from typing import List, Optional
from pydantic import BaseModel, Field

from src.db.recipes import (
    get_recipes_with_all_ingredients,
    get_recipes_with_any_ingredients,
    get_recipes_by_ids,  # IMPORTANT: re-ground semantic results
)
from src.retrieval.recipe_retriever import retrieve_recipes
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class IngredientSuggesterInput(BaseModel):
    ingredients: Optional[List[str]] = Field(
        default=None,
        description="Ingredients the user has available"
    )
    k: int = Field(default=5, ge=1, le=20)
    semantic_rerank: bool = True


# --------------------------------------------------
# Internal helpers
# --------------------------------------------------

# Explicit allow-list of fields that are SAFE to expose
ALLOWED_RECIPE_FIELDS = {
    "recipe_id",
    "name",
    "ingredients_json",
    "instructions",
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbs_pdv",
}


def _sanitize_records(records: List[dict]) -> List[dict]:
    """Strip any non-dataset fields to prevent hallucinations."""
    return [
        {k: v for k, v in r.items() if k in ALLOWED_RECIPE_FIELDS}
        for r in records
        if "recipe_id" in r
    ]


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def ingredient_suggester(
    ingredients: Optional[List[str]] = None,
    k: int = 5,
    semantic_rerank: bool = True,
    **kwargs,
) -> dict:
    """
    Suggest recipes based on ingredients the user has.

    STRICT GUARANTEES:
    - Returns ONLY dataset-backed recipes
    - Returns ONLY dataset-backed fields
    - Never infers ingredients, proteins, or cooking times
    """

    # ----------------------------
    # 0) No ingredients provided
    # ----------------------------
    if not ingredients:
        return {
            "recipe_ids": [],
            "recipes": [],
            "assumptions": [
                "No ingredients were provided, so no ingredient-based filtering was applied."
            ],
            "match_mode": "none",
            "source": "dataset",
        }

    assumptions: List[str] = []

    # ----------------------------
    # 1) STRICT match
    # ----------------------------
    df = get_recipes_with_all_ingredients(ingredients)
    match_mode = "strict"

    # ----------------------------
    # 2) RELAXED match
    # ----------------------------
    if df.empty:
        df = get_recipes_with_any_ingredients(
            ingredients=ingredients,
            min_matches=max(1, len(ingredients) - 1),
        )
        match_mode = "relaxed"

        if not df.empty:
            assumptions.append(
                "Recipes match most, but not all, provided ingredients."
            )
        else:
            return {
                "recipe_ids": [],
                "recipes": [],
                "assumptions": [
                    "No recipes matched the provided ingredients, even with relaxed matching."
                ],
                "match_mode": "relaxed",
                "source": "dataset",
            }

    # ----------------------------
    # 3) Semantic reranking (SAFE)
    # ----------------------------
    candidate_ids = df["recipe_id"].tolist()

    if semantic_rerank and candidate_ids:
        semantic = retrieve_recipes(
            query=" ".join(ingredients),
            k=min(len(candidate_ids), k * 3),
        )

        ranked_ids = [
            r["recipe_id"]
            for r in semantic.recipes
            if r.get("recipe_id") in candidate_ids
        ][:k]

        # 🔒 RE-GROUND through DB
        df = get_recipes_by_ids(ranked_ids)

    else:
        df = df.head(k)

    # ----------------------------
    # 4) Final sanitize
    # ----------------------------
    records = _sanitize_records(df.to_dict(orient="records"))

    return {
        "recipe_ids": [r["recipe_id"] for r in records],
        "recipes": records,
        "assumptions": assumptions,
        "match_mode": match_mode,
        "source": "dataset",
    }


# --------------------------------------------------
# Tool registration
# --------------------------------------------------

register_tool(
    ToolSpec(
        name="ingredient_suggester",
        description="Suggest recipes based on available ingredients. Dataset-grounded only.",
        callable=ingredient_suggester,
        kind="retrieval",
    )
)
