from typing import List, Optional

from src.db.recipes import (
    get_recipes_with_any_ingredients,
    exclude_ingredients,  # allergen exclusion
)
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Internal helpers
# --------------------------------------------------

# Explicit allow-list of fields that are SAFE to expose
ALLOWED_RECIPE_FIELDS = {
    "recipe_id",
    "name",
    "description",
    "ingredients_json",
    "instructions",
    "match_count",
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
    exclude: Optional[List[str]] = None,
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

    ingredients = [i.strip() for i in ingredients if i and i.strip()]
    if not ingredients:
        return {
            "recipe_ids": [], "recipes": [],
            "assumptions": ["No usable ingredients were provided."],
            "match_mode": "none", "source": "dataset",
        }

    assumptions: List[str] = []
    n = len(ingredients)

    # ----------------------------
    # 1) Match by how many of the user's ingredients each recipe uses.
    #    Require all n, then relax (n-1, n-2, ... 1). Results are already
    #    ranked by match_count DESC, so recipes using MORE of the fridge
    #    ingredients come first — exactly what "what's in my fridge" wants.
    # ----------------------------
    df = None
    matched = 0
    for min_matches in range(n, 0, -1):
        df = get_recipes_with_any_ingredients(ingredients, min_matches=min_matches)
        if not df.empty:
            matched = min_matches
            break

    if df is None or df.empty:
        return {
            "recipe_ids": [], "recipes": [],
            "assumptions": ["No recipes used any of the provided ingredients."],
            "match_mode": "none", "source": "dataset",
        }

    match_mode = "all" if matched == n else f"{matched}-of-{n}"
    if matched < n:
        assumptions.append(
            f"No recipe used all {n} ingredients; showing recipes that use "
            f"{matched} of them (ranked by how many they use)."
        )

    # ----------------------------
    # 2) Exclude allergens (deterministic, DB-backed)
    # ----------------------------
    if exclude:
        df = exclude_ingredients(df, exclude)
        if df.empty:
            return {
                "recipe_ids": [], "recipes": [],
                "assumptions": [
                    "All matching recipes were removed because they contained an "
                    "excluded ingredient (e.g. an allergen)."
                ],
                "match_mode": match_mode, "source": "dataset",
            }

    # ----------------------------
    # 3) Take the top-k (already ranked by match_count DESC) and sanitize.
    # ----------------------------
    df = df.head(k)
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
        description=(
            "Suggest recipes for the ingredients a user already has (the "
            "'what's in my fridge' problem). Pass ingredients=[...]; results are "
            "ranked by how many of those ingredients each recipe uses. Pass "
            "exclude=[...] for allergens. Dataset-grounded only."
        ),
        callable=ingredient_suggester,
        kind="retrieval",
    )
)
