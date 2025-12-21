# src/tools/nutrition.py

from typing import List
from pydantic import BaseModel, Field

from src.db.recipes import get_recipes_by_ids
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class NutritionAnalyzerInput(BaseModel):
    recipe_ids: List[int] = Field(..., min_items=1)


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def nutrition_analyzer(recipe_ids: List[int], **kwargs) -> dict:


    """
    Aggregate nutrition information for recipes.

    Defensive behavior:
    - Empty / invalid IDs → empty summary
    - DB failures → empty summary
    """

    if not recipe_ids:
        return {
            "type": "nutrition_summary",
            "recipe_ids": [],
            "per_recipe": {},
            "totals": {},
            "assumptions": ["No recipe IDs were provided."],
        }

    try:
        df = get_recipes_by_ids(recipe_ids)
    except Exception:
        return {
            "type": "nutrition_summary",
            "recipe_ids": recipe_ids,
            "per_recipe": {},
            "totals": {},
            "assumptions": ["Failed to retrieve nutrition data."],
        }

    if df.empty:
        return {
            "type": "nutrition_summary",
            "recipe_ids": recipe_ids,
            "per_recipe": {},
            "totals": {},
            "assumptions": ["No nutrition data available for the selected recipes."],
        }

    nutrition_cols = [
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
    ]

    # Sum totals safely
    totals = {
        col: float(df[col].fillna(0).sum())
        for col in nutrition_cols
        if col in df.columns
    }

    # Per-recipe breakdown
    per_recipe = {}
    for _, row in df.iterrows():
        rid = int(row["recipe_id"])
        per_recipe[rid] = {
            col: float(row[col]) if col in df.columns and row[col] is not None else 0.0
            for col in nutrition_cols
        }

    return {
        "type": "nutrition_summary",
        "recipe_ids": recipe_ids,
        "per_recipe": per_recipe,
        "totals": totals,
        "assumptions": [
            "Nutrition values are taken directly from the dataset.",
            "PDV values are summed without conversion to grams.",
        ],
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

# register_tool(
#     ToolSpec(
#         name="nutrition_analyzer",
#         description="Compute nutrition totals for recipes.",
#         callable=nutrition_analyzer,
#     )
# )
register_tool(
    ToolSpec(
        name="nutrition_analyzer",
        description="Compute nutrition totals for recipes.",
        callable=nutrition_analyzer,
        kind="calculation",
    )
)
