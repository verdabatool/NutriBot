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

# FDA reference Daily Values that the Food.com dataset's %DV are based on
# (verified by reconstructing each recipe's calories via the Atwater factors
# 9/4/4 from the macro %DV — matches within a few percent). Grams below are
# therefore reliable ESTIMATES derived from the stored %DV.
#   macro key -> (pdv column, daily value, output key, unit)
DAILY_VALUES = [
    ("protein_pdv",        50.0,   "protein_g",       "g"),
    ("carbs_pdv",          300.0,  "carbs_g",         "g"),
    ("total_fat_pdv",      65.0,   "total_fat_g",     "g"),
    ("saturated_fat_pdv",  20.0,   "saturated_fat_g", "g"),
    ("sugar_pdv",          50.0,   "sugar_g",         "g"),
    ("sodium_pdv",         2400.0, "sodium_mg",       "mg"),
]


def _macros_from(getter) -> dict:
    """Build a grams/mg macro dict from a %DV accessor (row.get or dict.get)."""
    out = {}
    for pdv_col, dv, out_key, _unit in DAILY_VALUES:
        pdv = getter(pdv_col)
        out[out_key] = round((float(pdv) / 100.0) * dv, 1) if pdv is not None else 0.0
    return out


def _empty(recipe_ids, note):
    return {
        "type": "nutrition_summary",
        "recipe_ids": recipe_ids,
        "per_recipe": {},
        "meal_total": {},
        "assumptions": [note],
    }


def nutrition_analyzer(recipe_ids: List[int], **kwargs) -> dict:
    """
    Calorie + macro breakdown (grams) for one or more recipes.

    - Per-recipe: calories and protein/carbs/fat/sat-fat/sugar/sodium in grams.
    - meal_total: the sum across all recipe_ids (a "meal").
    Grams are estimated from the dataset's stored %DV using FDA daily values.
    """
    if not recipe_ids:
        return _empty([], "No recipe IDs were provided.")

    try:
        df = get_recipes_by_ids(recipe_ids)
    except Exception:
        return _empty(recipe_ids, "Failed to retrieve nutrition data.")

    if df.empty:
        return _empty(recipe_ids, "No nutrition data available for the selected recipes.")

    per_recipe = {}
    for _, row in df.iterrows():
        rid = int(row["recipe_id"])
        cal = float(row["calories"]) if row.get("calories") is not None else 0.0
        macros = _macros_from(row.get)
        per_recipe[rid] = {
            "name": row.get("name"),
            "calories": round(cal, 1),
            **macros,
            "pdv": {
                c: float(row[c]) for c in
                ["total_fat_pdv", "saturated_fat_pdv", "carbs_pdv",
                 "sugar_pdv", "protein_pdv", "sodium_pdv"]
                if c in df.columns and row.get(c) is not None
            },
        }

    # Meal total = sum across recipes.
    macro_keys = [k for _, _, k, _ in DAILY_VALUES]
    meal_total = {"calories": round(sum(r["calories"] for r in per_recipe.values()), 1)}
    for k in macro_keys:
        meal_total[k] = round(sum(r.get(k, 0.0) for r in per_recipe.values()), 1)

    assumptions = [
        "Values are per serving, taken from the dataset (calories) and estimated "
        "in grams from the dataset's % Daily Value (FDA DVs: protein 50g, carbs "
        "300g, fat 65g, sat-fat 20g, sugar 50g, sodium 2400mg).",
    ]
    if len(per_recipe) > 1:
        assumptions.append(
            "meal_total sums one serving of each recipe in the meal."
        )

    return {
        "type": "nutrition_summary",
        "recipe_ids": recipe_ids,
        "per_recipe": per_recipe,
        "meal_total": meal_total,
        "assumptions": assumptions,
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
        description=(
            "Get the calorie and macro breakdown (protein/carbs/fat in grams) for "
            "one or more recipe_ids. Returns per-recipe values and a meal_total "
            "(the sum across all recipes, i.e. the whole meal)."
        ),
        callable=nutrition_analyzer,
        kind="calculation",
    )
)
