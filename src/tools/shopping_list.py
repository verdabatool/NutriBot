# src/tools/shopping_list.py

from typing import List, Dict, Set
from pydantic import BaseModel, Field
import json

from src.db.recipes import get_recipes_by_ids
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class ShoppingListInput(BaseModel):
    days: List[Dict] = Field(..., description="Meal plan days output")


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def shopping_list(days: List[Dict], **kwargs) -> dict:

    """
    Generate a consolidated shopping list from a meal plan.

    Defensive behavior:
    - Empty days → empty list
    - Missing recipe IDs → skipped
    - Bad ingredient JSON → skipped
    """

    if not days:
        return {
            "type": "shopping_list",
            "items": [],
            "assumptions": ["No meal plan days were provided."],
        }

    recipe_ids = [
        d.get("recipe_id")
        for d in days
        if isinstance(d, dict) and "recipe_id" in d
    ]

    if not recipe_ids:
        return {
            "type": "shopping_list",
            "items": [],
            "assumptions": ["Meal plan contained no valid recipe IDs."],
        }

    try:
        df = get_recipes_by_ids(recipe_ids)
    except Exception:
        return {
            "type": "shopping_list",
            "items": [],
            "assumptions": ["Failed to retrieve recipes for shopping list."],
        }

    if df.empty:
        return {
            "type": "shopping_list",
            "items": [],
            "assumptions": ["No recipes found for the meal plan."],
        }

    items: Set[str] = set()

    for _, row in df.iterrows():
        try:
            ingredients = json.loads(row.ingredients_json)
            for ing in ingredients:
                if isinstance(ing, str):
                    items.add(ing.strip().lower())
        except Exception:
            # Skip malformed ingredient lists
            continue

    return {
        "type": "shopping_list",
        "items": sorted(items),
        "assumptions": [
            "Ingredients were aggregated from the selected meal plan recipes.",
            "Ingredient quantities are not included.",
        ],
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

# register_tool(
#     ToolSpec(
#         name="shopping_list",
#         description="Generate a shopping list from a meal plan.",
#         callable=shopping_list,
#     )
# )
register_tool(
    ToolSpec(
        name="shopping_list",
        description="Generate a shopping list from a meal plan.",
        callable=shopping_list,
        kind="aggregation",
    )
)

