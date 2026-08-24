# src/tools/shopping_list.py

from typing import List, Dict, Optional
from collections import Counter
import json

from src.db.recipes import get_recipes_by_ids
from src.tools.registry import ToolSpec, register_tool


def _empty(note: str) -> dict:
    return {"type": "shopping_list", "recipe_count": 0, "item_count": 0,
            "items": [], "assumptions": [note]}


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def shopping_list(
    recipe_ids: Optional[List[int]] = None,
    days: Optional[List[Dict]] = None,
    **kwargs,
) -> dict:
    """
    Generate a consolidated shopping list from a meal plan or a set of recipes.

    Accepts either recipe_ids directly, or a meal_planner 'days' list (each
    entry carrying a recipe_id). Ingredients are de-duplicated across all
    recipes; each item notes how many recipes use it.
    """
    # Resolve the recipe ids from either input.
    ids: List[int] = []
    if recipe_ids:
        ids = [i for i in recipe_ids if i is not None]
    elif days:
        ids = [
            d.get("recipe_id")
            for d in days
            if isinstance(d, dict) and d.get("recipe_id") is not None
        ]

    if not ids:
        return _empty("No recipes were provided (pass recipe_ids or a meal plan).")

    # De-dupe ids but remember the full set for the recipe count.
    unique_ids = list(dict.fromkeys(ids))

    try:
        df = get_recipes_by_ids(unique_ids)
    except Exception:
        return _empty("Failed to retrieve recipes for the shopping list.")

    if df.empty:
        return _empty("No recipes found for the provided ids.")

    # Count how many recipes use each ingredient (de-duped within a recipe).
    counts: Counter = Counter()
    for _, row in df.iterrows():
        try:
            ingredients = json.loads(row.ingredients_json)
        except Exception:
            continue
        seen = {
            ing.strip().lower()
            for ing in ingredients
            if isinstance(ing, str) and ing.strip()
        }
        counts.update(seen)

    if not counts:
        return _empty("The selected recipes had no usable ingredient lists.")

    items = sorted(counts.keys())
    items_by_recipe_count = {ing: counts[ing] for ing in items}

    # Carry the SOURCE recipes (id + name) so the answer can cite the real
    # database recipes the list was built from — otherwise the list has no
    # verifiable database reference.
    source_recipes = [
        {"recipe_id": int(row["recipe_id"]), "name": row.get("name")}
        for _, row in df.iterrows()
    ]

    return {
        "type": "shopping_list",
        "recipe_count": int(df.shape[0]),
        "source_recipes": source_recipes,
        "item_count": len(items),
        "items": items,
        "items_by_recipe_count": items_by_recipe_count,
        "assumptions": [
            "Ingredients were consolidated (de-duplicated) across all recipes.",
        ],
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

register_tool(
    ToolSpec(
        name="shopping_list",
        description=(
            "Generate a consolidated shopping list. Pass recipe_ids=[...] (e.g. the "
            "ids from a meal plan) or a meal_planner 'days' list. Ingredients are "
            "de-duplicated across all recipes."
        ),
        callable=shopping_list,
        kind="aggregation",
    )
)

