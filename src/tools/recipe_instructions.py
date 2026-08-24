from typing import Dict, Any
from src.db.recipes import get_recipes_by_ids
from src.tools.parsing import parse_json_list
from src.tools.registry import ToolSpec, register_tool


def recipe_instructions(recipe_id: int) -> Dict[str, Any]:
    """
    Return grounded cooking instructions for a recipe.

    Instructions are stored in the ``steps_json`` column (a JSON list of
    strings); ingredients live in ``ingredients_json``.
    """

    df = get_recipes_by_ids([recipe_id])

    if df.empty:
        return {
            "instructions": None,
            "assumptions": ["Recipe not found in dataset."]
        }

    row = df.iloc[0]

    steps = parse_json_list(row.get("steps_json"))

    if not steps:
        return {
            "instructions": None,
            "assumptions": ["No instructions available in dataset."]
        }

    return {
        "recipe_id": int(row["recipe_id"]),
        "name": row.get("name"),
        "ingredients": parse_json_list(row.get("ingredients_json")),
        "instructions": steps,
        "assumptions": [],
    }


register_tool(
    ToolSpec(
        name="recipe_instructions",
        description="Fetch grounded cooking instructions for a recipe by ID.",
        callable=recipe_instructions,
        kind="presentation",
    )
)
