# src/tools/recipe_modifier.py

from typing import Dict, Any, Optional

from src.db.recipes import get_recipes_by_ids
from src.tools.parsing import parse_json_list
from src.tools.registry import ToolSpec, register_tool


def modify_recipe(
    recipe_id: int,
    servings_from: Optional[int] = None,
    servings_to: Optional[int] = None,
    modification: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Fetch a grounded recipe and prepare it for modification/scaling.

    Scaling: the dataset does not store per-ingredient quantities, so this tool
    returns a ``scale_factor`` (servings_to / servings_from) that the assistant
    should apply to any quantities mentioned in the steps.

    Modification: returns the real ingredients/steps plus the requested change
    (e.g. "gluten-free", "vegan", "halve the sugar") so the assistant can
    propose substitutions grounded on the actual recipe.
    """
    df = get_recipes_by_ids([recipe_id])
    if df.empty:
        return {
            "status": "not_found",
            "recipe_id": recipe_id,
            "assumptions": ["Recipe not found in dataset."],
        }

    row = df.iloc[0]

    scale_factor: Optional[float] = None
    if servings_from and servings_to and float(servings_from) > 0:
        scale_factor = round(float(servings_to) / float(servings_from), 4)

    return {
        "recipe_id": int(row["recipe_id"]),
        "name": row.get("name"),
        "ingredients": parse_json_list(row.get("ingredients_json")),
        "instructions": parse_json_list(row.get("steps_json")),
        "servings_from": servings_from,
        "servings_to": servings_to,
        "scale_factor": scale_factor,
        "modification": modification,
        "assumptions": [
            "Ingredient quantities are not stored in the dataset; apply "
            "scale_factor to any quantities that appear in the steps.",
            "Any ingredient substitutions are suggestions, not from the dataset.",
        ],
    }


register_tool(
    ToolSpec(
        name="modify_recipe",
        description=(
            "Scale a recipe to a new serving size and/or modify it (e.g. "
            "gluten-free, vegan, less sugar) for a given recipe_id. Returns the "
            "grounded ingredients and steps plus a scale_factor to apply."
        ),
        callable=modify_recipe,
        kind="calculation",
    )
)