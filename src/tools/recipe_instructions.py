from typing import Dict, Any
from src.db.recipes import get_recipes_by_ids
from src.tools.registry import ToolSpec, register_tool


def recipe_instructions(recipe_id: int) -> Dict[str, Any]:
    """
    Return grounded cooking instructions for a recipe.
    """

    df = get_recipes_by_ids([recipe_id])

    if df.empty:
        return {
            "instructions": None,
            "assumptions": ["Recipe not found in dataset."]
        }

    # Be explicit and defensive
    row = df.iloc[0]

    if "instructions" not in row or not row["instructions"]:
        return {
            "instructions": None,
            "assumptions": ["No instructions available in dataset."]
        }

    # return {
    #     "instructions": row["instructions"],
    #     "assumptions": []
    # }
    return {
        "instructions": None,
        "assumptions": ["No instructions available in dataset."],
        "status": "failure"
    }


# register_tool(
#     ToolSpec(
#         name="recipe_instructions",
#         description="Fetch grounded cooking instructions for a recipe by ID.",
#         callable=recipe_instructions,
#     )
# )
register_tool(
    ToolSpec(
        name="recipe_instructions",
        description="Fetch grounded cooking instructions for a recipe by ID.",
        callable=recipe_instructions,
        kind="presentation",
    )
)
