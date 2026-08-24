from typing import Dict, Any
from src.db.engine import engine
import pandas as pd

from src.tools.registry import ToolSpec, register_tool


def resolve_recipe_by_name(name: str) -> Dict[str, Any]:
    """
    Resolve a recipe name to a recipe_id using exact or fuzzy match.
    """

    if not name:
        return {
            "recipe_id": None,
            "matches": [],
            "assumptions": ["No recipe name provided."]
        }

    query = """
    SELECT recipe_id, name
    FROM recipes
    WHERE LOWER(name) LIKE ?
    LIMIT 5
    """

    df = pd.read_sql(
        query,
        engine,
        params=(f"%{name.lower()}%",)
    )

    if df.empty:
        return {
            "recipe_id": None,
            "matches": [],
            "assumptions": ["No matching recipe names found."]
        }

    # Choose the best match deterministically (first row)
    best = df.iloc[0]

    return {
        "recipe_id": int(best["recipe_id"]),
        "resolved_name": best["name"],
        "matches": df.to_dict(orient="records"),
        "assumptions": [
            "Recipe was resolved by fuzzy name matching."
        ]
    }


register_tool(
    ToolSpec(
        name="resolve_recipe_by_name",
        description=(
            "Look up a recipe_id ONLY when the USER names a specific dish and you "
            "don't already have its ID. Do NOT use this on recipes that recipe_lookup "
            "or ingredient_suggester just returned — those already include each "
            "recipe's (ID: N), so calling this afterward is redundant. Never call it "
            "more than once for the same name."
        ),
        callable=resolve_recipe_by_name,
        kind="resolver",
    )
)
