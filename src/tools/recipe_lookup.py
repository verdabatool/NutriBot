from typing import Optional, List
from pydantic import BaseModel, Field

from src.retrieval.recipe_retriever import retrieve_recipes
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class RecipeLookupInput(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def recipe_lookup(query: str, k: int = 5, exclude: Optional[List[str]] = None, **kwargs) -> dict:
    """
    Semantic recipe lookup.

    Args:
        query: what the user wants
        k: number of recipes to return
        exclude: ingredients to exclude (e.g. the user's allergens)

    Defensive behavior:
    - Empty / bad queries → empty result
    - Retrieval failures → empty result
    """

    if not query or not query.strip():
        return {
            "recipe_ids": [],
            "recipes": [],
        }

    try:
        result = retrieve_recipes(query=query, k=k, exclude=exclude)
    except Exception:
        # NEVER crash the execution loop
        return {
            "recipe_ids": [],
            "recipes": [],
        }

    return {
        "recipe_ids": result.recipe_ids or [],
        "recipes": result.recipes or [],
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

# register_tool(
#     ToolSpec(
#         name="recipe_lookup",
#         description="Find recipes using semantic search."
#                     "REQUIRED for any recipe suggestions. "
#                     "Returns the ONLY valid recipe names the assistant may mention.",
#         callable=recipe_lookup,
#     )
# )

register_tool(
    ToolSpec(
        name="recipe_lookup",
        description="Find recipes using semantic search. REQUIRED for any recipe suggestions. Returns the ONLY valid recipe names the assistant may mention. Pass exclude=[...] to filter out the user's allergens.",
        callable=recipe_lookup,
        kind="retrieval",
    )
)
