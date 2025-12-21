# src/tools/meal_planner.py

from typing import List, Optional
from pydantic import BaseModel, Field

from src.db.recipes import get_recipes_by_ids
from src.tools.registry import ToolSpec, register_tool


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class MealPlannerInput(BaseModel):
    days: int = Field(..., ge=1, le=14)
    candidate_recipe_ids: Optional[List[int]] = None
    calorie_target: Optional[int] = None
    diet_type: Optional[str] = None


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def meal_planner(
    days: int,
    candidate_recipe_ids: Optional[List[int]] = None,
    calorie_target: Optional[int] = None,
    diet_type: Optional[str] = None,
) -> dict:
    """
    Build a simple multi-day meal plan.

    This tool is defensive:
    - Missing candidate_recipe_ids → empty plan
    - calorie_target / diet_type are accepted but not enforced
    """

    # ----------------------------
    # Guard: missing candidates
    # ----------------------------
    if not candidate_recipe_ids:
        return {
            "type": "meal_plan",
            "days": [],
            "assumptions": [
                "No candidate recipes were provided, so a meal plan could not be generated."
            ],
        }

    df = get_recipes_by_ids(candidate_recipe_ids)

    if df.empty:
        return {
            "type": "meal_plan",
            "days": [],
            "assumptions": [
                "Candidate recipe IDs did not match any recipes in the database."
            ],
        }

    # ----------------------------
    # Build plan (round-robin)
    # ----------------------------
    plan = []
    idx = 0

    for day in range(1, days + 1):
        recipe = df.iloc[idx % len(df)]
        plan.append({
            "day": day,
            "recipe_id": int(recipe.recipe_id),
            "name": recipe.name,
        })
        idx += 1

    assumptions = []

    if calorie_target is not None:
        assumptions.append(
            f"Calorie target of approximately {calorie_target} kcal/day was noted but not strictly enforced."
        )

    if diet_type:
        assumptions.append(
            f"Diet preference '{diet_type}' was noted but not strictly enforced."
        )

    return {
        "type": "meal_plan",
        "days": plan,
        "assumptions": assumptions,
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

# register_tool(
#     ToolSpec(
#         name="meal_planner",
#         description="Create a multi-day meal plan from candidate recipes.",
#         callable=meal_planner,
#     )
# )
register_tool(
    ToolSpec(
        name="meal_planner",
        description="Create a multi-day meal plan from candidate recipes.",
        callable=meal_planner,
        kind="planning",
    )
)
