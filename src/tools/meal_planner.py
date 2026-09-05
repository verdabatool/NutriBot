# src/tools/meal_planner.py

from typing import List, Optional

from src.db.recipes import get_recipes_by_ids, exclude_ingredients
from src.tools.diet import (
    diet_forbidden_ingredients, diet_tags, is_restricted, is_known_diet,
)
from src.tools.registry import ToolSpec, register_tool


# Meal-type -> tag keywords used to pick a suitable recipe for each slot.
# (The dataset has no 'dinner' tag, so 'main-dish' stands in for dinner.)
MEAL_TYPE_TAGS = {
    "Breakfast": ["breakfast", "brunch"],
    "Lunch": ["lunch", "main-dish", "main-course"],
    "Dinner": ["main-dish", "main-course", "dinner"],
    "Snack": ["snacks", "appetizers", "side-dishes"],
}


def _empty(note: str) -> dict:
    return {"type": "meal_plan", "days": [], "daily_totals": [], "assumptions": [note]}


def _day_slots(meals_per_day: int) -> List[str]:
    """Meal labels for one day."""
    if meals_per_day <= 1:
        return ["Dinner"]
    if meals_per_day == 2:
        return ["Lunch", "Dinner"]
    if meals_per_day == 3:
        return ["Breakfast", "Lunch", "Dinner"]
    return ["Breakfast", "Lunch", "Dinner"] + [f"Snack {i}" for i in range(1, meals_per_day - 2)]


def _rows(df, diet_active: bool, tags: List[str]) -> List[dict]:
    """Return light recipe dicts (id, name, cal, tags), PREFERRING recipes whose
    tags positively mark them as diet-compliant.

    Note: this is only a *preference* toward tagged recipes — the hard diet
    guarantee (no forbidden ingredients) is enforced separately via
    exclude_ingredients before rows ever get here, so an untagged-but-clean recipe
    is still safe to keep."""
    if diet_active and tags and not df.empty and "tags_json" in df.columns:
        low = df["tags_json"].fillna("").str.lower()
        m = low.apply(lambda s: any(t in s for t in tags))
        if m.any():
            df = df[m]
    out = []
    for r in df.to_dict("records"):
        out.append({
            "recipe_id": int(r["recipe_id"]),
            "name": r.get("name"),
            "cal": (float(r["calories"]) if r.get("calories") is not None else None),
            "tags": str(r.get("tags_json") or "").lower(),
        })
    return out


# --------------------------------------------------
# Tool implementation
# --------------------------------------------------

def meal_planner(
    days: int,
    candidate_recipe_ids: Optional[List[int]] = None,
    calorie_target: Optional[int] = None,
    diet_type: Optional[str] = None,
    meals_per_day: int = 3,
    query: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> dict:
    """
    Build a multi-day meal plan. SELF-SUFFICIENT: finds its own recipes if no
    candidate_recipe_ids are given.

    Each day gets `meals_per_day` meals matched to the time of day — breakfast
    recipes for breakfast, lunch for lunch, dinner (main-dish) for dinner — and
    the day's meals are chosen so their calories total close to but UNDER
    calorie_target.

    Args: days, meals_per_day (default 3), calorie_target (per day), diet_type
    ('vegetarian' etc.; 'non-vegetarian'/'any' = no restriction), query, exclude
    (allergens), candidate_recipe_ids (optional).

    Returns a flat 'days' list (one entry per meal, with recipe_id + meal_label)
    plus 'daily_totals'. The 'days' list can be passed straight to shopping_list.
    """
    days = max(1, int(days))
    meals_per_day = max(1, int(meals_per_day))
    slots = _day_slots(meals_per_day)
    base_types: List[str] = []
    for s in slots:
        bt = s.split()[0]
        if bt not in base_types:
            base_types.append(bt)

    diet = (diet_type or "").strip()
    diet_active = is_restricted(diet)
    assumptions: List[str] = []

    # Deterministic diet enforcement: forbidden ingredients are excluded the same
    # way allergens are (word-boundary match against each recipe's real
    # ingredients), so e.g. a "vegan" plan can never include chicken/tuna/milk —
    # regardless of whether a recipe happened to be tagged. Combined with the
    # caller's allergen `exclude`.
    diet_bans = diet_forbidden_ingredients(diet) if diet_active else []
    tags = diet_tags(diet) if diet_active else []
    all_exclude = list(exclude or []) + diet_bans
    if diet_active and not is_known_diet(diet):
        assumptions.append(
            f"'{diet}' isn't a diet I can enforce by ingredient, so I matched on "
            f"recipe tags only — double-check each recipe fits."
        )

    try:
        from src.retrieval.recipe_retriever import retrieve_recipes
    except Exception:
        retrieve_recipes = None

    # If caller supplied recipes, use them (allergen + diet filtered); else fetch per type.
    provided_rows = None
    if candidate_recipe_ids:
        dfp = get_recipes_by_ids(candidate_recipe_ids)
        if all_exclude:
            dfp = exclude_ingredients(dfp, all_exclude)
        provided_rows = _rows(dfp, diet_active, tags)

    # ---- Build a candidate pool per meal type ----
    pools: dict = {}
    relaxed: List[str] = []
    for bt in base_types:
        kws = MEAL_TYPE_TAGS.get(bt, [])
        if provided_rows is not None:
            rows = provided_rows
        elif retrieve_recipes is not None:
            base_q = (query or "").strip()
            # Bias retrieval toward the diet by naming it in the query, and filter
            # forbidden ingredients (diet + allergens) out of the candidates.
            q = " ".join(x for x in [diet if diet_active else "", base_q, bt] if x).strip()
            ids = retrieve_recipes(query=q, k=40, exclude=all_exclude).recipe_ids
            rows = _rows(get_recipes_by_ids(ids), diet_active, tags) if ids else []
        else:
            rows = []

        typed = [r for r in rows if any(k in r["tags"] for k in kws)]
        if typed:
            pool = typed
        else:
            pool = rows
            if rows:
                relaxed.append(bt)

        # A single serving can't exceed the whole-day target.
        if calorie_target is not None:
            capped = [r for r in pool if r["cal"] is not None and r["cal"] <= float(calorie_target)]
            if capped:
                pool = capped
        pools[bt] = pool

    if all(not p for p in pools.values()):
        note = ("Could not find suitable recipes for the plan. Try a broader "
                "query or a different diet_type.")
        if diet_active:
            note = (f"Could not find enough {diet}-compliant recipes for this plan. "
                    f"Try a broader query.")
        return _empty(note)
    if diet_active:
        assumptions.append(
            f"Every recipe is {diet}-compliant — recipes containing "
            f"{diet}-incompatible ingredients were excluded."
        )
    if relaxed:
        assumptions.append(
            "Meal-type matching was relaxed for: " + ", ".join(relaxed) +
            " (not enough recipes tagged for that meal were found)."
        )

    # ---- Fill each day: right recipe for the right meal, day total under target ----
    plan: List[dict] = []
    daily_totals: List[dict] = []
    used: set = set()

    for day in range(1, days + 1):
        day_total = 0.0
        have_cals = False
        for slot_idx, label in enumerate(slots):
            bt = label.split()[0]
            pool = pools.get(bt) or next((p for p in pools.values() if p), [])
            if not pool:
                continue

            cal_pool = [r for r in pool if r["cal"] is not None]
            if calorie_target is not None and cal_pool:
                remaining_after = meals_per_day - slot_idx - 1
                budget_left = float(calorie_target) - day_total
                slot_target = budget_left / (remaining_after + 1)
                avail = [r for r in cal_pool if r["recipe_id"] not in used and r["cal"] <= budget_left]
                if not avail:                       # pool exhausted -> allow reuse
                    avail = [r for r in cal_pool if r["cal"] <= budget_left]
                if not avail:                       # nothing fits -> smallest available
                    avail = sorted(cal_pool, key=lambda r: r["cal"])[:1]
                pick = min(avail, key=lambda r: abs(r["cal"] - slot_target))
            else:
                avail = [r for r in pool if r["recipe_id"] not in used] or pool
                pick = avail[0]

            used.add(pick["recipe_id"])
            entry = {
                "day": day, "meal": slot_idx + 1, "meal_label": label,
                "recipe_id": pick["recipe_id"], "name": pick["name"],
            }
            if pick["cal"] is not None:
                entry["calories"] = round(pick["cal"], 1)
                day_total += pick["cal"]
                have_cals = True
            plan.append(entry)
        if have_cals:
            daily_totals.append({"day": day, "calories": round(day_total, 1)})

    if calorie_target is not None:
        assumptions.append(
            f"Each day has {meals_per_day} meal(s) suited to their time of day, "
            f"totaling under {calorie_target} kcal."
        )

    return {
        "type": "meal_plan",
        "days": plan,
        "daily_totals": daily_totals,
        "meals_per_day": meals_per_day,
        "assumptions": assumptions,
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

register_tool(
    ToolSpec(
        name="meal_planner",
        description=(
            "Build a multi-day meal plan. Use this for ANY request to plan meals "
            "over several days, with optional calorie/diet constraints. It finds "
            "its own recipes (breakfast recipes for breakfast, lunch for lunch, "
            "dinner for dinner) — just call it with days, meals_per_day (default "
            "3), calorie_target (per day), diet_type (e.g. 'vegetarian', 'vegan', "
            "'pescatarian', 'gluten-free', 'dairy-free', 'keto', 'paleo'; "
            "'non-vegetarian'/'any' = no restriction — the diet is strictly "
            "enforced by excluding incompatible ingredients), and optionally query "
            "or exclude (allergens). candidate_recipe_ids is optional. Returns a "
            "'days' list (one entry per meal) that can be passed to shopping_list, "
            "plus per-day calorie totals."
        ),
        callable=meal_planner,
        kind="planning",
    )
)