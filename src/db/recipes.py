# src/db/recipes.py

import re
from typing import List, Optional
import pandas as pd
from sqlalchemy import text

from src.db.engine import engine


# ==================================================
# Core recipe access
# ==================================================

def get_recipe_by_id(recipe_id: int) -> Optional[dict]:
    """
    Fetch a single recipe by ID.
    """
    query = """
    SELECT *
    FROM recipes
    WHERE recipe_id = ?
    """
    df = pd.read_sql(query, engine, params=(recipe_id,))
    return df.iloc[0].to_dict() if not df.empty else None


def get_recipes_by_ids(recipe_ids: List[int]) -> pd.DataFrame:
    """
    Fetch multiple recipes by ID.
    Preserves input order.
    """
    if not recipe_ids:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in recipe_ids)
    query = f"""
    SELECT *
    FROM recipes
    WHERE recipe_id IN ({placeholders})
    """

    df = pd.read_sql(
        query,
        engine,
        params=tuple(recipe_ids),
    )

    # Preserve input order
    order_map = {rid: i for i, rid in enumerate(recipe_ids)}
    df["_order"] = df["recipe_id"].map(order_map)

    return df.sort_values("_order").drop(columns="_order")


# ==================================================
# Ingredient-based retrieval (STRICT / LOOSE / FALLBACK)
# ==================================================

def get_recipes_with_all_ingredients(ingredients: List[str]) -> pd.DataFrame:
    """
    Return recipes that contain ALL specified ingredients.
    """
    if not ingredients:
        return pd.DataFrame()

    ingredients = [i.lower() for i in ingredients]
    placeholders = ",".join("?" for _ in ingredients)

    query = f"""
    SELECT r.*
    FROM recipes r
    JOIN recipe_ingredients ri
      ON r.recipe_id = ri.recipe_id
    WHERE ri.ingredient IN ({placeholders})
    GROUP BY r.recipe_id
    HAVING COUNT(DISTINCT ri.ingredient) = ?
    """

    params = tuple(ingredients) + (len(ingredients),)
    return pd.read_sql(query, engine, params=params)


def get_recipes_with_any_ingredients(
        ingredients: List[str],
        min_matches: int = 1,
) -> pd.DataFrame:
    """
    Return recipes that match AT LEAST `min_matches` of the given ingredients.
    SQLite-safe implementation.
    """

    if not ingredients or min_matches <= 0:
        return pd.DataFrame()

    ingredients = [ing.lower().strip() for ing in ingredients]

    score_clauses = []
    where_clauses = []
    params = {}

    for i, ing in enumerate(ingredients):
        key = f"ing{i}"
        score_clauses.append(
            f"CASE WHEN LOWER(ingredients_json) LIKE :{key} THEN 1 ELSE 0 END"
        )
        where_clauses.append(f"LOWER(ingredients_json) LIKE :{key}")
        params[key] = f"%{ing}%"

    score_sql = " + ".join(score_clauses)
    where_sql = " OR ".join(where_clauses)

    query = f"""
    SELECT *
    FROM (
        SELECT
            *,
            ({score_sql}) AS match_count
        FROM recipes
        WHERE ({where_sql})
    )
    WHERE match_count >= :min_matches
    ORDER BY match_count DESC
    """

    params["min_matches"] = min_matches

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)



def get_recipes_with_partial_ingredients(
    ingredients: List[str],
    min_matches: int,
) -> pd.DataFrame:
    """
    Return recipes that match AT LEAST `min_matches` ingredients.
    Fallback matcher using ingredients_json.
    """
    if not ingredients or min_matches <= 0:
        return pd.DataFrame()

    ingredients = [ing.lower().strip() for ing in ingredients]

    like_clauses = []
    params = {}

    for i, ing in enumerate(ingredients):
        key = f"ing{i}"
        like_clauses.append(f"LOWER(ingredients_json) LIKE :{key}")
        params[key] = f"%{ing}%"

    where_sql = " OR ".join(like_clauses)

    score_sql = " + ".join(
        [
            f"CASE WHEN LOWER(ingredients_json) LIKE :ing{i} THEN 1 ELSE 0 END"
            for i in range(len(ingredients))
        ]
    )

    query = f"""
    SELECT *,
           ({score_sql}) AS match_count
    FROM recipes
    WHERE ({where_sql})
    HAVING match_count >= :min_matches
    ORDER BY match_count DESC
    """

    params["min_matches"] = min_matches

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


# ==================================================
# Exclusion filters
# ==================================================

def exclude_ingredients(df: pd.DataFrame, banned: List[str]) -> pd.DataFrame:
    """
    Exclude recipes whose ingredients contain any banned term (e.g. allergens).

    Matching is by word-boundary substring on each recipe's ingredient list, so
    banning "peanuts" also removes "peanut butter"/"peanut oil", and "chicken"
    also removes "chicken breast"/"chicken broth". For allergy SAFETY this errs
    toward over-exclusion rather than missing an allergen.
    """
    if df.empty or not banned or "ingredients_json" not in df.columns:
        return df

    stems = []
    for b in banned:
        b = (b or "").strip().lower()
        if not b:
            continue
        # simple singularization so "peanuts" -> "peanut", "eggs" -> "egg"
        stem = b[:-1] if b.endswith("s") and len(b) > 3 else b
        stems.append(re.escape(stem))

    if not stems:
        return df

    pattern = re.compile(r"\b(" + "|".join(stems) + r")")
    contains_banned = (
        df["ingredients_json"].fillna("").str.lower().apply(lambda s: bool(pattern.search(s)))
    )
    return df[~contains_banned]


# ==================================================
# Lightweight helpers (used by tools)
# ==================================================

def get_recipe_ingredients(recipe_id: int) -> List[str]:
    """
    Return ingredient list for a recipe.
    """
    query = """
    SELECT ingredient
    FROM recipe_ingredients
    WHERE recipe_id = ?
    """
    df = pd.read_sql(query, engine, params=(recipe_id,))
    return df["ingredient"].tolist()


def get_recipe_tags(recipe_id: int) -> List[str]:
    """
    Return tags for a recipe.
    """
    query = """
    SELECT tag
    FROM recipe_tags
    WHERE recipe_id = ?
    """
    df = pd.read_sql(query, engine, params=(recipe_id,))
    return df["tag"].tolist()


def get_recipe_nutrition(recipe_id: int) -> dict:
    """
    Return nutrition fields for a single recipe.
    """
    query = """
    SELECT
        calories,
        total_fat_pdv,
        sugar_pdv,
        sodium_pdv,
        protein_pdv,
        saturated_fat_pdv,
        carbs_pdv
    FROM recipes
    WHERE recipe_id = ?
    """
    df = pd.read_sql(query, engine, params=(recipe_id,))
    return df.iloc[0].to_dict() if not df.empty else {}
