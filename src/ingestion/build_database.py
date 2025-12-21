import ast
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.db.engine import engine
from src.config.settings import PROCESSED_DIR

# --------------------------------------------------
# Paths
# --------------------------------------------------
CSV_PATH = PROCESSED_DIR / "PROCESSED_recipes.csv"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# --------------------------------------------------
# Main builder
# --------------------------------------------------
def build_database():
    print("🚀 Starting database build")
    print("⚠️ This will DROP and rebuild recipe tables")

    # ----------------------------
    # Load processed data
    # ----------------------------
    print(f"📄 Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in ["tags", "steps", "ingredients"]:
        df[col] = df[col].apply(ast.literal_eval)

    # ----------------------------
    # Reset schema (IMPORTANT)
    # ----------------------------
    print("🧱 Resetting database schema")
    print(f"📜 Loading schema from {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text()

    with engine.begin() as conn:
        raw = conn.connection

        # 🔥 Drop tables explicitly (prevents zombie schemas)
        raw.executescript("""
        DROP TABLE IF EXISTS recipe_ingredients;
        DROP TABLE IF EXISTS recipe_tags;
        DROP TABLE IF EXISTS recipes;
        """)

        # ✅ Create fresh schema
        raw.executescript(schema_sql)

    # ----------------------------
    # Insert recipes
    # ----------------------------
    print(f"📥 Inserting {len(df)} recipes")

    df_db = df.copy()
    df_db["steps_json"] = df_db["steps"].apply(json.dumps)
    df_db["ingredients_json"] = df_db["ingredients"].apply(json.dumps)
    df_db["tags_json"] = df_db["tags"].apply(json.dumps)

    recipe_cols = [
        "name",
        "description",
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbs_pdv",
        "steps_json",
        "ingredients_json",
        "tags_json",
        "document",
    ]

    df_db[recipe_cols].to_sql(
        "recipes",
        engine,
        if_exists="append",   # ⚠️ never replace constrained tables
        index=False
    )

    # ----------------------------
    # Fetch recipe IDs
    # ----------------------------
    print("🔁 Fetching recipe IDs")

    recipes = pd.read_sql(
        "SELECT recipe_id FROM recipes ORDER BY recipe_id",
        engine
    )

    assert len(recipes) == len(df), (
        "❌ Row count mismatch between CSV and database "
        f"({len(df)} vs {len(recipes)})"
    )

    # ----------------------------
    # Build ingredient & tag tables
    # ----------------------------
    ingredient_rows = []
    tag_rows = []

    print("🧩 Building ingredient & tag tables")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        rid = int(recipes.iloc[i]["recipe_id"])

        for ing in row["ingredients"]:
            ingredient_rows.append({
                "recipe_id": rid,
                "ingredient": ing.strip().lower()
            })

        for tag in row["tags"]:
            tag_rows.append({
                "recipe_id": rid,
                "tag": tag.strip().lower()
            })

    pd.DataFrame(ingredient_rows).drop_duplicates().to_sql(
        "recipe_ingredients",
        engine,
        if_exists="append",
        index=False
    )

    pd.DataFrame(tag_rows).drop_duplicates().to_sql(
        "recipe_tags",
        engine,
        if_exists="append",
        index=False
    )

    # ----------------------------
    # Done
    # ----------------------------
    print("✅ Database build complete")
    print(f"📊 Total recipes: {len(df)}")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    build_database()
