CREATE TABLE IF NOT EXISTS recipes (
  recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  description TEXT,
  minutes INTEGER,
  n_steps INTEGER,
  n_ingredients INTEGER,

  calories REAL,
  total_fat_pdv REAL,
  sugar_pdv REAL,
  sodium_pdv REAL,
  protein_pdv REAL,
  saturated_fat_pdv REAL,
  carbs_pdv REAL,

  steps_json TEXT,
  ingredients_json TEXT,
  tags_json TEXT,
  document TEXT
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
  recipe_id INTEGER,
  ingredient TEXT,
  PRIMARY KEY (recipe_id, ingredient)
);

CREATE TABLE IF NOT EXISTS recipe_tags (
  recipe_id INTEGER,
  tag TEXT,
  PRIMARY KEY (recipe_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_recipe_name ON recipes(name);
CREATE INDEX IF NOT EXISTS idx_ingredient ON recipe_ingredients(ingredient);
CREATE INDEX IF NOT EXISTS idx_tag ON recipe_tags(tag);
