"""Diet rules — turn a stated diet into concrete, enforceable constraints.

A user's diet is honored in two complementary ways:
  1. FORBIDDEN INGREDIENTS — a list of ingredient terms that must not appear in a
     recipe (enforced deterministically via db.exclude_ingredients, the same
     word-boundary matcher used for allergens). This is the hard safety net: a
     "vegan" plan can never contain chicken, tuna, or milk even if the recipe
     wasn't tagged vegan.
  2. COMPLIANCE TAGS — dataset tags that positively mark a recipe as fitting the
     diet (e.g. "vegetarian", "vegan", "gluten-free"). Preferred when available.

Unknown / macro-only diets (e.g. an unrecognized label) return no forbidden list;
callers should fall back to tag matching and note that strict enforcement wasn't
possible rather than silently serving a non-compliant recipe.
"""
from __future__ import annotations

from typing import List

# Diets that impose no restriction at all.
NO_RESTRICTION = {
    "", "non-vegetarian", "non vegetarian", "nonveg", "none", "any",
    "no preference", "omnivore", "no restriction", "everything",
}

# --- ingredient groups (word-stem terms; matched with a leading word boundary) ---
_MEAT = [
    "beef", "pork", "chicken", "turkey", "lamb", "veal", "venison", "bacon",
    "ham", "sausage", "salami", "pepperoni", "prosciutto", "chorizo", "meat",
    "steak", "mince", "hamburger", "burger", "meatball", "meatloaf", "duck",
    "goose", "liver", "lard", "suet", "gelatin", "gelatine",
]
_FISH = [
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "sardine", "anchovy",
    "trout", "mackerel", "herring", "shrimp", "prawn", "crab", "lobster", "clam",
    "oyster", "mussel", "scallop", "squid", "octopus", "seafood", "caviar", "roe",
]
_DAIRY = [
    "milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee", "whey",
    "casein", "custard", "paneer",
]
_EGG = ["egg", "mayonnaise", "mayo"]
_HONEY = ["honey"]
_GLUTEN = [
    "wheat", "flour", "bread", "breadcrumb", "pasta", "noodle", "spaghetti",
    "macaroni", "barley", "rye", "couscous", "cracker", "semolina", "tortilla",
    "bun", "bagel", "malt", "bulgur", "farro",
]
_HIGH_CARB = [
    "sugar", "bread", "pasta", "rice", "potato", "flour", "noodle", "corn",
    "cereal", "syrup", "honey",
]

# --- canonical diet -> (forbidden ingredient terms, compliance tags) ---
_RULES = {
    "vegetarian": (_MEAT + _FISH, ["vegetarian", "vegan"]),
    "vegan":      (_MEAT + _FISH + _DAIRY + _EGG + _HONEY, ["vegan"]),
    "pescatarian": (_MEAT, ["pescatarian", "vegetarian", "vegan"]),  # fish allowed
    "gluten-free": (_GLUTEN, ["gluten-free"]),
    "dairy-free": (_DAIRY, ["dairy-free", "vegan"]),
    "keto":       (_HIGH_CARB, ["keto", "low-carb"]),
    "low-carb":   (_HIGH_CARB, ["low-carb", "keto"]),
    "paleo":      (_DAIRY + _GLUTEN + ["beans", "lentil", "chickpea", "soy", "peanut"],
                   ["paleo"]),
}

# Map common phrasings a user might type to a canonical diet key.
_ALIASES = {
    "vegetarian": "vegetarian", "veggie": "vegetarian", "veg": "vegetarian",
    "vegan": "vegan", "plant-based": "vegan", "plant based": "vegan",
    "pescatarian": "pescatarian", "pescetarian": "pescatarian",
    "gluten-free": "gluten-free", "gluten free": "gluten-free", "gf": "gluten-free",
    "celiac": "gluten-free", "coeliac": "gluten-free",
    "dairy-free": "dairy-free", "dairy free": "dairy-free",
    "lactose-free": "dairy-free", "lactose free": "dairy-free",
    "keto": "keto", "ketogenic": "keto",
    "low-carb": "low-carb", "low carb": "low-carb",
    "paleo": "paleo", "paleolithic": "paleo",
}


def canonical_diet(diet: str) -> str:
    """Normalize a stated diet to a canonical key (''=no restriction, or the
    raw lower-cased text if unrecognized)."""
    d = (diet or "").strip().lower()
    if d in NO_RESTRICTION:
        return ""
    return _ALIASES.get(d, d)


def is_restricted(diet: str) -> bool:
    """True if the diet imposes any restriction we should honor."""
    return canonical_diet(diet) != ""


def diet_forbidden_ingredients(diet: str) -> List[str]:
    """Ingredient terms that must NOT appear in a recipe for this diet.
    Empty for no-restriction OR an unrecognized diet (enforce via tags instead)."""
    key = canonical_diet(diet)
    if not key:
        return []
    forbidden, _tags = _RULES.get(key, ([], []))
    return list(forbidden)


def diet_tags(diet: str) -> List[str]:
    """Dataset tags that positively mark a recipe as fitting this diet."""
    key = canonical_diet(diet)
    if not key:
        return []
    _forbidden, tags = _RULES.get(key, ([], [key]))
    return list(tags)


def is_known_diet(diet: str) -> bool:
    """True if we have explicit rules (forbidden ingredients + tags) for this diet."""
    return canonical_diet(diet) in _RULES
