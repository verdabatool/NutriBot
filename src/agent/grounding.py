"""Grounding guard: never let a fabricated recipe ID reach the user.

Checks that every ``(ID: N)`` a reply cites is a real recipe whose name matches
the database — catching both invented IDs and real IDs the model captioned with
a made-up name.
"""
from __future__ import annotations

import re
from typing import List

# Matches "(ID: 12345)" while tolerating markdown emphasis the model may wrap the
# marker or number in — e.g. "(ID: **12345**)", "(ID: `12345`)", "(**ID: 12345**)".
# Some models (e.g. gpt-oss-20b) bold the id; without this the grounding and
# allergen guards would fail to see it and a fabricated/allergen id could slip
# through. Capture group 1 is always the bare digits.
ID_MARKER_RE = re.compile(r"\(\s*[*_`~]*\s*ID:\s*[*_`~]*\s*(\d+)\s*[*_`~]*\s*\)")

_NAME_STOPWORDS = {
    "the", "and", "with", "for", "recipe", "recipes", "quick", "easy", "made",
    "from", "your", "this", "that", "style", "homemade", "classic", "simple",
    "best", "delicious", "healthy", "one", "pot",
}


def _claimed_name_before(text: str, idx: int) -> str:
    """Grab the recipe name written just before a '(ID: N)' marker."""
    seg = text[:idx]
    seg = re.split(r"[\n•]|\d+\.\s|[-*]\s", seg)[-1]      # tail of the current list item
    seg = re.sub(r"[*_#>`]", "", seg)                      # strip markdown
    return seg.strip().strip("-–—:•. ").strip()


def ungrounded_ids(text: str) -> List[str]:
    """Return recipe ids in the text that are hallucinated: either the id does not
    exist, OR the name the model wrote next to it does not match that id's real
    recipe (the model invented a plausible name over a random real id)."""
    text = str(text or "")
    if "ID:" not in text:
        return []
    try:
        from src.db.recipes import get_recipes_by_ids
    except Exception:
        return []

    bad: List[str] = []
    for m in ID_MARKER_RE.finditer(text):
        rid = int(m.group(1))
        try:
            df = get_recipes_by_ids([rid])
        except Exception:
            continue  # can't check -> don't block
        if df.empty:
            bad.append(str(rid))                          # id not in DB at all
            continue
        actual = str(df.iloc[0]["name"]).lower()
        claimed = _claimed_name_before(text, m.start()).lower()
        claimed_tokens = {w for w in re.findall(r"[a-z]{3,}", claimed)} - _NAME_STOPWORDS
        actual_tokens = {w for w in re.findall(r"[a-z]{3,}", actual)} - _NAME_STOPWORDS
        # If the claimed name shares no meaningful word with the real recipe name,
        # the model fabricated the name over this id.
        if claimed_tokens and not (claimed_tokens & actual_tokens):
            bad.append(str(rid))
    return bad


def allergen_hits(text: str, allergens: List[str]) -> List[int]:
    """Return recipe ids cited in the text whose REAL database ingredients contain
    one of the user's allergens.

    This is the deterministic safety net that makes allergen exclusion independent
    of whether the LLM remembered to pass exclude=[...] to a tool. Matching mirrors
    exclude_ingredients() — singularized, word-boundary — so a recipe flagged here
    is exactly one the exclusion filter would have removed (e.g. banning "peanuts"
    also catches "peanut butter"/"peanut oil"). Errs toward over-detection: safer
    to re-check a borderline recipe than to serve an allergen.
    """
    text = str(text or "")
    if not allergens or "ID:" not in text:
        return []
    try:
        from src.db.recipes import get_recipe_ingredients
    except Exception:
        return []

    stems = []
    for a in allergens:
        a = (a or "").strip().lower()
        if not a:
            continue
        stems.append(re.escape(a[:-1] if a.endswith("s") and len(a) > 3 else a))
    if not stems:
        return []
    pattern = re.compile(r"\b(" + "|".join(stems) + r")")

    hits: List[int] = []
    for m in ID_MARKER_RE.finditer(text):
        rid = int(m.group(1))
        try:
            ings = get_recipe_ingredients(rid)
        except Exception:
            continue  # can't check -> don't block
        if any(pattern.search(str(ing).lower()) for ing in ings):
            hits.append(rid)
    return hits


def diet_hits(text: str, forbidden: List[str]) -> List[int]:
    """Return recipe ids cited in the text whose REAL ingredients contain a term
    forbidden by the user's diet (e.g. meat/fish for vegetarian).

    Unlike allergen_hits, matching is PRECISE (whole word, optional plural) — a
    diet guard that wrongly rejects a valid recipe is worse than missing an edge
    case, so "egg" must not match "eggplant" nor "butter" match "butternut". The
    caller supplies the forbidden terms (see tools.diet.diet_forbidden_ingredients);
    an empty/unknown-diet list means no check runs.
    """
    text = str(text or "")
    terms = [re.escape((t or "").strip().lower()) for t in forbidden if (t or "").strip()]
    if not terms or "ID:" not in text:
        return []
    try:
        from src.db.recipes import get_recipe_ingredients
    except Exception:
        return []

    pattern = re.compile(r"\b(" + "|".join(terms) + r")s?\b")

    hits: List[int] = []
    for m in ID_MARKER_RE.finditer(text):
        rid = int(m.group(1))
        try:
            ings = get_recipe_ingredients(rid)
        except Exception:
            continue  # can't check -> don't block
        if any(pattern.search(str(ing).lower()) for ing in ings):
            hits.append(rid)
    return hits