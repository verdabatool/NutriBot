"""Grounding guard: never let a fabricated recipe ID reach the user.

Checks that every ``(ID: N)`` a reply cites is a real recipe whose name matches
the database — catching both invented IDs and real IDs the model captioned with
a made-up name.
"""
from __future__ import annotations

import re
from typing import List

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
    if "(ID:" not in text:
        return []
    try:
        from src.db.recipes import get_recipes_by_ids
    except Exception:
        return []

    bad: List[str] = []
    for m in re.finditer(r"\(ID:\s*(\d+)\)", text):
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