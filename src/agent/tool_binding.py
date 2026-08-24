"""Bind the registry's tools for the agent.

Wraps each tool so it accepts JSON/dict input and returns concise, factual text,
fixes an inferred-schema quirk that breaks Groq tool validation, and builds the
per-user save_user_profile tool.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from langchain_core.tools import StructuredTool

# Importing src.tools runs every tool module's register_tool(...) call, which
# populates the registry. Without this import the registry is empty and the
# agent is built with zero tools (i.e. answers are ungrounded).
import src.tools  # noqa: F401  (side-effect import: registers all tools)
from src.tools.registry import list_tools


def _format_recipe_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).title()


def _clean_blurb(desc: str, limit: int = 200) -> str:
    """Return a tidy description that ends on a sentence/word boundary (no mid-word cuts)."""
    desc = re.sub(r"\s+", " ", (desc or "").strip())
    if not desc or len(desc) <= limit:
        return desc
    # Prefer keeping whole sentences up to the limit.
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", desc):
        if not out:
            out = sent
        elif len(out) + 1 + len(sent) <= limit:
            out += " " + sent
        else:
            break
    # A single very long sentence: cut at the last whole word.
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return out


def _wrap_tool(fn, tool_name: str):
    """Allow JSON or dict input; return concise, factual text."""
    def _wrapped(tool_input=None, **kwargs):
        if tool_input is not None and isinstance(tool_input, str):
            try:
                kwargs = json.loads(tool_input)
            except json.JSONDecodeError:
                kwargs = {"query": tool_input}
        elif tool_input is not None and isinstance(tool_input, dict):
            kwargs = tool_input

        try:
            raw = fn(**(kwargs or {}))
            print(f"\n[TOOL: {tool_name}] Input: {kwargs}")
            print(f"[TOOL: {tool_name}] Output: {str(raw)[:500]}\n")

            if isinstance(raw, dict) and "recipes" in raw:
                recipes = raw.get("recipes") or []
                if recipes:
                    lines = []
                    for r in recipes[:10]:
                        if isinstance(r, dict):
                            rid = r.get("recipe_id") or r.get("id") or "N/A"
                            name = r.get("name") or r.get("title") or "Unknown"
                            # Include the real dataset description so the agent
                            # can use it as a grounded blurb instead of inventing one.
                            desc = _clean_blurb(r.get("description") or "")
                            line = f"- {_format_recipe_name(name)} (ID: {rid})"
                            line += f" — {desc}" if desc else " — (no description in dataset)"
                            lines.append(line)
                    return "Recipes:\n" + "\n".join(lines) if lines else "No recipes found."
                return "No recipes found."

            if tool_name == "resolve_recipe_by_name":
                if isinstance(raw, dict):
                    rid = raw.get("recipe_id") or raw.get("id")
                    name = raw.get("resolved_name") or raw.get("name") or raw.get("title") or ""
                    matches = raw.get("matches") or raw.get("recipes") or []
                    if rid:
                        return f"Resolved: {_format_recipe_name(name)} (ID: {rid})"
                    if matches:
                        lines = []
                        for m in matches[:10]:
                            if isinstance(m, dict):
                                mid = m.get("recipe_id") or m.get("id") or "N/A"
                                mname = m.get("name") or m.get("title") or "Unknown"
                                lines.append(f"- {_format_recipe_name(mname)} (ID: {mid})")
                        return "Multiple matches:\n" + "\n".join(lines)
                return "Recipe not found."

            if tool_name == "recipe_instructions":
                if isinstance(raw, dict):
                    rid = raw.get("recipe_id") or raw.get("id") or "N/A"
                    name = raw.get("name") or raw.get("title") or "Recipe"
                    ingredients = raw.get("ingredients") or []
                    steps = raw.get("instructions") or raw.get("steps") or []

                    out = [f"Recipe: {_format_recipe_name(name)} (ID: {rid})"]
                    if ingredients:
                        out.append("\nIngredients:")
                        for ing in ingredients:
                            text = ing.get("text") if isinstance(ing, dict) else str(ing)
                            out.append(f"- {text}")
                    out.append("\nSteps:")
                    if isinstance(steps, list) and steps:
                        for i, step in enumerate(steps, 1):
                            text = step.get("step") or step.get("text") if isinstance(step, dict) else str(step)
                            out.append(f"{i}. {text}")
                    else:
                        out.append("No instructions available.")
                    return "\n".join(out).strip()
                return "No instructions found."

            if tool_name == "nutrition_analyzer":
                if isinstance(raw, dict):
                    per_recipe = raw.get("per_recipe") or {}
                    meal_total = raw.get("meal_total") or {}

                    def _fmt_block(title, d):
                        rows = [
                            ("calories", "Calories", "kcal"),
                            ("protein_g", "Protein", "g"),
                            ("carbs_g", "Carbs", "g"),
                            ("total_fat_g", "Fat", "g"),
                            ("saturated_fat_g", "Saturated fat", "g"),
                            ("sugar_g", "Sugar", "g"),
                            ("sodium_mg", "Sodium", "mg"),
                        ]
                        out = [title]
                        for key, label, unit in rows:
                            if key in d:
                                out.append(f"  - {label}: {d[key]} {unit}")
                        return "\n".join(out)

                    blocks = []
                    for rid, d in per_recipe.items():
                        nm = _format_recipe_name(d.get("name") or "") or f"Recipe {rid}"
                        blocks.append(_fmt_block(f"{nm} (ID: {rid}) — per serving:", d))
                    if len(per_recipe) > 1 and meal_total:
                        blocks.append(_fmt_block("Meal total (all recipes):", meal_total))
                    text = "\n".join(blocks).strip()
                    return text if text else str(raw)
                return str(raw)

            if tool_name == "meal_planner":
                if isinstance(raw, dict) and raw.get("days"):
                    from collections import OrderedDict
                    by_day = OrderedDict()
                    for e in raw["days"]:
                        by_day.setdefault(e.get("day"), []).append(e)
                    totals = {d.get("day"): d.get("calories") for d in raw.get("daily_totals", [])}
                    out = []
                    for day, meals in by_day.items():
                        out.append(f"Day {day}:")
                        for e in meals:
                            label = e.get("meal_label") or f"Meal {e.get('meal')}"
                            line = f"  - {label}: {_format_recipe_name(e.get('name') or '')} (ID: {e.get('recipe_id')})"
                            if e.get("calories") is not None:
                                line += f" — {round(e['calories'])} kcal"
                            out.append(line)
                        if day in totals and totals[day] is not None:
                            out.append(f"  Day total: ~{round(totals[day])} kcal")
                    for a in raw.get("assumptions", []):
                        out.append(f"Note: {a}")
                    return "\n".join(out).strip()
                return str(raw)

            if tool_name == "modify_recipe":
                if isinstance(raw, dict):
                    if raw.get("status") == "not_found":
                        return "Recipe not found in the dataset."
                    name = _format_recipe_name(raw.get("name") or "Recipe")
                    out = [f"Recipe: {name} (ID: {raw.get('recipe_id')})"]
                    sf = raw.get("scale_factor")
                    if sf is not None:
                        out.append(
                            f"Scale factor: {sf} (from {raw.get('servings_from')} to "
                            f"{raw.get('servings_to')} servings) — multiply each "
                            f"ingredient amount by {sf}."
                        )
                    if raw.get("modification"):
                        out.append(f"Requested modification: {raw.get('modification')}")
                    ings = raw.get("ingredients") or []
                    if ings:
                        out.append("Original ingredients:")
                        out += [f"- {i}" for i in ings]
                    steps = raw.get("instructions") or []
                    if steps:
                        out.append("Original steps:")
                        out += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
                    for a in raw.get("assumptions", []):
                        out.append(f"Note: {a}")
                    return "\n".join(out).strip()
                return str(raw)

            if tool_name == "shopping_list":
                if isinstance(raw, dict) and raw.get("items"):
                    counts = raw.get("items_by_recipe_count") or {}
                    # Name the SOURCE recipes (with IDs) so the answer stays grounded.
                    src = raw.get("source_recipes") or []
                    n_items = raw.get("item_count", len(raw["items"]))
                    if src:
                        names = ", ".join(
                            f"{_format_recipe_name(r.get('name') or '')} (ID: {r.get('recipe_id')})"
                            for r in src
                        )
                        header = f"Shopping list ({n_items} items) for {names}:"
                    else:
                        header = f"Shopping list ({n_items} items from {raw.get('recipe_count', '?')} recipes):"
                    out = [header]
                    for it in raw["items"]:
                        c = counts.get(it)
                        out.append(f"- {it}" + (f" (in {c} recipes)" if c and c > 1 else ""))
                    for a in raw.get("assumptions", []):
                        out.append(f"Note: {a}")
                    return "\n".join(out).strip()
                if isinstance(raw, dict):
                    return (raw.get("assumptions") or ["No shopping list could be built."])[0]
                return str(raw)

            return str(raw)
        except Exception as e:
            print(f"[TOOL ERROR: {tool_name}] {str(e)}")
            return f"Error: {str(e)}"

    return _wrapped


def handle_parsing_error(error) -> str:
    """Return a usable string instead of raw parser errors."""
    s = str(error)
    # Strip common noise
    s = re.sub(r"(?i)could not parse LLM output:\s*", "", s)
    s = re.split(r"For troubleshooting", s, flags=re.IGNORECASE)[0]
    finals = re.findall(
        r"(?:\*\*Final Answer:\*\*|Final Answer:)\s*(.+?)(?=(?:\n\*\*Thought:|\nThought:|\nAction:|\n\*\*|$))",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if finals:
        txt = finals[-1].strip()
    else:
        blocks = [p.strip() for p in re.split(r"\n{2,}", s) if len(p.strip()) > 20]
        txt = blocks[-1] if blocks else "Sorry, I couldn't parse the response. Please try again."
    txt = re.sub(r"(OUTPUT_PARSING_FAILURE|visit: https?://\S+).*", "", txt, flags=re.IGNORECASE | re.DOTALL)
    return txt.strip()


def _schema_without_kwargs(schema):
    """Drop the spurious 'kwargs' field that **kwargs adds to an inferred schema.

    Some models (e.g. Llama 4 Scout) send `kwargs: null`, which fails Groq's tool
    schema validation ("expected object, but got null"). Removing the field from
    the schema the model sees avoids that entirely.
    """
    fields = getattr(schema, "model_fields", None)
    if not fields or "kwargs" not in fields:
        return schema
    try:
        from pydantic import create_model
        kept = {name: (fi.annotation, fi) for name, fi in fields.items() if name != "kwargs"}
        return create_model(getattr(schema, "__name__", "ToolArgs"), **kept)
    except Exception:
        return schema  # fail safe: keep the original schema


def _make_save_profile_tool(username: str) -> StructuredTool:
    """A profile-saving tool bound to one user (so the LLM can't touch others)."""
    from src.services.users import get_user_by_username, update_user_profile

    def save_user_profile(
        allergies: Optional[str] = None,
        diet_type: Optional[str] = None,
        health_goal: Optional[str] = None,
    ) -> str:
        """Persist the user's allergies / diet / health goal to their profile."""
        if not username:
            return ("Not saved to a profile (no one is logged in), but I'll "
                    "remember it for this conversation.")
        user = get_user_by_username(username)
        if not user:
            return "Could not save — user not found."

        updates = {}
        if allergies:
            # Merge with existing allergies (comma-separated, de-duplicated).
            existing = [a.strip() for a in re.split(r"[,;]", user.get("allergies") or "") if a.strip()]
            for a in re.split(r"[,;]", allergies):
                a = a.strip()
                if a and a.lower() not in [e.lower() for e in existing]:
                    existing.append(a)
            updates["allergies"] = ", ".join(existing)
        if diet_type:
            updates["diet_type"] = diet_type.strip()
        if health_goal:
            updates["health_goal"] = health_goal.strip()

        if not updates:
            return "Nothing to save."
        ok, msg = update_user_profile(username, **updates)
        return ("Saved to your profile: " +
                "; ".join(f"{k}: {v}" for k, v in updates.items())) if ok else f"Save failed: {msg}"

    return StructuredTool.from_function(
        func=save_user_profile,
        name="save_user_profile",
        description=(
            "Persist the CURRENT user's allergies, dietary preference (diet_type), "
            "and/or health goal to their saved profile so they are remembered next "
            "session. Call this whenever the user states one, e.g. 'I'm allergic to "
            "peanuts' (allergies='peanuts'), 'I'm vegan' (diet_type='vegan'), 'I "
            "want to lose weight' (health_goal='lose weight')."
        ),
    )


def build_tools(username: str | None = None) -> List[StructuredTool]:
    """Build the agent's tool list: every registered tool (wrapped for concise,
    factual text) plus the per-user save_user_profile tool.

    The profile-save tool is always exposed (the prompt references it). For a
    logged-in user it writes to the DB; for a guest it politely no-ops. This
    avoids "tool call validation failed" when the model tries to call it.
    """
    tools: List[StructuredTool] = []
    for spec in list_tools().values():
        base = StructuredTool.from_function(
            spec.callable, name=spec.name, description=spec.description
        )
        tools.append(
            StructuredTool.from_function(
                func=_wrap_tool(spec.callable, spec.name),
                name=spec.name,
                description=spec.description,
                args_schema=_schema_without_kwargs(base.args_schema),
            )
        )
    tools.append(_make_save_profile_tool(username))
    return tools