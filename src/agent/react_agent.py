# from __future__ import annotations

# from pathlib import Path
# from typing import List
# import json
# import re

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.tools import StructuredTool
# from langchain.agents import create_react_agent, AgentExecutor
# from langgraph.checkpoint.memory import InMemorySaver

# # Ensure tools are registered
# from src.tools.registry import list_tools

# PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "react_agent.txt"


# def _format_recipe_name(name: str) -> str:
#     """Normalize recipe naming (title case, single spaces)."""
#     return re.sub(r"\s+", " ", (name or "").strip()).title()


# def _wrap_tool(fn, tool_name: str):
#     """Wrap tool callables to accept JSON or dict and return concise, factual text."""
#     def _wrapped(tool_input=None, **kwargs):
#         # Normalize input
#         if tool_input is not None and isinstance(tool_input, str):
#             try:
#                 kwargs = json.loads(tool_input)
#             except json.JSONDecodeError:
#                 # Generic fallback for lookups
#                 kwargs = {"query": tool_input}
#         elif tool_input is not None and isinstance(tool_input, dict):
#             kwargs = tool_input

#         try:
#             raw = fn(**(kwargs or {}))
#             print(f"\n[TOOL: {tool_name}] Input: {kwargs}")
#             print(f"[TOOL: {tool_name}] Output: {str(raw)[:500]}\n")

#             # Common formatting patterns
#             if isinstance(raw, dict) and "recipes" in raw:
#                 recipes = raw.get("recipes") or []
#                 if recipes:
#                     lines = []
#                     for r in recipes[:10]:
#                         if isinstance(r, dict):
#                             rid = r.get("recipe_id") or r.get("id") or "N/A"
#                             name = r.get("name") or r.get("title") or "Unknown"
#                             lines.append(f"- {_format_recipe_name(name)} (ID: {rid})")
#                     return "Recipes:\n" + "\n".join(lines) if lines else "No recipes found."
#                 return "No recipes found."

#             if tool_name == "resolve_recipe_by_name":
#                 if isinstance(raw, dict):
#                     rid = raw.get("recipe_id") or raw.get("id")
#                     name = raw.get("resolved_name") or raw.get("name") or raw.get("title") or ""
#                     matches = raw.get("matches") or raw.get("recipes") or []
#                     if rid:
#                         return f"Resolved: {_format_recipe_name(name)} (ID: {rid})"
#                     if matches:
#                         lines = []
#                         for m in matches[:10]:
#                             if isinstance(m, dict):
#                                 mid = m.get("recipe_id") or m.get("id") or "N/A"
#                                 mname = m.get("name") or m.get("title") or "Unknown"
#                                 lines.append(f"- {_format_recipe_name(mname)} (ID: {mid})")
#                         return "Multiple matches:\n" + "\n".join(lines)
#                 return "Recipe not found."

#             if tool_name == "recipe_instructions":
#                 if isinstance(raw, dict):
#                     rid = raw.get("recipe_id") or raw.get("id") or "N/A"
#                     name = raw.get("name") or raw.get("title") or "Recipe"
#                     ingredients = raw.get("ingredients") or []
#                     steps = raw.get("instructions") or raw.get("steps") or []

#                     out = [f"Recipe: {_format_recipe_name(name)} (ID: {rid})"]
#                     if ingredients:
#                         out.append("\nIngredients:")
#                         for ing in ingredients:
#                             text = ing.get("text") if isinstance(ing, dict) else str(ing)
#                             out.append(f"- {text}")
#                     out.append("\nSteps:")
#                     if isinstance(steps, list) and steps:
#                         for i, step in enumerate(steps, 1):
#                             text = step.get("step") or step.get("text") if isinstance(step, dict) else str(step)
#                             out.append(f"{i}. {text}")
#                     else:
#                         out.append("No instructions available.")
#                     return "\n".join(out).strip()
#                 return "No instructions found."

#             if tool_name == "nutrition_analyzer":
#                 if isinstance(raw, dict):
#                     keys = ["calories", "protein", "carbohydrates", "fat", "fiber", "sugar"]
#                     lines = [f"- {k.capitalize()}: {raw[k]}" for k in keys if k in raw]
#                     return "Nutrition:\n" + "\n".join(lines) if lines else str(raw)
#                 return str(raw)

#             # Default: stringify
#             return str(raw)

#         except Exception as e:
#             print(f"[TOOL ERROR: {tool_name}] {str(e)}")
#             return f"Error: {str(e)}"

#     return _wrapped


# def _handle_parsing_error(error) -> str:
#     """Gracefully extract a usable message when the LLM mixes formats."""
#     s = str(error)
#     s = re.split(r"For troubleshooting", s, flags=re.IGNORECASE)[0]
#     finals = re.findall(
#         r"(?:\*\*Final Answer:\*\*|Final Answer:)\s*(.+?)(?=(?:\n\*\*Thought:|\nThought:|\nAction:|\n\*\*|$))",
#         s,
#         flags=re.IGNORECASE | re.DOTALL,
#     )
#     if finals:
#         txt = finals[-1].strip()
#     else:
#         blocks = [p.strip() for p in re.split(r"\n{2,}", s) if len(p.strip()) > 20]
#         txt = blocks[-1] if blocks else "Sorry, I couldn't parse the assistant response. Please try again."
#     txt = re.sub(r"(OUTPUT_PARSING_FAILURE|visit: https?://\S+).*", "", txt, flags=re.IGNORECASE | re.DOTALL)
#     return txt.strip()


# def build_agent():
#     """Build and return the ReAct agent with canonical tools and preference support."""
#     prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
#     prompt = PromptTemplate(
#         input_variables=["input", "agent_scratchpad", "tools", "tool_names", "user_preference"],
#         template=prompt_text,
#     )

#     # Choose a reliable Groq model; keep outputs short to avoid 413
#     llm = ChatGroq(
#        model="openai/gpt-oss-120b",
#        temperature=0.2,
#        max_tokens=512,
#     )

#     # Register ONLY canonical tool names to avoid confusion and reduce prompt size
#     tools: List[StructuredTool] = []
#     for spec in list_tools().values():
#         tools.append(
#             StructuredTool.from_function(
#                 func=_wrap_tool(spec.callable, spec.name),
#                 name=spec.name,                # canonical name (e.g., recipe_lookup)
#                 description=spec.description,
#             )
#         )

#     agent = create_react_agent(llm, tools, prompt)
#     executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         verbose=False,
#         checkpointer=InMemorySaver(),
#         handle_parsing_errors=_handle_parsing_error,
#         max_iterations=4,  # keep fast; avoids “Agent stopped due to iteration limit”
#     )
#     return executor


from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json
import re
from os import getenv

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

# Tools registry.
# Importing src.tools runs every tool module's register_tool(...) call,
# which populates TOOL_REGISTRY. Without this import the registry is empty
# and the agent is built with zero tools (i.e. answers are ungrounded).
import src.tools  # noqa: F401  (side-effect import: registers all tools)
from src.tools.registry import list_tools

# NOTE: gpt-oss models on Groq use native tool-calling in a way that is
# incompatible with LangChain's *text* ReAct agent; and text ReAct with Llama
# loops/hallucinates. We use a native tool-calling agent with a Llama chat
# model, overridable via GROQ_MODEL.
DEFAULT_MODEL = getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Model used automatically when the main model is rate-limited.
FALLBACK_MODEL = getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

# System prompt for the tool-calling agent (grounding + persona).
SYSTEM_PROMPT = (
    "You are NutriChat — a friendly, practical recipe assistant grounded in a "
    "recipe database that you access through tools.\n\n"
    "GROUNDING (most important):\n"
    "- Every recipe name, ID, ingredient, cooking step, and nutrition value you "
    "state MUST come from a tool result — never invent or guess them.\n"
    "- Call a tool before answering any question about specific recipes, and use "
    "only the real names and IDs the tools return.\n"
    "- Once you know WHICH recipe, to answer how it is made — its exact steps, "
    "cooking times, or temperatures — open it with recipe_instructions and read "
    "its real steps before answering.\n"
    "- If a tool returns nothing, say so honestly and offer alternatives.\n"
    "- If the database genuinely lacks a specific detail (e.g. an exact bake time "
    "not written in the steps), you MAY answer briefly from general cooking "
    "knowledge, but clearly label it as general (e.g. 'The recipe doesn't specify; "
    "generally ...'). Never present general knowledge as if it came from the data.\n\n"
    "CHOOSING TOOLS:\n"
    "- Read each tool's description and reason about what the user actually wants, "
    "then use the tool whose purpose fits. Prefer real tool data over your own "
    "knowledge, and don't call tools you don't need.\n\n"
    "PRESENTING RECIPES:\n"
    "- When you recommend recipes, WRITE OUT each option's real name and (ID: N) "
    "explicitly, with a short blurb from the tool data when available. Never refer "
    "to options you didn't actually list, and never show an ID a tool didn't give "
    "you.\n"
    "- If a question points to ONE specific recipe (named or by ID), answer it "
    "directly and concisely (don't dump the whole recipe unless asked).\n"
    "- CLARIFY WHEN AMBIGUOUS: if a question could apply to MANY different recipes "
    "(e.g. 'how long do I bake lasagna?' — there are many lasagnas with different "
    "times), do NOT pick one arbitrarily. Call recipe_lookup and ask which recipe "
    "they mean, listing a few real options with their IDs. Answer only once a "
    "specific recipe is identified.\n"
    "- When sharing a full recipe, order it: Title (+ID), Ingredients, Steps (in "
    "the tool's order), Nutrition if available, optional 'Pro tip'.\n\n"
    "CONVERSATION MEMORY:\n"
    "- The earlier conversation is available — use it. Resolve references like "
    "'it', 'this recipe', 'the pasta', 'the 3rd one', 'the last one' to the "
    "specific recipe (name + ID) most recently discussed, and act on that recipe. "
    "Keep applying constraints the user stated earlier. If a reference is genuinely "
    "ambiguous, ask one short clarifying question.\n\n"
    "PERSONALIZATION & SAFETY:\n"
    "- The USER CONTEXT below may list saved allergies, diet, and health goal. "
    "NEVER suggest a recipe containing an allergen, and pass exclude=[<allergens>] "
    "when you search recipes. Honor the stated diet and goal in every suggestion.\n"
    "- When the user states a new allergy, diet, or health goal, save it with "
    "save_user_profile and keep applying it for the rest of the chat.\n\n"
    "DATASET FACTS (state honestly):\n"
    "- Ingredient quantities are NOT stored, so express scaling as a factor to "
    "multiply each ingredient by, and ask for the original serving count if it "
    "wasn't given. Nutrition grams are ESTIMATED from the dataset's % Daily Value. "
    "Any recipe modification/substitution is your suggestion, not from the data.\n\n"
    "STYLE: warm, concise, and complete — don't truncate lists or use '...'.\n"
    "USER CONTEXT: {user_preference}"
)


# --------------------------------------------------
# Per-session chat memory (summary-buffer, to cap token growth)
# --------------------------------------------------
# Keep the most recent turns verbatim; fold older turns into a running summary
# that preserves the key facts (recipes + IDs, allergies/diet/goals, decisions).
MEMORY_KEEP_LAST = int(getenv("MEMORY_KEEP_LAST", "8"))     # recent messages kept verbatim
MEMORY_TRIGGER_AT = int(getenv("MEMORY_TRIGGER_AT", "14"))  # summarize once history exceeds this

_SUMMARIZER = None


def _get_summarizer():
    """A cheap, fast model for summarizing old turns (separate from the main LLM)."""
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = ChatGroq(
            model=getenv("GROQ_SUMMARY_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            max_tokens=512,
        )
    return _SUMMARIZER


class SummarizingChatMessageHistory(BaseChatMessageHistory):
    """Chat history that keeps recent messages verbatim and summarizes older ones."""

    def __init__(self, summarizer=None, keep_last=MEMORY_KEEP_LAST, trigger_at=MEMORY_TRIGGER_AT):
        self._messages: List[BaseMessage] = []
        self.summary: str = ""
        self._summarizer = summarizer
        self._keep_last = max(2, keep_last)
        self._trigger_at = max(self._keep_last + 2, trigger_at)

    @property
    def messages(self) -> List[BaseMessage]:
        out: List[BaseMessage] = []
        if self.summary:
            out.append(SystemMessage(
                content=f"Summary of earlier conversation (for context): {self.summary}"
            ))
        out.extend(self._messages)
        return out

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._maybe_summarize()

    def add_messages(self, messages) -> None:
        self._messages.extend(messages)
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        if len(self._messages) <= self._trigger_at:
            return
        overflow = self._messages[:-self._keep_last]
        recent = self._messages[-self._keep_last:]
        convo = "\n".join(
            f"{getattr(m, 'type', 'msg')}: {getattr(m, 'content', '')}" for m in overflow
        )
        prompt = (
            "You maintain a running summary of a cooking-assistant conversation. "
            "Update it to fold in the new messages, PRESERVING specifics: recipes "
            "discussed (names + IDs), the user's allergies/diet/health goals/serving "
            "sizes, stated preferences, and any decisions or the current task. Be "
            "concise but do not drop concrete facts.\n\n"
            f"Existing summary:\n{self.summary or '(none)'}\n\n"
            f"New messages to fold in:\n{convo}\n\nUpdated summary:"
        )
        try:
            resp = _get_summarizer().invoke(prompt) if self._summarizer is None else self._summarizer.invoke(prompt)
            new_summary = (getattr(resp, "content", None) or "").strip()
            if new_summary:
                self.summary = new_summary
                self._messages = recent  # drop folded messages only on success
        except Exception:
            # Fail safe: if summarization fails, keep messages verbatim (never lose context).
            pass

    def clear(self) -> None:
        self._messages = []
        self.summary = ""


_SESSION_HISTORIES: dict[str, SummarizingChatMessageHistory] = {}


def _get_session_history(thread_id: str) -> SummarizingChatMessageHistory:
    """Return (creating if needed) the summarizing message history for a conversation."""
    if thread_id not in _SESSION_HISTORIES:
        _SESSION_HISTORIES[thread_id] = SummarizingChatMessageHistory()
    return _SESSION_HISTORIES[thread_id]


# --------------------------------------------------
# Grounding guard: never let a fabricated recipe ID reach the user
# --------------------------------------------------
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


def forget_last_turn(thread_id: str, n: int = 2) -> None:
    """Drop the last exchange (human+AI) from a thread's memory — used to purge a
    hallucinated answer so it never becomes remembered context."""
    h = _SESSION_HISTORIES.get(thread_id)
    msgs = getattr(h, "_messages", None)
    if msgs:
        h._messages = msgs[:-n] if len(msgs) >= n else []


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
                    out = [
                        f"Shopping list ({raw.get('item_count', len(raw['items']))} "
                        f"items from {raw.get('recipe_count', '?')} recipes):"
                    ]
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


def _handle_parsing_error(error) -> str:
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


def build_agent(username: str | None = None, model: str | None = None):
    """Build a native tool-calling agent (grounded in the recipe database).

    If ``username`` is given (a logged-in user), a bound save_user_profile tool
    is added so stated allergies/diet/goals can be persisted to the DB.
    ``model`` overrides GROQ_MODEL (used to build the fallback agent).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # temperature 0 => deterministic and reliable tool use
    llm = ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=float(getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(getenv("LLM_MAX_TOKENS", "2048")),
    )

    # Build tools with proper argument schemas (inferred from each tool's real
    # signature) so the model calls them with the right keys, while still
    # returning concise, factual text via _wrap_tool.
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

    # Always expose the profile-save tool (the prompt references it). For a
    # logged-in user it writes to the DB; for a guest it politely no-ops. This
    # avoids "tool call validation failed" when the model tries to call it.
    tools.append(_make_save_profile_tool(username))

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=_handle_parsing_error,
        max_iterations=int(getenv("AGENT_MAX_ITERATIONS", "6")),
    )

    # Wrap with per-conversation memory so the agent remembers earlier turns
    # (e.g. "I'm allergic to peanuts"). Keyed by the "thread_id" passed in
    # config={"configurable": {"thread_id": ...}}.
    return RunnableWithMessageHistory(
        executor,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Thread ID",
                description="Unique conversation identifier.",
                default="default",
                is_shared=True,
            ),
        ],
    )