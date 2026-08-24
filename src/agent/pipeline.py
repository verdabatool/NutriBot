"""Shared grounded chat pipeline.

One place that turns user text into a clean, grounded answer — wrapping the
agent invoke with automatic model fallback (on rate-limit / oversized-request
errors), the grounding guard (purge + retry when a reply cites fabricated recipe
IDs), output sanitization, and optional truncation completion.

The Streamlit UI, the CLI, and the eval harness all go through this so they
share one behaviour instead of each re-implementing the guard and fallback.
"""
from __future__ import annotations

import re
import traceback
from os import getenv
from typing import Callable, List, Optional

from langchain_groq import ChatGroq

from src.agent.agent import build_agent, FALLBACK_MODEL
from src.agent.grounding import ungrounded_ids, allergen_hits
from src.agent.memory import forget_last_turn

# Trim very long input to reduce token usage (avoid Groq 413 request-too-large).
MAX_INPUT_CHARS = 600

# Model used to complete a cut-off answer (falls back to the light model).
_COMPLETION_MODEL = getenv("GROQ_COMPLETION_MODEL", "llama-3.3-70b-versatile")


# --------------------------------------------------
# Error classification
# --------------------------------------------------
# We classify a failed agent invoke into a recovery bucket so the pipeline knows
# whether to fall back to the lighter model, report a DB-busy message, or give a
# generic apology. We prefer the exception TYPE / HTTP status over message text:
# groq raises typed exceptions (RateLimitError == 429; oversized requests arrive
# as an APIStatusError with status_code 413), which is far more robust than
# grepping the message string. A string heuristic remains only as a last resort
# for errors that reach us without a usable type/status (e.g. wrapped upstream).
#
# (We keep the fallback at the pipeline level, rebuilding a whole agent on the
# lighter model, rather than LangChain's llm.with_fallbacks: with_fallbacks does
# not compose with create_tool_calling_agent's internal bind_tools, we only want
# to fall back on specific errors, and rebuilding preserves per-thread memory.)


def _iter_causes(exc: BaseException):
    """Yield the exception and its __cause__/__context__ chain (deduped)."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _status_code(exc: BaseException):
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def classify_error(exc: BaseException) -> Optional[str]:
    """Bucket an invoke failure as 'rate_limit', 'overflow', 'model_unavailable',
    'locked', or None."""
    try:
        import groq
    except Exception:
        groq = None

    for e in _iter_causes(exc):
        code = _status_code(e)
        if code == 429 or (groq is not None and isinstance(e, groq.RateLimitError)):
            return "rate_limit"
        if code == 413:
            return "overflow"
        # A 404 means the configured model id is gone or off-limits (e.g. Groq
        # decommissioned it) — recover by falling back to the light model rather
        # than failing every message with a generic apology.
        if code == 404 or (groq is not None and isinstance(e, groq.NotFoundError)):
            return "model_unavailable"
        # SQLite has no distinct "locked" exception subclass — it is an
        # OperationalError identified by its message.
        if type(e).__name__ == "OperationalError" and "locked" in str(e).lower():
            return "locked"

    # Last-resort heuristic for errors that arrived without a type/status.
    msg = str(exc).lower()
    if any(t in msg for t in ("rate limit", "rate_limit", "429", "tokens per day", "tpd", "quota")):
        return "rate_limit"
    if any(t in msg for t in ("413", "request too large", "too large")):
        return "overflow"
    if any(t in msg for t in ("model_not_found", "does not exist or you do not have access", "404")):
        return "model_unavailable"
    if "locked" in msg:
        return "locked"
    return None


def _extract_output(result) -> str:
    """Pull the assistant text out of an agent-executor result."""
    if isinstance(result, dict) and "output" in result:
        return result["output"]
    if isinstance(result, dict) and result.get("messages"):
        last = result["messages"][-1]
        return getattr(last, "content", None) or getattr(last, "text", None) or str(last)
    return str(result)


# --------------------------------------------------
# Output sanitization (strip agent scaffolding / leaked tool markup)
# --------------------------------------------------
def clean_response(text: str) -> str:
    """Remove parser noise and troubleshooting footer; dedupe lines."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"(?i)could not parse LLM output:\s*", "", text)
    text = re.sub(r"(?i)output_parsing_failure.*", "", text)
    text = re.split(r"For troubleshooting", text, flags=re.IGNORECASE)[0]
    lines = [ln.rstrip() for ln in text.split("\n")]
    out, seen = [], set()
    for ln in lines:
        key = ln.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(ln)
    return "\n".join(out).strip()


def sanitize_answer(text: str) -> str:
    """Keep only the user-facing content; strip Thought/Action and tool/parser errors."""
    s = str(text or "")
    m = re.search(r"(?:\*\*Final Answer:\*\*|Final Answer:)\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    else:
        # Remove internal ReAct scaffolding lines
        s = re.sub(r"(?m)^\s*Thought:\s.*$", "", s)
        s = re.sub(r"(?m)^\s*Action:\s.*$", "", s)
        s = re.sub(r"(?m)^\s*Action Input:\s.*$", "", s)
        # Remove tool error leakage and parser noise
        s = re.sub(r"(?i)resolve_recipe_by_name is not a valid tool, try one of \[.*?\]\.?", "", s)
        s = re.sub(r"(?i)could not parse LLM output:\s*", "", s)
        s = re.sub(r"(?i)sorry, I couldn't parse the response\. please try again\.", "", s)
        # Strip fenced code blocks that sometimes leak
        s = re.sub(r"```+.*?```+", "", s, flags=re.DOTALL)

    # Strip leaked pseudo tool-calls a model may emit as plain text, e.g.
    #   <nutrition_analyzer>{"recipe_ids": [15253]}</nutrition_analyzer>
    #   <function=...>...</function> / <tool_call>...</tool_call>
    s = re.sub(r"<([a-zA-Z_][\w\-]*)>\s*\{.*?\}\s*</\1>", "", s, flags=re.DOTALL)
    s = re.sub(r"</?(function|tool|tool_call|tool_calls|invoke|parameter)\b[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|[^>]*\|>", "", s)

    result = clean_response(s)
    if not result.strip():
        return ("I couldn't complete that one just now — please try again, or "
                "rephrase your question (e.g. name the dish or its recipe ID).")
    return result


# --------------------------------------------------
# Truncation completion (gracefully finish a cut-off answer)
# --------------------------------------------------
def is_truncated(text: str) -> bool:
    """Heuristic to detect mid-sentence/list cutoffs."""
    if not text or len(text) < 60:
        return False
    tail = text.strip()[-120:]
    if tail.endswith((":", "-", "–", "—", "•")):
        return True
    if not text.strip().endswith((".", "!", "?")) and len(text) > 200:
        return True
    # common half-step range like "Blend 30-45"
    if re.search(r"\b\d+\s*[–-]\s*\d+$", tail):
        return True
    return False


def _finish_answer(partial: str) -> str:
    """Complete the current sentence/list item without adding new sections."""
    try:
        llm = ChatGroq(model=_COMPLETION_MODEL, temperature=0.2, max_tokens=256)
    except Exception:
        llm = ChatGroq(model=FALLBACK_MODEL, temperature=0.2, max_tokens=256)
    prompt = (
        "Continue the assistant's answer exactly from where it stopped. "
        "Finish the current sentence or list item, keep the same tone, and do not start new sections. "
        "Do not introduce new facts; only complete what's already begun. "
        "Output only the continuation.\n\n"
        f"Answer so far:\n{partial.strip()}\n\nContinuation:"
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", None) or getattr(resp, "text", None) or ""


# --------------------------------------------------
# The pipeline
# --------------------------------------------------
# The retry nudge appended when the grounding guard fires.
_GUARD_RETRY_NUDGE = (
    "\n\n[IMPORTANT: Do NOT answer from memory. Call recipe_lookup and use ONLY "
    "the real recipe names and IDs it returns. Never invent IDs.]"
)

# LangChain's sentinel when the agent loops without producing a final answer.
_ITER_LIMIT_SENTINEL = "stopped due to max iterations"

# Parse the user's allergens out of the USER CONTEXT string the UI/eval passes,
# e.g. "ALLERGIES (never suggest these): peanuts, shellfish | Diet: vegan".
_ALLERGY_RE = re.compile(r"allerg\w*[^:]*:\s*([^|]+)", re.IGNORECASE)


def parse_allergens(user_preference: str) -> List[str]:
    """Extract the allergen list from the USER CONTEXT string (empty if none)."""
    m = _ALLERGY_RE.search(user_preference or "")
    if not m:
        return []
    return [a.strip() for a in re.split(r"[,;/]", m.group(1)) if a.strip()]


class ChatPipeline:
    """Turns user text into a clean, grounded answer.

    Build once (optionally bound to a logged-in ``username`` and/or a specific
    ``model``) and reuse across turns, passing a ``thread_id`` per conversation.
    """

    def __init__(self, username: str | None = None, model: str | None = None):
        self.username = username
        self.model = model
        self.agent = build_agent(username=username, model=model)
        self._fallback = None  # built lazily on first rate-limit/overflow

    def _fallback_agent(self):
        if self._fallback is None:
            self._fallback = build_agent(username=self.username, model=FALLBACK_MODEL)
        return self._fallback

    def _invoke(self, agent, text: str, thread_id: str, user_preference: str, callbacks):
        config: dict = {"configurable": {"thread_id": thread_id}}
        if callbacks:
            config["callbacks"] = callbacks
        result = agent.invoke({"input": text, "user_preference": user_preference}, config=config)
        return _extract_output(result)

    def answer(
        self,
        user_text: str,
        thread_id: str = "default",
        user_preference: str = "No preference",
        *,
        guard: bool = True,
        complete_truncated: bool = False,
        callbacks: Optional[List] = None,
        on_status: Optional[Callable[[str], None]] = None,
        raise_errors: bool = False,
    ) -> str:
        """Produce a grounded answer.

        guard              — purge + retry once if the reply cites fabricated IDs.
        complete_truncated — make a follow-up call to finish a cut-off answer.
        callbacks          — LangChain callbacks passed to invoke (e.g. a tool recorder).
        on_status          — optional status reporter (e.g. a Streamlit spinner setter).
        raise_errors       — re-raise unrecoverable errors instead of returning a
                             friendly message (the eval harness uses this to record them).
        """
        def status(msg: str):
            if on_status:
                on_status(msg)

        text = (user_text or "")[:MAX_INPUT_CHARS]

        # --- Primary invoke, with automatic fallback on rate-limit / overflow ---
        try:
            status("Cooking up an answer…")
            output = self._invoke(self.agent, text, thread_id, user_preference, callbacks)
        except Exception as e:
            traceback.print_exc()
            kind = classify_error(e)
            if kind in ("rate_limit", "overflow", "model_unavailable"):
                # The lighter fallback model handles a rate limit, a request too
                # large for the primary model, or a primary model that is gone /
                # off-limits (404 — e.g. Groq decommissioned the configured id).
                try:
                    note = ("Main model unavailable" if kind == "model_unavailable"
                            else "Main model busy")
                    status(f"{note} — retrying on {FALLBACK_MODEL}…")
                    output = self._invoke(
                        self._fallback_agent(), text, thread_id, user_preference, callbacks
                    )
                except Exception:
                    traceback.print_exc()
                    if raise_errors:
                        raise
                    return ("⏳ **The main model is unavailable** and the fallback model "
                            "is also unreachable right now. Please wait a few minutes and "
                            "try again.")
            elif kind == "locked":
                if raise_errors:
                    raise
                return ("🔒 The database is busy — it may be open in another tool "
                        "(like DB Browser for SQLite). Please close it and try again.")
            else:
                if raise_errors:
                    raise
                return ("Sorry, I hit a snag with that one. Could you rephrase it — "
                        "or, if you meant one of the options above, tell me the recipe "
                        "name or its number?")

        # --- Grounding guard: never let fabricated recipe IDs reach the user ---
        if guard and ungrounded_ids(output):
            output = self._run_guard(text, thread_id, user_preference, callbacks, status)

        # --- Allergen guard: never let a cited recipe contain a known allergen.
        # This is deterministic (checked against the DB), so safety does not depend
        # on the model remembering to pass exclude=[...] to a tool.
        allergens = parse_allergens(user_preference)
        if guard and allergens and allergen_hits(output, allergens):
            output = self._run_allergen_guard(
                text, thread_id, user_preference, allergens, callbacks, status
            )

        # --- Never surface LangChain's raw "max iterations" sentinel ---
        if _ITER_LIMIT_SENTINEL in (output or "").lower():
            return ("I couldn't pin that down in a few tries. Could you narrow it a "
                    "little — name the dish or give its recipe ID — and I'll get you a "
                    "grounded answer?")

        answer = sanitize_answer(output)

        # --- Gracefully finish cut-off responses ---
        if complete_truncated and is_truncated(answer):
            try:
                cont = _finish_answer(answer)
                if cont.strip():
                    answer = sanitize_answer(answer + "\n" + cont.strip())
            except Exception:
                traceback.print_exc()

        return answer

    def _run_guard(self, text, thread_id, user_preference, callbacks, status) -> str:
        """The model fabricated recipe IDs (answered from memory instead of the DB):
        purge that turn from memory and retry once, forcing a database lookup."""
        forget_last_turn(thread_id)
        try:
            status("Verifying against the database…")
            retry = self._invoke(
                self.agent, text + _GUARD_RETRY_NUDGE, thread_id, user_preference, callbacks
            )
            if not ungrounded_ids(retry):
                return retry
            forget_last_turn(thread_id)
            return ("I caught myself about to answer from memory instead of the "
                    "database. Please ask again (e.g. “quick pasta recipes”) and "
                    "I'll pull real, grounded recipes.")
        except Exception:
            traceback.print_exc()
            return ("I caught myself about to answer from memory instead of the "
                    "database. Please ask again and I'll pull grounded recipes.")

    def _run_allergen_guard(self, text, thread_id, user_preference, allergens,
                            callbacks, status) -> str:
        """A cited recipe actually contains one of the user's allergens: purge that
        turn and retry once, forcing the model to exclude the allergens. If the
        retry is still unsafe, refuse rather than serve an allergen."""
        alg = ", ".join(allergens)
        forget_last_turn(thread_id)
        nudge = (
            f"\n\n[SAFETY: the user is ALLERGIC to {alg}. Do NOT recommend any recipe "
            f"containing these. Call recipe_lookup with exclude={allergens} and "
            f"suggest ONLY recipes whose ingredients are free of them. If the request "
            f"itself asks for one of these ingredients, briefly decline and offer an "
            f"allergen-free alternative instead — do not keep searching.]"
        )
        try:
            status("Double-checking this is allergy-safe…")
            retry = self._invoke(self.agent, text + nudge, thread_id, user_preference, callbacks)
            if _ITER_LIMIT_SENTINEL in (retry or "").lower():
                raise RuntimeError("iteration limit on allergen retry")
            if not allergen_hits(retry, allergens):
                return retry
            forget_last_turn(thread_id)
        except Exception:
            traceback.print_exc()
        return (f"I can't recommend that — the options I found contain {alg}, which "
                f"you're allergic to. Want me to look for {alg}-free alternatives "
                f"instead?")