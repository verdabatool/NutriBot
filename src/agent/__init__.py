"""Grounded tool-calling agent package.

Public API:
  - ChatPipeline      : the shared invoke → fallback → grounding-guard flow.
  - build_agent       : build the raw tool-calling agent.
  - ungrounded_ids    : detect fabricated recipe IDs in a reply.
  - forget_last_turn  : drop the last exchange from a thread's memory.
  - DEFAULT_MODEL / FALLBACK_MODEL : configured Groq model ids.
"""
from src.agent.agent import build_agent, DEFAULT_MODEL, FALLBACK_MODEL
from src.agent.grounding import ungrounded_ids, allergen_hits
from src.agent.memory import forget_last_turn
from src.agent.pipeline import ChatPipeline

__all__ = [
    "ChatPipeline",
    "build_agent",
    "ungrounded_ids",
    "allergen_hits",
    "forget_last_turn",
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
]