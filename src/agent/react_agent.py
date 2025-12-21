# from __future__ import annotations
#
# from pathlib import Path
# from typing import List
#
# from dotenv import load_dotenv
# load_dotenv()
#
# from langchain_groq import ChatGroq
# from langchain_core.tools import StructuredTool
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver
#
# # Grounded tool callables
# from src.tools.recipe_lookup import recipe_lookup
# from src.tools.ingredient_suggester import ingredient_suggester
# from src.tools.nutrition import nutrition_analyzer
# from src.tools.meal_planner import meal_planner
# from src.tools.shopping_list import shopping_list
# from src.tools.recipe_instructions import recipe_instructions
# from src.tools.recipe_resolver import resolve_recipe_by_name
#
# PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "react_agent.txt"
#
#
# def build_agent():
#     """
#     Returns a LangChain agent runnable created via langchain.agents.create_agent.
#     Conversation memory is handled by the checkpointer (thread_id-based).
#     """
#     system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
#
#     llm = ChatGroq(
#         model="llama-3.1-8b-instant",
#         temperature=0,
#     )
#
#     tools: List[StructuredTool] = [
#         StructuredTool.from_function(
#             func=recipe_lookup,
#             name="recipe_lookup",
#             description="Find recipes via semantic search. Use this before suggesting recipes when user did NOT give explicit ingredients.",
#         ),
#         StructuredTool.from_function(
#             func=ingredient_suggester,
#             name="ingredient_suggester",
#             description="Suggest recipes that match a list of user-provided ingredients. Use only when user lists ingredients they have.",
#         ),
#         StructuredTool.from_function(
#             func=nutrition_analyzer,
#             name="nutrition_analyzer",
#             description="Compute nutrition totals for one or more recipe_ids. Use only with IDs returned by tools.",
#         ),
#         StructuredTool.from_function(
#             func=meal_planner,
#             name="meal_planner",
#             description="Create a multi-day plan from candidate_recipe_ids. Requires days + candidate_recipe_ids.",
#         ),
#         StructuredTool.from_function(
#             func=shopping_list,
#             name="shopping_list",
#             description="Generate a shopping list from a meal plan's days list (meal_planner output).",
#         ),
#         StructuredTool.from_function(
#             recipe_instructions,
#             name="recipe_instructions",
#             description="Return grounded cooking instructions for a recipe.."
#         ),
#         StructuredTool.from_function(
#             resolve_recipe_by_name,
#             name="resolve_recipe_by_name",
#             description="Resolve a recipe name to a recipe_id before fetching details."
#         ),
#
#     ]
#
#     # checkpointer enables memory across turns when you pass config thread_id
#     agent = create_agent(
#         model=llm,
#         tools=tools,
#         system_prompt=system_prompt,
#         checkpointer=InMemorySaver(),
#     )
#
#     return agent
# src/tools/types.py
from __future__ import annotations

from pathlib import Path
from typing import List
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.tools.registry import list_tools
from src.tools.types import coerce_tool_result

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "react_agent.txt"


def _wrap_tool(fn, tool_name: str):
    """
    Wrap tool callables so they:
    - never crash the agent loop
    - always return a normalized ToolResult dict
    """
    def _wrapped(**kwargs):
        start = time.time()
        try:
            raw = fn(**kwargs)
        except Exception as e:
            raw = {
                "status": "failure",
                "assumptions": [
                    f"Tool '{tool_name}' raised {type(e).__name__}: {e}"
                ],
            }

        result = coerce_tool_result(tool_name, raw)

        # lightweight debug info (safe to ignore)
        elapsed_ms = int((time.time() - start) * 1000)
        result.data.setdefault("_debug", {})
        result.data["_debug"]["latency_ms"] = elapsed_ms

        return result.model_dump()

    return _wrapped


def build_agent():
    """
    Returns a LangChain agent runnable created via create_agent.
    Tool behavior is now standardized and crash-safe.
    """
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    tools: List[StructuredTool] = []

    # SINGLE SOURCE OF TRUTH: registry
    for spec in list_tools().values():
        tools.append(
            StructuredTool.from_function(
                func=_wrap_tool(spec.callable, spec.name),
                name=spec.name,
                description=spec.description,
            )
        )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent
