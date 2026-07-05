"""Build the native tool-calling agent, grounded in the recipe database.

This is a native tool-calling agent (not a text-ReAct agent): gpt-oss models on
Groq use native tool-calling in a way incompatible with LangChain's text ReAct
agent, and text ReAct with Llama loops/hallucinates. We use create_tool_calling_agent
with a Llama chat model, overridable via GROQ_MODEL.
"""
from __future__ import annotations

from os import getenv

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agent.prompt import SYSTEM_PROMPT
from src.agent.memory import get_session_history
from src.agent.tool_binding import build_tools, handle_parsing_error

DEFAULT_MODEL = getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Model used automatically when the main model is rate-limited or overloaded.
FALLBACK_MODEL = getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")


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

    tools = build_tools(username)

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=handle_parsing_error,
        max_iterations=int(getenv("AGENT_MAX_ITERATIONS", "6")),
    )

    # Wrap with per-conversation memory so the agent remembers earlier turns
    # (e.g. "I'm allergic to peanuts"). Keyed by the "thread_id" passed in
    # config={"configurable": {"thread_id": ...}}.
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
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