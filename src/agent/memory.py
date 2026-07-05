"""Per-session chat memory (summary-buffer, to cap token growth).

Keeps the most recent turns verbatim; folds older turns into a running summary
that preserves the key facts (recipes + IDs, allergies/diet/goals, decisions).
"""
from __future__ import annotations

from os import getenv
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage

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


def get_session_history(thread_id: str) -> SummarizingChatMessageHistory:
    """Return (creating if needed) the summarizing message history for a conversation."""
    if thread_id not in _SESSION_HISTORIES:
        _SESSION_HISTORIES[thread_id] = SummarizingChatMessageHistory()
    return _SESSION_HISTORIES[thread_id]


def forget_last_turn(thread_id: str, n: int = 2) -> None:
    """Drop the last exchange (human+AI) from a thread's memory — used to purge a
    hallucinated answer so it never becomes remembered context."""
    h = _SESSION_HISTORIES.get(thread_id)
    msgs = getattr(h, "_messages", None)
    if msgs:
        h._messages = msgs[:-n] if len(msgs) >= n else []