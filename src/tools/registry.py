# src/tools/registry.py
from dataclasses import dataclass
from typing import Callable, Dict, Literal

ToolKind = Literal[
    "retrieval",
    "resolver",
    "calculation",
    "planning",
    "aggregation",
    "presentation",
]


@dataclass
class ToolSpec:
    name: str
    description: str
    callable: Callable
    kind: ToolKind = "retrieval"


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(tool: ToolSpec):
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool.name}' already registered")
    TOOL_REGISTRY[tool.name] = tool


def list_tools() -> Dict[str, ToolSpec]:
    return TOOL_REGISTRY
