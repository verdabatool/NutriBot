# src/tools/registry.py
from dataclasses import dataclass, field
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
    version: str = "1.0"
    tags: Dict[str, str] = field(default_factory=dict)


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(tool: ToolSpec):
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool.name}' already registered")
    TOOL_REGISTRY[tool.name] = tool


def list_tools() -> Dict[str, ToolSpec]:
    return TOOL_REGISTRY
