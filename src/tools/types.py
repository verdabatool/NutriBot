# src/tools/types.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ToolStatus = Literal["success", "partial", "failure"]


class ToolResult(BaseModel):
    status: ToolStatus = "success"
    data: Dict[str, Any] = Field(default_factory=dict)

    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    tool: Optional[str] = None


def coerce_tool_result(tool_name: str, raw: Any) -> ToolResult:
    """
    Backward-compatible normalization for all tool outputs.
    Existing tools returning dicts will keep working.
    """
    if isinstance(raw, ToolResult):
        result = raw

    elif isinstance(raw, dict):
        assumptions = raw.get("assumptions") or []
        warnings = raw.get("warnings") or []
        status = raw.get("status", "success")

        data = {
            k: v
            for k, v in raw.items()
            if k not in ("assumptions", "warnings", "status", "tool")
        }

        result = ToolResult(
            status=status,
            data=data,
            assumptions=assumptions,
            warnings=warnings,
        )
    else:
        result = ToolResult(
            status="failure",
            data={},
            assumptions=[f"Unsupported tool return type: {type(raw)}"],
        )

    result.tool = tool_name
    return result
