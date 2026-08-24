"""Shared parsing helpers for tools."""
from __future__ import annotations

import json
from typing import Any, List


def parse_json_list(value: Any) -> List[str]:
    """Parse a JSON-encoded list of strings; tolerate bad/empty data."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []
