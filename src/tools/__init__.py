"""
Tool package.

Importing this package imports every tool module, which runs their
``register_tool(...)`` calls at import time and populates the registry.
Without these imports the registry stays empty and the agent is built
with zero tools.
"""

from src.tools import (  # noqa: F401
    recipe_lookup,
    recipe_resolver,
    recipe_instructions,
    recipe_modifier,
    nutrition,
    ingredient_suggester,
    meal_planner,
    shopping_list,
)