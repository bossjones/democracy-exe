"""Main Discord bot module.

This module serves as the entry point for the Discord bot functionality.
It imports and uses the modular components defined in the submodules.
"""
from __future__ import annotations

import pathlib

from democracy_exe.chatbot.core import DemocracyBot


# Path to this module's directory
HERE = pathlib.Path(__file__).parent

__all__ = ["DemocracyBot"]
