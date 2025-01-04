from __future__ import annotations

from typing import Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# SOURCE: lang-memgpt
class GraphConfig(TypedDict):
    model: str | None
    """The model to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""

# SOURCE: lang-memgpt
# Define the schema for the state maintained throughout the conversation
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: list[str]
    """The core memories associated with the user."""
    recall_memories: list[str]
    """The recall memories retrieved for the current context."""


__all__ = [
    "State",
    "GraphConfig",
]
