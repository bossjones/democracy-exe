from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State dictionary for tracking agent interactions.

    Attributes:
        query: The original query text
        response: The current response text
        current_agent: Name of the currently active agent
        context: Additional context dictionary
    """
    query: str
    response: str
    current_agent: str
    context: dict[str, Any]


class Query(BaseModel):
    """Model representing an input query.

    Attributes:
        text: The query text
        context: Optional dictionary of additional context
    """
    text: str
    context: dict[str, Any] | None = Field(default_factory=dict)


class Response(BaseModel):
    """Model representing the response output.

    Attributes:
        text: The response text
    """
    text: str


def create_initial_state(query: Query) -> AgentState:
    """Create initial agent state from a query.

    Args:
        query: The input Query object

    Returns:
        AgentState with initialized values
    """
    return {
        "query": query.text,
        "response": "",
        "current_agent": "",
        "context": query.context
    }


def update_state(state: AgentState, key: str, value: Any) -> AgentState:
    """Update a specific key in the agent state.

    Args:
        state: Current agent state
        key: Key to update
        value: New value for the key

    Returns:
        New AgentState with updated value
    """
    new_state = state.copy()
    new_state[key] = value
    return new_state


def get_response(state: AgentState) -> Response:
    """Extract response from final state.

    Args:
        state: Final agent state

    Returns:
        Response object containing the response text
    """
    return Response(text=state["response"])


"""
This state.py file defines the structures and helper functions for managing the state in our AI system:
AgentState: A TypedDict that defines the structure of the state passed between agents (same as in base.py, but repeated here for convenience).
Query: A Pydantic model representing the input query, including optional context.
Response: A Pydantic model representing the output response.
create_initial_state(): A function to create the initial state from a Query object.
update_state(): A helper function to update a specific key in the state while maintaining immutability.
get_response(): A function to extract the response from the final state.
These structures and functions provide a consistent way to handle input, output, and state management throughout the AI system. The use of Pydantic models (Query and Response) allows for easy integration with FastAPI or other frameworks that support these models.
The helper functions make it easier to work with the state in a functional programming style, ensuring immutability and consistency.
"""
