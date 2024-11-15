from __future__ import annotations

from typing import Any

import pytest

from democracy_exe.ai.base import AgentNode, AgentState, BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def process(self, state: AgentState) -> AgentState:
        """Process the state by updating the response."""
        return {**state, "response": f"Processed by {self.__class__.__name__}"}


def test_agent_node_call() -> None:
    """Test that AgentNode processes state through the wrapped agent."""
    agent = MockAgent()
    node = AgentNode(agent)

    state: AgentState = {
        "query": "test query",
        "response": "",
        "current_agent": "mock_agent",
        "context": {},
    }

    updated_state: AgentState = node(state)

    assert updated_state["response"] == "Processed by MockAgent"


def test_agent_node_process_state_unchanged() -> None:
    """Test that AgentNode does not modify the input state object."""
    agent = MockAgent()
    node = AgentNode(agent)

    state: AgentState = {
        "query": "test query",
        "response": "",
        "current_agent": "mock_agent",
        "context": {},
    }

    updated_state = node(state)

    assert updated_state is not state
    assert state["response"] == ""


def test_agent_node_process_context_passed() -> None:
    """Test that AgentNode passes context to the wrapped agent."""

    class ContextAgent(BaseAgent):
        """Agent that checks for context in state."""

        def process(self, state: AgentState) -> AgentState:
            assert "key" in state["context"]
            assert state["context"]["key"] == "value"
            return state

    agent = ContextAgent()
    node = AgentNode(agent)

    state: AgentState = {
        "query": "test query",
        "response": "",
        "current_agent": "context_agent",
        "context": {"key": "value"},
    }

    node(state)


def test_agent_node_no_state_mutation() -> None:
    """Test that AgentNode preserves input state immutability."""

    class MutatingAgent(BaseAgent):
        """Agent that attempts to mutate state."""

        def process(self, state: AgentState) -> AgentState:
            # Mutate input state (bad practice)
            state["response"] = "mutated"
            return state

    agent = MutatingAgent()
    node = AgentNode(agent)

    state: AgentState = {
        "query": "test query",
        "response": "",
        "current_agent": "mutating_agent",
        "context": {},
    }

    # Even though agent mutates state internally,
    # the original state should remain unchanged
    result = node(state)
    assert state["response"] == ""
    assert result["response"] == "mutated"
    assert result is not state
