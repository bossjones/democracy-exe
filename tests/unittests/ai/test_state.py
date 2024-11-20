from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import pytest

from democracy_exe.ai.state import AgentState, Query, Response, create_initial_state, get_response, update_state


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sample_query() -> Query:
    """Create a sample query for testing.

    Returns:
        Query object with test data
    """
    return Query(text="test query", context={"key": "value"})


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing.

    Returns:
        AgentState with test data
    """
    return {
        "query": "test query",
        "response": "",
        "current_agent": "agent1",
        "context": {},
    }


def test_create_initial_state(sample_query: Query) -> None:
    """Test create_initial_state function.

    Args:
        sample_query: Fixture providing test query
    """
    state = create_initial_state(sample_query)

    assert state["query"] == "test query"
    assert state["response"] == ""
    assert state["current_agent"] == ""
    assert state["context"] == {"key": "value"}


def test_create_initial_state_no_context() -> None:
    """Test create_initial_state with no context."""
    query = Query(text="test query")
    state = create_initial_state(query)

    assert state["query"] == "test query"
    assert state["response"] == ""
    assert state["current_agent"] == ""
    assert state["context"] == {}


def test_update_state(sample_state: AgentState) -> None:
    """Test update_state function.

    Args:
        sample_state: Fixture providing test state
    """
    new_state = update_state(sample_state, "response", "updated response")

    assert new_state["query"] == "test query"
    assert new_state["response"] == "updated response"
    assert new_state["current_agent"] == "agent1"
    assert new_state["context"] == {}

    # Original state should be unchanged
    assert sample_state["response"] == ""


def test_update_state_context(sample_state: AgentState) -> None:
    """Test update_state with context.

    Args:
        sample_state: Fixture providing test state
    """
    new_context: dict[str, str] = {"key1": "updated_value", "key2": "value2"}
    new_state = update_state(sample_state, "context", new_context)

    assert new_state["query"] == "test query"
    assert new_state["response"] == ""
    assert new_state["current_agent"] == "agent1"
    assert new_state["context"] == new_context

    # Original state should be unchanged
    assert sample_state["context"] == {}


def test_get_response(sample_state: AgentState) -> None:
    """Test get_response function.

    Args:
        sample_state: Fixture providing test state
    """
    # Update the test state with a response
    state = update_state(sample_state, "response", "test response")
    response = get_response(state)

    assert isinstance(response, Response)
    assert response.text == "test response"
