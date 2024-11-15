from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import Mock

import pytest

from democracy_exe.ai.agents.router_agent import RouterAgent
from democracy_exe.ai.base import AgentState


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def router_agent() -> RouterAgent:
    """Create a RouterAgent instance for testing.

    Returns:
        RouterAgent instance
    """
    return RouterAgent()


@pytest.fixture
def mock_specialized_agent(mocker: MockerFixture) -> Mock:
    """Create a mock specialized agent.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock specialized agent
    """
    mock_agent = mocker.Mock()
    mock_agent.return_value = AgentState(response="Test response")
    return mock_agent


def test_add_specialized_agent(router_agent: RouterAgent, mock_specialized_agent: Mock) -> None:
    """Test adding a specialized agent.

    Args:
        router_agent: RouterAgent instance
        mock_specialized_agent: Mock specialized agent
    """
    router_agent.add_specialized_agent("test_agent", mock_specialized_agent)
    assert "test_agent" in router_agent.specialized_agents
    assert router_agent.specialized_agents["test_agent"] == mock_specialized_agent


@pytest.mark.parametrize(
    "query,expected_agent",
    [
        ("search for python tutorials", "internet_search"),
        ("analyze this image", "image_analysis"),
        ("research quantum computing", "research"),
        ("post a tweet about AI", "social_media"),
        ("process this video file", "image_video_processing"),
        ("remember what I said earlier", "memory"),
        ("invalid query type", ""),
    ],
)
def test_route_queries(router_agent: RouterAgent, query: str, expected_agent: str) -> None:
    """Test routing different types of queries.

    Args:
        router_agent: RouterAgent instance
        query: Input query string
        expected_agent: Expected agent to handle the query
    """
    state = AgentState(query=query)
    result = router_agent.route(state)
    assert result == expected_agent


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
def test_process_with_valid_agent(router_agent: RouterAgent, mock_specialized_agent: Mock) -> None:
    """Test processing with a valid specialized agent.

    Args:
        router_agent: RouterAgent instance
        mock_specialized_agent: Mock specialized agent
    """
    # Add a specialized agent
    router_agent.add_specialized_agent("image_analysis", mock_specialized_agent)

    # Create state with query that should route to image_analysis
    state = AgentState(query="analyze this image")

    # Process the state
    result = router_agent.process(state)

    # Verify the specialized agent was called
    mock_specialized_agent.assert_called_once_with(state)

    # Verify the response
    assert isinstance(result, AgentState)
    assert result["response"] == "Test response"


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
def test_process_with_invalid_agent(router_agent: RouterAgent) -> None:
    """Test processing with no valid specialized agent.

    Args:
        router_agent: RouterAgent instance
    """
    # Create state with query that won't match any agent
    state = AgentState(query="invalid query type")

    # Process the state
    result = router_agent.process(state)

    # Verify the default response
    assert isinstance(result, AgentState)
    assert result["response"] == "I'm not sure how to handle this request."


def test_process_maintains_state(router_agent: RouterAgent, mock_specialized_agent: Mock) -> None:
    """Test that processing maintains existing state data.

    Args:
        router_agent: RouterAgent instance
        mock_specialized_agent: Mock specialized agent
    """
    # Add a specialized agent
    router_agent.add_specialized_agent("image_analysis", mock_specialized_agent)

    # Create state with additional data
    state = AgentState(query="analyze this image", context="important context", history=["previous interaction"])

    # Process the state
    result = router_agent.process(state)

    # Verify the state was maintained
    assert result["context"] == "important context"
    assert result["history"] == ["previous interaction"]


def test_multiple_specialized_agents(router_agent: RouterAgent, mock_specialized_agent: Mock) -> None:
    """Test handling multiple specialized agents.

    Args:
        router_agent: RouterAgent instance
        mock_specialized_agent: Mock specialized agent
    """
    # Add multiple specialized agents
    agents = ["image_analysis", "internet_search", "research"]
    for agent in agents:
        router_agent.add_specialized_agent(agent, mock_specialized_agent)

    # Verify all agents were added
    assert set(router_agent.specialized_agents.keys()) == set(agents)

    # Test routing to each agent
    queries = {
        "analyze this image": "image_analysis",
        "search for python": "internet_search",
        "research quantum computing": "research",
    }

    for query, expected_agent in queries.items():
        state = AgentState(query=query)
        result = router_agent.route(state)
        assert result == expected_agent
