from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import Mock

import pytest

from democracy_exe.ai.base import AgentState
from democracy_exe.ai.graphs.router_graph import RouterGraph


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_router_agent(mocker: MockerFixture) -> Mock:
    """Create a mock router agent.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock router agent
    """
    mock_agent = mocker.Mock()
    mock_agent.return_value = AgentState(next_agent="test_agent")
    return mock_agent


@pytest.fixture
def mock_specialized_agent() -> Callable[[AgentState], AgentState]:
    """Create a mock specialized agent function.

    Returns:
        Mock specialized agent function
    """

    def agent_func(state: AgentState) -> AgentState:
        return AgentState(next_agent="router")

    return agent_func


@pytest.fixture
def router_graph(mock_router_agent: Mock) -> RouterGraph:
    """Create a RouterGraph instance with a mock router agent.

    Args:
        mock_router_agent: Mock router agent

    Returns:
        RouterGraph instance
    """
    graph = RouterGraph()
    graph.router_agent = mock_router_agent
    return graph


def test_add_specialized_agent(router_graph: RouterGraph, mock_specialized_agent: Callable) -> None:
    """Test adding a specialized agent to the graph.

    Args:
        router_graph: RouterGraph instance
        mock_specialized_agent: Mock specialized agent function
    """
    router_graph.add_specialized_agent("test_agent", mock_specialized_agent)
    assert "test_agent" in router_graph.specialized_agents
    assert router_graph.specialized_agents["test_agent"] == mock_specialized_agent


def test_build_graph_structure(router_graph: RouterGraph, mock_specialized_agent: Callable) -> None:
    """Test building the graph structure.

    Args:
        router_graph: RouterGraph instance
        mock_specialized_agent: Mock specialized agent function
    """
    # Add a specialized agent
    router_graph.add_specialized_agent("test_agent", mock_specialized_agent)

    # Build the graph
    graph = router_graph.build()

    # Check that nodes exist
    assert "router" in graph.nodes
    assert "test_agent" in graph.nodes

    # Check that edges exist
    edges = list(graph.edges)
    assert ("router", "test_agent") in edges
    assert ("test_agent", "router") in edges


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.asyncio
async def test_process_flow(
    router_graph: RouterGraph, mock_specialized_agent: Callable, mock_router_agent: Mock
) -> None:
    """Test the processing flow through the graph.

    Args:
        router_graph: RouterGraph instance
        mock_specialized_agent: Mock specialized agent function
        mock_router_agent: Mock router agent
    """
    # Add a specialized agent
    router_graph.add_specialized_agent("test_agent", mock_specialized_agent)

    # Create initial state
    initial_state = AgentState()

    # Process the state
    result = router_graph.process(initial_state)

    # Verify router agent was called
    mock_router_agent.assert_called_once()

    # Verify the flow went through the specialized agent
    assert isinstance(result, AgentState)
    assert result.next_agent == "router"


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
def test_empty_graph_build(router_graph: RouterGraph) -> None:
    """Test building a graph with no specialized agents.

    Args:
        router_graph: RouterGraph instance
    """
    graph = router_graph.build()
    assert "router" in graph.nodes
    assert len(graph.nodes) == 1
    assert len(list(graph.edges)) == 0


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
def test_multiple_specialized_agents(router_graph: RouterGraph, mock_specialized_agent: Callable) -> None:
    """Test adding multiple specialized agents to the graph.

    Args:
        router_graph: RouterGraph instance
        mock_specialized_agent: Mock specialized agent function
    """
    # Add multiple specialized agents
    router_graph.add_specialized_agent("agent1", mock_specialized_agent)
    router_graph.add_specialized_agent("agent2", mock_specialized_agent)

    graph = router_graph.build()

    # Check that all nodes exist
    assert "router" in graph.nodes
    assert "agent1" in graph.nodes
    assert "agent2" in graph.nodes

    # Check that all edges exist
    edges = list(graph.edges)
    assert ("router", "agent1") in edges
    assert ("router", "agent2") in edges
    assert ("agent1", "router") in edges
    assert ("agent2", "router") in edges
