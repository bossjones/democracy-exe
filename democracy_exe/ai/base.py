from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

from langgraph.graph import Graph
from langgraph.graph.state import CompiledStateGraph


T = TypeVar('T', bound='BaseAgent')
G = TypeVar('G', bound='BaseGraph')

class AgentState(TypedDict):
    """State dictionary passed between agents in the graph.

    Attributes:
        query: The input query or command
        response: The generated response
        current_agent: Name of the currently active agent
        context: Additional contextual information
    """
    query: str
    response: str
    current_agent: str
    context: dict[str, Any]

class BaseAgent(ABC):
    """Abstract base class for all agents in the system.

    All concrete agent implementations must inherit from this class
    and implement the process method.
    """

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state.

        Args:
            state: Current agent state containing query and context

        Returns:
            Updated agent state after processing
        """
        pass

class BaseGraph(ABC):
    """Abstract base class for building agent graphs.

    Provides common interface for constructing and compiling
    graphs of connected agents.
    """

    def __init__(self) -> None:
        """Initialize the graph."""
        self.graph: Graph = Graph()

    @abstractmethod
    def build(self) -> Graph:
        """Build and return the configured graph.

        Returns:
            Constructed graph ready for compilation
        """
        pass

    def compile(self) -> Callable[[AgentState], AgentState]:
        """Compile the built graph.

        Returns:
            Compiled graph ready for execution
        """
        graph = self.build()
        return lambda state: graph.compile()(state)

class AgentNode:
    """Wrapper class to use agents as nodes in the graph."""

    def __init__(self, agent: BaseAgent) -> None:
        """Initialize agent node.

        Args:
            agent: The agent to wrap as a node
        """
        self.agent = agent

    def __call__(self, state: AgentState) -> AgentState:
        """Process state through the wrapped agent.

        Args:
            state: Current agent state

        Returns:
            Updated agent state after processing
        """
        return self.agent.process(state)

def conditional_edge(state: AgentState) -> str:
    """Determine next agent based on current state.

    Args:
        state: Current agent state

    Returns:
        Name of the next agent to route to
    """
    return state["current_agent"]



"""
This base.py file defines the core structures and abstract classes that will be used throughout the AI module:
AgentState: A TypedDict that defines the structure of the state passed between agents.
BaseAgent: An abstract base class that all agents should inherit from, defining the process method.
BaseGraph: An abstract base class for graphs, providing a common interface for building and compiling graphs.
AgentNode: A wrapper class that allows agents to be used as nodes in the graph.
conditional_edge: A function used for conditional routing in the graph based on the current agent.
"""
