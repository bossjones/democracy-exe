from __future__ import annotations

from typing import Any

from langgraph.graph import Graph

from democracy_exe.ai.agents.memory_agent import MemoryAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class MemoryGraph(BaseGraph):
    """Graph for orchestrating memory storage and retrieval operations.

    This graph manages the flow of memory operations through store and retrieve stages.
    It coordinates these operations using a directed graph structure where each node
    represents a specific memory operation.

    Attributes:
        memory_agent: Instance of MemoryAgent that performs the actual memory operations
    """

    def __init__(self) -> None:
        """Initialize the memory graph with its agent."""
        super().__init__()
        self.memory_agent = MemoryAgent()

    def build(self) -> Graph:
        """Construct the memory operations workflow graph.

        Creates a directed graph with nodes for memory operations and edges
        defining the flow between operations.

        Returns:
            Configured LangGraph Graph instance ready for execution
        """
        # Add memory agent node
        self.graph.add_node("memory", AgentNode(self.memory_agent))

        # Add edges for store and retrieve operations
        self.graph.add_edge("memory", "store")
        self.graph.add_edge("store", "retrieve")
        self.graph.add_edge("retrieve", "memory")

        # Set the entry point
        self.graph.set_entry_point("memory")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        """Process a memory operation through the workflow.

        Args:
            state: Current agent state containing the memory operation request

        Returns:
            Updated agent state with operation results
        """
        compiled_graph = self.compile()
        return compiled_graph(state)

    def store(self, key: str, value: Any) -> None:
        """Store a value in memory under the specified key.

        Args:
            key: The identifier to store the value under
            value: The value to store
        """
        self.memory_agent.store(key, value)

    def retrieve(self, key: str) -> Any:
        """Retrieve a value from memory by its key.

        Args:
            key: The identifier of the value to retrieve

        Returns:
            The stored value if found, None otherwise
        """
        return self.memory_agent.retrieve(key)


memory_graph = MemoryGraph()

"""
This memory_graph.py file defines the MemoryGraph class, which encapsulates the functionality for storing and retrieving information:
The MemoryGraph class inherits from BaseGraph and implements the build method.
It has a MemoryAgent instance to handle the actual storage and retrieval operations.
The build method constructs the graph:
It adds the memory agent node.
It adds edges for store and retrieve operations, creating a cycle that allows for multiple operations.
It sets the entry point of the graph to the memory node.
The process method compiles the graph and processes the given state.
The store and retrieve methods provide a simple interface to interact with the memory agent directly.
An instance of MemoryGraph is created at the module level for easy access.
This implementation allows for flexible memory operations within the AI system. The memory graph can be used independently or integrated into the larger router graph as needed. The cyclic nature of the graph allows for multiple store and retrieve operations to be performed in sequence if necessary.
"""
