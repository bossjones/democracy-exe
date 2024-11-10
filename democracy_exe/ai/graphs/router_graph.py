from __future__ import annotations

from collections.abc import Callable
from typing import Dict

from langgraph.graph import Graph

from democracy_exe.ai.agents.router_agent import RouterAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph, conditional_edge


class RouterGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.router_agent = RouterAgent()
        self.specialized_agents: dict[str, Callable[[AgentState], AgentState]] = {}

    def add_specialized_agent(self, name: str, agent: Callable[[AgentState], AgentState]):
        self.specialized_agents[name] = agent

    def build(self) -> Graph:
        # Create a new graph
        self.graph = Graph()

        # Always add router node, even for empty graph
        self.graph.add_node("router", AgentNode(self.router_agent))
        self.graph.set_entry_point("router")

        # Add specialized agent nodes and edges
        for agent_name, agent_func in self.specialized_agents.items():
            self.graph.add_node(agent_name, agent_func)
            # Add bidirectional edges
            self.graph.add_edge(agent_name, "router")
            self.graph.add_edge("router", agent_name)

        return self.graph

    def process(self, state: dict) -> dict:
        """Process state through the graph.

        Args:
            state: Initial agent state

        Returns:
            Final state after processing
        """
        # Create base state
        base_state = {
            "query": "",
            "response": "",
            "current_agent": "",
            "context": {}
        }

        # Update with input state
        if isinstance(state, dict):
            base_state.update(state)

        # Build and compile graph
        graph = self.build()
        compiled = graph.compile()

        # Process state through graph
        return compiled.invoke(base_state)

router_graph = RouterGraph()

"""
This router_graph.py file defines the RouterGraph class, which is responsible for managing the overall flow of the AI system:
The RouterGraph class inherits from BaseGraph and implements the build method.
It has a RouterAgent instance and a dictionary of specialized agents.
The add_specialized_agent method allows adding new specialized agents to the graph.
The build method constructs the graph:
It adds the router node and all specialized agent nodes.
It creates conditional edges from the router to specialized agents.
It adds edges from specialized agents back to the router.
It sets the entry point of the graph to the router.
The process method compiles the graph and processes the given state.
An instance of RouterGraph is created at the module level for easy access.
This implementation allows for a flexible and extensible routing system. The router can dynamically decide which specialized agent to call based on the current state, and the flow can return to the router after each specialized agent processes the state.
"""
