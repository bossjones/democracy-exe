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
        # Add router node
        self.graph.add_node("router", AgentNode(self.router_agent))

        # Add specialized agent nodes
        for agent_name, agent_func in self.specialized_agents.items():
            self.graph.add_node(agent_name, agent_func)

        # Add conditional edge from router to specialized agents
        self.graph.add_conditional_edges(
            "router",
            conditional_edge,
            {agent_name: agent_name for agent_name in self.specialized_agents}
        )

        # Add edges from specialized agents back to router
        for agent_name in self.specialized_agents:
            self.graph.add_edge(agent_name, "router")

        # Set the entry point
        self.graph.set_entry_point("router")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        compiled_graph = self.compile()
        return compiled_graph(state)

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
