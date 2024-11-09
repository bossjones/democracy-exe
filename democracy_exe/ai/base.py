from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from langgraph.graph import Graph


class AgentState(TypedDict):
    query: str
    response: str
    current_agent: str
    context: dict[str, Any]

class BaseAgent(ABC):
    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        pass

class BaseGraph(ABC):
    def __init__(self):
        self.graph = Graph()

    @abstractmethod
    def build(self) -> Graph:
        pass

    def compile(self) -> Any:
        return self.build().compile()

class AgentNode:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    def __call__(self, state: AgentState) -> AgentState:
        return self.agent.process(state)

def conditional_edge(state: AgentState) -> str:
    return state["current_agent"]



"""
This base.py file defines the core structures and abstract classes that will be used throughout the AI module:
AgentState: A TypedDict that defines the structure of the state passed between agents.
BaseAgent: An abstract base class that all agents should inherit from, defining the process method.
BaseGraph: An abstract base class for graphs, providing a common interface for building and compiling graphs.
AgentNode: A wrapper class that allows agents to be used as nodes in the graph.
conditional_edge: A function used for conditional routing in the graph based on the current agent.
"""
