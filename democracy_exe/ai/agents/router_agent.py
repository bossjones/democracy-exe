from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Optional

from democracy_exe.ai.base import AgentState, BaseAgent


class RouterAgent(BaseAgent):
    """Agent responsible for routing queries to specialized agents.

    This agent acts as a dispatcher that analyzes incoming queries and routes them
    to the appropriate specialized agent based on the query content. It maintains
    a registry of specialized agents and their processing functions.
    """

    def __init__(self) -> None:
        """Initialize the router agent with an empty specialized agents registry."""
        self.specialized_agents: dict[str, Callable[[AgentState], AgentState]] = {}

    def add_specialized_agent(self, name: str, process_func: Callable[[AgentState], AgentState]) -> None:
        """Register a new specialized agent.

        Args:
            name: Identifier for the specialized agent
            process_func: Function that processes agent state for the specialized agent
        """
        self.specialized_agents[name] = process_func

    def route(self, state: AgentState) -> str:
        """Determine which specialized agent should handle the query.

        Args:
            state: Current agent state containing the query

        Returns:
            Name of the specialized agent to handle the query, or empty string if no
            suitable agent is found
        """
        query = state["query"].lower()
        if "search" in query:
            return "internet_search"
        elif "image" in query or "analyze" in query:
            return "image_analysis"
        elif "research" in query:
            return "research"
        elif "tweet" in query or "social media" in query:
            return "social_media"
        elif "process" in query and ("image" in query or "video" in query):
            return "image_video_processing"
        elif "remember" in query or "recall" in query:
            return "memory"
        else:
            return ""

    def process(self, state: AgentState) -> AgentState:
        """Process the agent state by routing to appropriate specialized agent.

        Args:
            state: Current agent state containing the query

        Returns:
            Updated agent state after processing by specialized agent or with
            error message if no suitable agent is found
        """
        next_agent = self.route(state)
        if next_agent in self.specialized_agents:
            state["current_agent"] = next_agent
            return self.specialized_agents[next_agent](state)
        else:
            state["response"] = "I'm not sure how to handle this request."
            return state


router_agent = RouterAgent()

"""
This router_agent.py file defines the RouterAgent class, which is responsible for directing queries to the appropriate specialized agent:
The RouterAgent class inherits from BaseAgent.
It maintains a dictionary of specialized agents that can be added dynamically.
The route method implements the logic for determining which specialized agent should handle a given query. This can be expanded or refined as needed.
The process method uses the routing logic to delegate the query to the appropriate specialized agent, or returns a default response if no suitable agent is found.
Next, let's look at the memory_agent.py file:
"""