from __future__ import annotations

from typing import Any, Dict

from democracy_exe.ai.base import AgentState, BaseAgent


class MemoryAgent(BaseAgent):
    """Agent for storing and retrieving key-value information in memory.

    This agent provides a simple memory storage system that can remember and recall
    information based on user queries. It supports commands like 'remember' and 'recall'
    to store and retrieve values respectively.
    """

    def __init__(self) -> None:
        """Initialize the memory agent with an empty storage dictionary."""
        self.memory_storage: dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Store a value in memory under the specified key.

        Args:
            key: The identifier to store the value under
            value: The value to store
        """
        self.memory_storage[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve a value from memory by its key.

        Args:
            key: The identifier of the value to retrieve

        Returns:
            The stored value if found, None otherwise
        """
        return self.memory_storage.get(key, None)

    def process(self, state: AgentState) -> AgentState:
        """Process user queries to store or retrieve information.

        Handles commands starting with 'remember'/'store' for storing information
        and 'recall'/'retrieve' for retrieving information.

        Args:
            state: Current agent state containing the user query

        Returns:
            Updated agent state with the response message
        """
        query = state["query"].lower()
        if query.startswith("remember") or query.startswith("store"):
            parts = query.split(maxsplit=2)
            if len(parts) == 3:
                _, key, value = parts
                self.store(key, value)
                state["response"] = f"I've remembered that {key} is {value}."
            else:
                state["response"] = "I couldn't understand what to remember. Please use the format: remember [key] [value]"
        elif query.startswith("recall") or query.startswith("retrieve"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                _, key = parts
                value = self.retrieve(key)
                state["response"] = f"{key} is {value}" if value is not None else f"I don't remember anything about {key}."
            else:
                state["response"] = "I couldn't understand what to recall. Please use the format: recall [key]"
        else:
            state["response"] = "I can remember things for you or recall them. Try saying 'remember [key] [value]' or 'recall [key]'."
        return state

memory_agent = MemoryAgent()


"""
This memory_agent.py file defines the MemoryAgent class, which is responsible for storing and retrieving information:
The MemoryAgent class inherits from BaseAgent.
It uses a simple dictionary to store key-value pairs.
The process method handles both storing new information and retrieving existing information based on the query.
It provides user-friendly responses for successful operations and error cases.
These agents work together with the graphs we defined earlier to create a flexible and extensible AI system. The RouterAgent directs queries to specialized agents like the MemoryAgent, which can then perform their specific tasks.
"""
