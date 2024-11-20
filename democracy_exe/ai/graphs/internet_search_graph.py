from __future__ import annotations

from typing import Any

from langgraph.graph import Graph

from democracy_exe.ai.agents.internet_search_agent import InternetSearchAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class InternetSearchGraph(BaseGraph):
    """Graph for orchestrating internet search operations.

    This graph manages the flow of internet search operations through multiple stages:
    query parsing, search execution, and results processing. It coordinates these
    operations using a directed graph structure where each node represents a specific
    search stage.

    Attributes:
        search_agent: Instance of InternetSearchAgent that performs the actual
            search operations
    """

    def __init__(self) -> None:
        """Initialize the internet search graph with its agent."""
        super().__init__()
        self.search_agent = InternetSearchAgent()

    def build(self) -> Graph:
        """Construct the internet search workflow graph.

        Creates a directed graph with nodes for each search stage and edges
        defining the flow between stages.

        Returns:
            Configured LangGraph Graph instance ready for execution
        """
        # Add search agent node
        self.graph.add_node("search", AgentNode(self.search_agent))

        # Add nodes for different search stages
        self.graph.add_node("parse_query", self.parse_query)
        self.graph.add_node("execute_search", self.execute_search)
        self.graph.add_node("process_results", self.process_results)

        # Add edges to create the search flow
        self.graph.add_edge("search", "parse_query")
        self.graph.add_edge("parse_query", "execute_search")
        self.graph.add_edge("execute_search", "process_results")
        self.graph.add_edge("process_results", "search")

        # Set the entry point
        self.graph.set_entry_point("search")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        """Process a search request through the search workflow.

        Args:
            state: Current agent state containing the search query

        Returns:
            Updated agent state with search results
        """
        compiled_graph = self.compile()
        return compiled_graph(state)

    def parse_query(self, state: AgentState) -> AgentState:
        """Parse and preprocess the search query.

        Processes the raw query string to optimize it for search execution.

        Args:
            state: Agent state containing the raw query

        Returns:
            Updated state with parsed query
        """
        parsed_query = self.search_agent.parse_query(state["query"])
        state["parsed_query"] = parsed_query
        return state

    def execute_search(self, state: AgentState) -> AgentState:
        """Execute the search using the parsed query.

        Performs the actual internet search operation using the search agent.

        Args:
            state: Agent state containing the parsed query

        Returns:
            Updated state with search results
        """
        search_results = self.search_agent.execute_search(state["parsed_query"])
        state["search_results"] = search_results
        return state

    def process_results(self, state: AgentState) -> AgentState:
        """Process and format the search results.

        Processes raw search results into a formatted response suitable for
        presentation to the user.

        Args:
            state: Agent state containing the search results

        Returns:
            Updated state with processed results in the response field
        """
        processed_results = self.search_agent.process_results(state["search_results"])
        state["response"] = processed_results
        return state


internet_search_graph = InternetSearchGraph()


"""
This internet_search_graph.py file defines the InternetSearchGraph class, which encapsulates the functionality for performing internet searches:
The InternetSearchGraph class inherits from BaseGraph and implements the build method.
It has an InternetSearchAgent instance to handle the actual search operations.
The build method constructs the graph:
It adds the main search agent node.
It adds nodes for different stages of the search process: parsing the query, executing the search, and processing the results.
It adds edges to create the search flow, allowing for a complete search cycle.
It sets the entry point of the graph to the main search node.
The process method compiles the graph and processes the given state.
The parse_query, execute_search, and process_results methods correspond to the different stages of the search process. They update the state with the results of each stage.
An instance of InternetSearchGraph is created at the module level for easy access.
This implementation allows for a structured and modular approach to internet searching within the AI system. The search process is broken down into distinct stages, each handled by a separate node in the graph. This makes it easy to modify or extend individual parts of the search process as needed.
The cyclic nature of the graph allows for potential refinement of search queries based on initial results, if such functionality is desired in the future.
"""
