from __future__ import annotations

from langgraph.graph import Graph

from democracy_exe.ai.agents.internet_search_agent import InternetSearchAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class InternetSearchGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.search_agent = InternetSearchAgent()

    def build(self) -> Graph:
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
        compiled_graph = self.compile()
        return compiled_graph(state)

    def parse_query(self, state: AgentState) -> AgentState:
        # Implement query parsing logic
        parsed_query = self.search_agent.parse_query(state["query"])
        state["parsed_query"] = parsed_query
        return state

    def execute_search(self, state: AgentState) -> AgentState:
        # Implement search execution logic
        search_results = self.search_agent.execute_search(state["parsed_query"])
        state["search_results"] = search_results
        return state

    def process_results(self, state: AgentState) -> AgentState:
        # Implement results processing logic
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
