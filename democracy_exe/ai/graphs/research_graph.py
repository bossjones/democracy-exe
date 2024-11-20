from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from langgraph.graph import Graph

from democracy_exe.ai.agents.research_agent import ResearchAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class ResearchGraph(BaseGraph):
    """Graph for orchestrating comprehensive research operations.

    This graph manages the flow of research operations through multiple stages:
    planning, source gathering, analysis, synthesis, and report generation.
    It coordinates these operations using a directed graph structure where each
    node represents a specific research stage.

    Attributes:
        research_agent: Instance of ResearchAgent that performs the actual
            research operations
    """

    def __init__(self) -> None:
        """Initialize the research graph with its agent."""
        super().__init__()
        self.research_agent = ResearchAgent()

    def build(self) -> Graph:
        """Construct the research workflow graph.

        Creates a directed graph with nodes for each research stage and edges
        defining the flow between stages. Includes conditional edges for
        iterative research when more sources are needed.

        Returns:
            Configured LangGraph Graph instance ready for execution
        """
        # Add research agent node
        self.graph.add_node("research", AgentNode(self.research_agent))

        # Add nodes for different research stages
        self.graph.add_node("plan_research", self.plan_research)
        self.graph.add_node("gather_sources", self.gather_sources)
        self.graph.add_node("analyze_information", self.analyze_information)
        self.graph.add_node("synthesize_findings", self.synthesize_findings)
        self.graph.add_node("generate_report", self.generate_report)

        # Add edges to create the research flow
        self.graph.add_edge("research", "plan_research")
        self.graph.add_edge("plan_research", "gather_sources")
        self.graph.add_edge("gather_sources", "analyze_information")
        self.graph.add_edge("analyze_information", "synthesize_findings")
        self.graph.add_edge("synthesize_findings", "generate_report")
        self.graph.add_edge("generate_report", "research")

        # Add conditional edges for iterative research
        self.graph.add_conditional_edges(
            "analyze_information",
            self.need_more_sources,
            {True: "gather_sources", False: "synthesize_findings"}
        )

        # Set the entry point
        self.graph.set_entry_point("research")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        """Process a research request through the research workflow.

        Args:
            state: Current agent state containing the research query

        Returns:
            Updated agent state with research results
        """
        compiled_graph = self.compile()
        return compiled_graph(state)

    def plan_research(self, state: AgentState) -> AgentState:
        """Create a structured research plan from the initial query.

        Args:
            state: Agent state containing the research query

        Returns:
            Updated state with research plan
        """
        research_plan = self.research_agent.plan_research(state["query"])
        state["research_plan"] = research_plan
        return state

    def gather_sources(self, state: AgentState) -> AgentState:
        """Gather relevant sources based on the research plan.

        Args:
            state: Agent state containing the research plan

        Returns:
            Updated state with gathered sources
        """
        sources = self.research_agent.gather_sources(state["research_plan"])
        state["sources"] = sources
        return state

    def analyze_information(self, state: AgentState) -> AgentState:
        """Analyze gathered sources to extract key information.

        Args:
            state: Agent state containing the sources to analyze

        Returns:
            Updated state with analysis results
        """
        analysis = self.research_agent.analyze_information(state["sources"])
        state["analysis"] = analysis
        return state

    def need_more_sources(self, state: AgentState) -> bool:
        """Determine if additional sources are needed for the research.

        Args:
            state: Agent state containing the current analysis

        Returns:
            True if more sources are needed, False otherwise
        """
        return self.research_agent.need_more_sources(state["analysis"])

    def synthesize_findings(self, state: AgentState) -> AgentState:
        """Synthesize analyzed information into coherent findings.

        Args:
            state: Agent state containing the analysis to synthesize

        Returns:
            Updated state with synthesized findings
        """
        synthesis = self.research_agent.synthesize_findings(state["analysis"])
        state["synthesis"] = synthesis
        return state

    def generate_report(self, state: AgentState) -> AgentState:
        """Generate a comprehensive research report.

        Args:
            state: Agent state containing the synthesized findings

        Returns:
            Updated state with generated report in the response field
        """
        report = self.research_agent.generate_report(state["synthesis"])
        state["response"] = report
        return state


research_graph = ResearchGraph()


"""
This research_graph.py file defines the ResearchGraph class, which encapsulates the functionality for conducting in-depth research:
The ResearchGraph class inherits from BaseGraph and implements the build method.
It has a ResearchAgent instance to handle the actual research operations.
The build method constructs the graph:
It adds the main research agent node.
It adds nodes for different stages of the research process: planning, gathering sources, analyzing information, synthesizing findings, and generating a report.
It adds edges to create the research flow, allowing for a complete research cycle.
It includes a conditional edge that allows the graph to return to the "gather_sources" stage if more information is needed.
It sets the entry point of the graph to the main research node.
The process method compiles the graph and processes the given state.
Each method (plan_research, gather_sources, etc.) corresponds to a different stage of the research process. They update the state with the results of each stage.
The need_more_sources method acts as a decision point, determining whether to gather more sources or move on to synthesizing findings.
An instance of ResearchGraph is created at the module level for easy access.
This implementation provides a structured and iterative approach to conducting research within the AI system. The research process is broken down into distinct stages, each handled by a separate node in the graph. The conditional edge allows for dynamic decision-making during the research process, enabling the agent to gather more information if needed.
This design allows for complex research tasks that can adapt based on the information gathered and analyzed, potentially leading to more thorough and accurate research outcomes.
"""
