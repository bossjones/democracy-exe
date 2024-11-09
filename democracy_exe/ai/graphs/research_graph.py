from __future__ import annotations

from langgraph.graph import Graph

from democracy_exe.ai.agents.research_agent import ResearchAgent
from democracy_exe.ai.base import AgentNode, AgentState, BaseGraph


class ResearchGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        self.research_agent = ResearchAgent()

    def build(self) -> Graph:
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
        self.graph.add_conditional_edges("analyze_information", self.need_more_sources,
                                       {True: "gather_sources", False: "synthesize_findings"})

        # Set the entry point
        self.graph.set_entry_point("research")

        return self.graph

    def process(self, state: AgentState) -> AgentState:
        compiled_graph = self.compile()
        return compiled_graph(state)

    def plan_research(self, state: AgentState) -> AgentState:
        research_plan = self.research_agent.plan_research(state["query"])
        state["research_plan"] = research_plan
        return state

    def gather_sources(self, state: AgentState) -> AgentState:
        sources = self.research_agent.gather_sources(state["research_plan"])
        state["sources"] = sources
        return state

    def analyze_information(self, state: AgentState) -> AgentState:
        analysis = self.research_agent.analyze_information(state["sources"])
        state["analysis"] = analysis
        return state

    def need_more_sources(self, state: AgentState) -> bool:
        return self.research_agent.need_more_sources(state["analysis"])

    def synthesize_findings(self, state: AgentState) -> AgentState:
        synthesis = self.research_agent.synthesize_findings(state["analysis"])
        state["synthesis"] = synthesis
        return state

    def generate_report(self, state: AgentState) -> AgentState:
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
