from __future__ import annotations

import os

from typing import Dict, List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from democracy_exe.ai.base import AgentState, BaseAgent


class ResearchAgent(BaseAgent):
    def __init__(self):
        # Ensure you have set the necessary API keys
        if "TAVILY_API_KEY" not in os.environ:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.search_tool = TavilySearchResults(max_results=5)
        self.llm = ChatOpenAI(temperature=0.7)

    def plan_research(self, query: str) -> list[str]:
        plan_prompt = ChatPromptTemplate.from_template(
            "Given the research topic: {query}\n"
            "Create a list of 3-5 specific questions or subtopics to investigate."
        )
        plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)
        plan = plan_chain.run(query=query)
        return plan.split('\n')

    def gather_sources(self, research_plan: list[str]) -> list[dict[str, str]]:
        all_sources = []
        for question in research_plan:
            results = self.search_tool.invoke({"query": question})
            all_sources.extend(results.get("results", []))
        return all_sources

    def analyze_information(self, sources: list[dict[str, str]]) -> str:
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze the following information and provide a summary:\n"
            "{sources}\n"
            "Highlight key points, identify any conflicting information, "
            "and note areas where more research might be needed."
        )
        analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        sources_text = "\n".join([f"Title: {s['title']}\nContent: {s['snippet']}" for s in sources[:5]])
        return analysis_chain.run(sources=sources_text)

    def synthesize_findings(self, analysis: str) -> str:
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Based on the following analysis:\n"
            "{analysis}\n"
            "Synthesize the main findings, draw conclusions, and suggest any "
            "further areas of research or open questions."
        )
        synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)
        return synthesis_chain.run(analysis=analysis)

    def generate_report(self, synthesis: str) -> str:
        report_prompt = ChatPromptTemplate.from_template(
            "Create a comprehensive research report based on the following synthesis:\n"
            "{synthesis}\n"
            "Structure the report with an introduction, main findings, conclusion, "
            "and suggestions for further research."
        )
        report_chain = LLMChain(llm=self.llm, prompt=report_prompt)
        return report_chain.run(synthesis=synthesis)

    def process(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            research_plan = self.plan_research(query)
            sources = self.gather_sources(research_plan)
            analysis = self.analyze_information(sources)
            synthesis = self.synthesize_findings(analysis)
            report = self.generate_report(synthesis)
            state["response"] = report
        except Exception as e:
            state["response"] = f"An error occurred during the research process: {e!s}"
        return state

research_agent = ResearchAgent()


"""
This research_agent.py file defines the ResearchAgent class, which is responsible for conducting in-depth research on a given topic. Here's a breakdown of its components:
The agent uses both the TavilySearchResults tool for internet searches and the ChatOpenAI model for language processing tasks.
The plan_research method creates a research plan by generating specific questions or subtopics to investigate.
The gather_sources method uses the search tool to find relevant information for each question in the research plan.
The analyze_information method processes the gathered sources and provides a summary and analysis.
The synthesize_findings method takes the analysis and synthesizes the main findings and conclusions.
The generate_report method creates a comprehensive research report based on the synthesis.
The process method orchestrates the entire research process, handling potential errors and updating the state with the final research report.
This implementation provides a structured approach to conducting research, combining internet search capabilities with language model processing to create more insightful and comprehensive results.
"""
