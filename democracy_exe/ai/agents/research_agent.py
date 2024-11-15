from __future__ import annotations

import os

from typing import Any, Dict, List

from langchain.chains.llm import LLMChain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from democracy_exe.ai.base import AgentState, BaseAgent
from democracy_exe.aio_settings import aiosettings


class ResearchAgent(BaseAgent):
    """Agent for conducting comprehensive research on given topics.

    This agent combines internet search capabilities with language model processing
    to perform structured research. It follows a multi-step process: planning research,
    gathering sources, analyzing information, and generating reports.

    Raises:
        ValueError: If TAVILY_API_KEY or OPENAI_API_KEY environment variables are not set.
    """

    def __init__(self) -> None:
        """Initialize the research agent with search tool and language model.

        Raises:
            ValueError: If required API keys are not set in environment variables.
        """
        self.search_tool = TavilySearchResults(
            api_key=str(aiosettings.tavily_api_key),
            max_results=5
        )
        self.llm = ChatOpenAI(
            openai_api_key=str(aiosettings.openai_api_key),
            temperature=0.7
        )

    def plan_research(self, query: str) -> list[str]:
        """Create a structured research plan from the initial query.

        Args:
            query: The main research topic or question

        Returns:
            List of specific questions or subtopics to investigate
        """
        plan_prompt = ChatPromptTemplate.from_template(
            "Given the research topic: {query}\n"
            "Create a list of 3-5 specific questions or subtopics to investigate."
        )
        plan_chain = LLMChain(llm=self.llm, prompt=plan_prompt)
        plan = plan_chain.run(query=query)
        return plan.split('\n')

    def gather_sources(self, research_plan: list[str]) -> list[dict[str, str]]:
        """Gather relevant sources for each question in the research plan.

        Args:
            research_plan: List of research questions to investigate

        Returns:
            List of source dictionaries containing search results
        """
        all_sources = []
        for question in research_plan:
            results = self.search_tool.invoke({"query": question})
            all_sources.extend(results.get("results", []))
        return all_sources

    def analyze_information(self, sources: list[dict[str, str]]) -> str:
        """Analyze gathered sources to extract key information.

        Args:
            sources: List of source dictionaries to analyze

        Returns:
            Analysis summary highlighting key points and potential conflicts
        """
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze the following information and provide a summary:\n"
            "{sources}\n"
            "Highlight key points, identify any conflicting information, "
            "and note areas where more research might be needed."
        )
        analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        sources_text = "\n".join([f"Title: {s['title']}\nContent: {s['snippet']}" for s in sources[:5]])
        return analysis_chain.run(sources=sources_text)

    def need_more_sources(self, analysis: str) -> bool:
        """Determine if additional research is needed.

        Args:
            analysis: Current analysis text

        Returns:
            True if more sources are needed, False otherwise
        """
        return len(analysis) < 1000

    def synthesize_findings(self, analysis: str) -> str:
        """Synthesize analyzed information into coherent findings.

        Args:
            analysis: Analyzed information text

        Returns:
            Synthesized findings with conclusions and suggestions
        """
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Based on the following analysis:\n"
            "{analysis}\n"
            "Synthesize the main findings, draw conclusions, and suggest any "
            "further areas of research or open questions."
        )
        synthesis_chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)
        return synthesis_chain.run(analysis=analysis)

    def generate_report(self, synthesis: str) -> str:
        """Generate a comprehensive research report.

        Args:
            synthesis: Synthesized findings text

        Returns:
            Formatted research report with introduction, findings, and conclusion
        """
        report_prompt = ChatPromptTemplate.from_template(
            "Create a comprehensive research report based on the following synthesis:\n"
            "{synthesis}\n"
            "Structure the report with an introduction, main findings, conclusion, "
            "and suggestions for further research."
        )
        report_chain = LLMChain(llm=self.llm, prompt=report_prompt)
        return report_chain.run(synthesis=synthesis)

    def process(self, state: AgentState) -> AgentState:
        """Process the research request and generate a comprehensive report.

        Args:
            state: Current agent state containing the research query

        Returns:
            Updated agent state with the research report or error message
        """
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
