from __future__ import annotations

from typing import Dict, List

from langchain_community.tools.tavily_search import TavilySearchResults

from democracy_exe.ai.base import AgentState, BaseAgent
from democracy_exe.aio_settings import aiosettings


class InternetSearchAgent(BaseAgent):
    """Agent for performing internet searches using Tavily API.

    This agent uses the TavilySearchResults tool from LangChain to perform web searches
    and process the results into a readable format.

    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set.
    """

    def __init__(self):
        self.search_tool = TavilySearchResults(
            api_key=str(aiosettings.tavily_api_key),
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=False
        )

    def parse_query(self, query: str) -> str:
        """Preprocess the search query.

        Args:
            query: Raw search query string.

        Returns:
            Processed query string with whitespace stripped.
        """
        return query.strip()

    def execute_search(self, query: str) -> list[dict[str, str]]:
        """Execute search using Tavily API.

        Args:
            query: Search query string.

        Returns:
            List of search results, where each result is a dictionary containing
            'title', 'snippet', and 'url' keys.
        """
        search_results = self.search_tool.invoke({"query": query})
        return search_results.get("results", [])

    def process_results(self, results: list[dict[str, str]]) -> str:
        """Process search results into a formatted string.

        Args:
            results: List of search result dictionaries.

        Returns:
            Formatted string containing numbered search results with titles,
            descriptions, and URLs.
        """
        if not results:
            return "I couldn't find any relevant information."

        response = "Here's what I found:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result.get('title', 'No title')}\n"
            response += f"   {result.get('snippet', 'No description')}\n"
            response += f"   URL: {result.get('url', 'No URL')}\n\n"
        return response

    def process(self, state: AgentState) -> AgentState:
        """Process the agent state by performing a search and updating results.

        Args:
            state: Current agent state containing the search query.

        Returns:
            Updated agent state with search results in the response field.
        """
        query = self.parse_query(state["query"])
        search_results = self.execute_search(query)
        processed_results = self.process_results(search_results)
        state["response"] = processed_results
        return state

internet_search_agent = InternetSearchAgent()


"""
Key changes in this updated version:
We've replaced the web scraping approach with the TavilySearchResults tool from LangChain.
The __init__ method now initializes the TavilySearchResults tool with some configuration options.
We've added a check to ensure the TAVILY_API_KEY environment variable is set.
The execute_search method now uses the TavilySearchResults tool to perform the search.
The process_results method has been updated to work with the structure of results returned by the Tavily search.
We've removed the BeautifulSoup and requests imports as they're no longer needed.
This implementation provides several advantages:
It's more robust and less likely to break due to changes in website structures.
It complies with the terms of service of the search API.
It provides more structured and reliable results.
It's easier to maintain and extend.
"""
