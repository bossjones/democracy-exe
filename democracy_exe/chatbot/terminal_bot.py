from __future__ import annotations

import asyncio
import signal
import sys

from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from rich import print as rprint

from democracy_exe.ai.graphs import AgentState, router_graph
from democracy_exe.aio_settings import aiosettings


async def main() -> None:
    """Main function to run the LangGraph Chatbot in the terminal.

    This function handles user input and processes it through the AI pipeline.
    """
    logger.info("Starting the LangGraph Chatbot")
    rprint("[bold green]Welcome to the LangGraph Chatbot! Type 'quit' to exit.[/bold green]")
    logger.info("Welcome to the LangGraph Chatbot! Type 'quit' to exit.")

    while True:
        user_input = await asyncio.to_thread(input, "You: ")

        if user_input.lower() == 'quit':
            rprint("[bold red]Goodbye![/bold red]")
            logger.info("Goodbye!")
            break

        # Create a HumanMessage from the user input
        message = HumanMessage(content=user_input)

        try:

            state = AgentState(
                query=message.content,
                response="",
                current_agent="",
                context={"message": message}
            )
            result: AgentState = router_graph.process(state)

            # Print the AI's response
            rprint(f"[bold blue]AI:[/bold blue] {result['response']}")
            logger.info(f"AI: {result['response']}")
        except Exception as e:
            logger.exception("Error processing message")
            rprint("[bold red]An error occurred while processing your message.[/bold red]")


def handle_sigterm(signo, frame):
    sys.exit(128 + signo)  # this will raise SystemExit and cause atexit to be called


signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    asyncio.run(main())
