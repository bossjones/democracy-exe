from __future__ import annotations

import asyncio
import signal
import sys

from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from rich import print as rprint

from democracy_exe.agentic import _utils as agentic_utils
from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.aio_settings import aiosettings


class FlushingStderr:
    def write(self, message):
        sys.stderr.write(message)
        sys.stderr.flush()

logger.remove()
logger.add(FlushingStderr(), enqueue=True)

async def go_terminal_bot(graph: CompiledStateGraph = memgraph) -> None:
    """Main function to run the LangGraph Chatbot in the terminal.

    This function handles user input and processes it through the AI pipeline.
    """
    logger.info("Starting the DemocracyExeAI Chatbot")
    rprint("[bold green]Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.[/bold green]")
    logger.info("Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.")

    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    await logger.complete()

    # Flush stderr before input prompt
    sys.stderr.flush()

    while True:
        user_input = await asyncio.to_thread(input, "You: ")

        if user_input.lower() == 'quit':
            rprint("[bold red]Goodbye![/bold red]")
            logger.info("Goodbye!")
            break

        # Create a HumanMessage from the user input
        message = HumanMessage(content=user_input)

        try:
            user_input = {"messages": [message]}
            stream_terminal_bot(graph, user_input, config)
        except Exception as e:
            logger.exception("Error processing message")
            rprint("[bold red]An error occurred while processing your message.[/bold red]")


def stream_terminal_bot(graph: CompiledStateGraph = memgraph, user_input: dict = None, thread: dict = None, interruptable: bool = False) -> None:
    """Stream the LangGraph Chatbot in the terminal."""
    # Run the graph until the first interruption
    for event in graph.stream(user_input, thread, stream_mode="values"):
        logger.debug(event)
        event['messages'][-1].pretty_print()

    if interruptable:
        # Get user feedback
        user_approval = input("Do you want to call the tool? (yes[y]/no[n]): ")

        # Check approval
        if user_approval.lower() == "yes" or user_approval.lower() == "y":

            # If approved, continue the graph execution
            for event in graph.stream(None, thread, stream_mode="values"):
                event['messages'][-1].pretty_print()

        else:
            print("Operation cancelled by user.")



def invoke_terminal_bot(graph: CompiledStateGraph = memgraph, user_input: dict = None, thread: dict = None) -> None:
    """Invoke the LangGraph Chatbot in the terminal."""
    messages = graph.invoke(user_input, thread)
    for m in messages['messages']:
        m.pretty_print()
    rprint(f"[bold blue]AI:[/bold blue] {messages['response']}")
    logger.info(f"AI: {messages['response']}")
    return messages['response']

# def handle_sigterm(signo, frame):
#     sys.exit(128 + signo)  # this will raise SystemExit and cause atexit to be called


# signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    asyncio.run(go_terminal_bot())
