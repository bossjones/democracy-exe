"""Lang-MemGPT: A Long-Term Memory Agent.

This module implements an agent with long-term memory capabilities using LangGraph.
The agent can store, retrieve, and use memories to enhance its interactions with users.

Key Components:
1. Memory Types: Core (always available) and Recall (contextual/semantic)
2. Tools: For saving and retrieving memories + performing other tasks.
3. Vector Database: for recall memory. Uses Pinecone by default.

Configuration: Requires Pinecone and Fireworks API keys (see README for setup)
"""
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false

from __future__ import annotations

import json
import logging
import uuid

from datetime import UTC, datetime, timezone
from typing import Literal, Optional, Tuple, Union

import langsmith
import rich
import structlog
import tiktoken

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig, ensure_config, get_executor_for_config
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph  # type: ignore
from langgraph.prebuilt import ToolNode


logger = structlog.get_logger(__name__)

from democracy_exe.agentic import _constants as constants
from democracy_exe.agentic import _schemas as schemas
from democracy_exe.agentic import _utils as agentic_utils
from democracy_exe.aio_settings import aiosettings


# Type alias for chat models
ChatModelLike = Union[ChatOpenAI, ChatAnthropic]

_EMPTY_VEC = [0.0] * 768

# Initialize the search tool
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]


@tool
async def save_recall_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.

    Returns:
        str: The saved memory.
    """
    config: RunnableConfig = ensure_config()
    configurable: schemas.GraphConfig = agentic_utils.ensure_configurable(config)
    embeddings = agentic_utils.get_embeddings(model_name=aiosettings.openai_embeddings_model)
    vector = await embeddings.aembed_query(memory)
    current_time = datetime.now(tz=UTC)
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],  # pyright: ignore[reportUndefinedVariable]
        event_id=str(uuid.uuid4()),
    )
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: memory,
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: current_time,
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"], # pyright: ignore[reportUndefinedVariable]
            },
        }
    ]
    logger.error(f"configurable: {configurable}")
    logger.error(f"documents: {documents}")
    logger.error(f"path: {path}")
    logger.error(f"current_time: {current_time}")
    logger.error(f"vector: {vector}")
    logger.error(f"embeddings: {embeddings}")

    # TODO: fix this to work with chroma/scikitlearn
    # agentic_utils.get_index().upsert(
    #     vectors=documents,
    #     namespace=aiosettings.pinecone_namespace,
    # )
    # await logger.complete()
    return memory


@tool
def search_memory(query: str, top_k: int = 5) -> list[str]:
    """Search for memories in the database based on semantic similarity.

    Args:
        query (str): The search query.
        top_k (int): The number of results to return.

    Returns:
        list[str]: A list of relevant memories.
    """
    config: RunnableConfig = ensure_config()
    configurable: schemas.GraphConfig = agentic_utils.ensure_configurable(config)
    embeddings = agentic_utils.get_embeddings(model_name=aiosettings.openai_embeddings_model)
    vector = embeddings.embed_query(query)
    with langsmith.trace("query", inputs={"query": query, "top_k": top_k}) as rt:
        # TODO: fix this to work with chroma/scikitlearn
        response = agentic_utils.get_index().query(
            vector=vector,
            filter={
                "user_id": {"$eq": configurable["user_id"]}, # pyright: ignore[reportUndefinedVariable]
                constants.TYPE_KEY: {"$eq": "recall"},
            },
            namespace=aiosettings.pinecone_namespace,
            include_metadata=True,
            top_k=top_k,
        )
        rt.end(outputs={"response": response})
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches] # pyright: ignore[reportUndefinedVariable]
    return memories


# @langsmith.traceable
def fetch_core_memories(user_id: str) -> tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path: str = constants.PATCH_PATH.format(user_id=user_id)
    logger.error(f"path: {path}")
    # TODO: fix this to work with chroma/scikitlearn
    response = agentic_utils.get_index().fetch(
        ids=[path], namespace=aiosettings.pinecone_namespace
    )
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories


@tool
def store_core_memory(memory: str, index: int | None = None) -> str:
    """Store a core memory in the database.

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    config: RunnableConfig = ensure_config()
    configurable: schemas.GraphConfig = agentic_utils.ensure_configurable(config)
    path, memories = fetch_core_memories(configurable["user_id"]) # pyright: ignore[reportUndefinedVariable]
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": _EMPTY_VEC,
            "metadata": {
                constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: datetime.now(tz=UTC),
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"], # pyright: ignore[reportUndefinedVariable]
            },
        }
    ]
    # TODO: fix this to work with chroma/scikitlearn
    # agentic_utils.get_index().upsert(
    #     vectors=documents,
    #     namespace=aiosettings.pinecone_namespace,
    # )
    return "Memory stored."


# Combine all tools
all_tools: list[BaseTool | TavilySearchResults] = tools + [save_recall_memory, search_memory, store_core_memory]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Core Memories\n"
            "Core memories are fundamental to understanding the user and are"
            " always available:\n{core_memories}\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
            "Current system time: {current_time}\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)


# @langsmith.traceable
async def agent(
    state: schemas.State,
    config: RunnableConfig,
) -> schemas.State:
    """Process the current state and generate a response using the LLM.

    This function is a core component of the memory-enabled agent that:
    1. Retrieves the appropriate LLM model based on configuration
    2. Binds available tools to the model
    3. Formats core and recall memories for context
    4. Generates a response using the LLM with full context
    5. Logs key information for debugging

    Args:
        state: The current conversation state containing messages and memories.
               Expected keys: "messages", "core_memories", "recall_memories"
        config: Runtime configuration for the agent execution.
                Must contain "model" and "user_id" in its configurable dict.

    Returns:
        Updated state dictionary containing the LLM's response in the "messages" key.

    Raises:
        KeyError: If required state keys are missing
        ValueError: If model configuration is invalid
        RuntimeError: If LLM invocation fails
    """
    configurable: schemas.GraphConfig = agentic_utils.ensure_configurable(config)
    llm: ChatModelLike = agentic_utils.get_chat_model(
        model_name=configurable["model"],  # type: ignore
        model_provider=aiosettings.llm_provider,
        temperature=0.0
    )
    bound = prompt | llm.bind_tools(all_tools)
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    logger.error(f"core_str: {core_str}")
    logger.error(f"recall_str: {recall_str}")
    logger.error(f"agent state: {state}")
    logger.error(f"agent config: {config}")
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "recall_memories": recall_str,
            "current_time": datetime.now(tz=UTC).isoformat(),
        }
    )
    # await logger.complete()
    return {
        "messages": prediction,
    }


def load_memories(state: schemas.State, config: RunnableConfig) -> schemas.State:
    """Load core and recall memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """
    configurable: schemas.GraphConfig = agentic_utils.ensure_configurable(config)
    user_id: str | int = configurable["user_id"]
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])

    with get_executor_for_config(config) as executor:
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_memory.invoke, convo_str),
        ]
        _, core_memories = futures[0].result()
        recall_memories = futures[1].result()
    return {
        "core_memories": core_memories,
        "recall_memories": recall_memories,
    }


def route_tools(state: schemas.State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    logger.error(f"state: {state}")
    msg = state["messages"][-1]
    rich.inspect(msg, all=True)
    if msg.tool_calls:
        return "tools"
    return END


# Create the graph and add nodes
builder = StateGraph(schemas.State, schemas.GraphConfig)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(all_tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")

# Compile the graph
memgraph: CompiledStateGraph = builder.compile(interrupt_before=["agent"])

__all__ = ["memgraph"]
