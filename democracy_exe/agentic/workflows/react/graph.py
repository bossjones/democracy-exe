"""React Agent with Long-Term Memory.

This module implements a React agent with long-term memory capabilities using LangChain and LangGraph.
It manages user profiles, todo lists, and custom instructions through a state graph architecture.
"""
# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import langsmith
import rich

# from democracy_exe.aio_settings import aiosettings
import structlog
import tiktoken

from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, merge_message_runs
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.config import RunnableConfig, ensure_config, get_executor_for_config
from langchain_core.tools import tool
from langchain_core.tracers.schemas import Run
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


logger = structlog.get_logger(__name__)


import structlog


logger = structlog.get_logger(__name__)
from pydantic import BaseModel, Field
from trustcall import create_extractor
from trustcall._base import ExtractionOutputs, InputsLike

from democracy_exe.agentic.workflows.react import _utils, configuration
from democracy_exe.agentic.workflows.react.prompts import (
    CREATE_INSTRUCTIONS,
    MODEL_SYSTEM_MESSAGE,
    TRUSTCALL_INSTRUCTION,
)


# if aiosettings.debug_langgraph_studio:
#     print("DEBUG_LANGGRAPH_STUDIO is True")
#     logger.remove()
#     from democracy_exe.bot_logger import get_logger, global_log_config
#     from langchain.globals import set_debug, set_verbose
#     # Setting the global debug flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.
#     set_debug(True)
#     # Setting the verbose flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.
#     set_verbose(True)

#     # SOURCE: https://github.com/Delgan/loguru/blob/420704041797daf804b505e5220805528fe26408/docs/resources/recipes.rst#L1083
#     global_log_config(
#         log_level=logging.getLevelName("DEBUG"),
#         json=False,
#     )

# logger.remove()


# import nest_asyncio

# nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions


class Memory(BaseModel):
    """A single memory entry containing user-related information.

    Attributes:
        content (str): The main content of the memory (e.g., "User expressed interest in learning about French")
    """
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

class MemoryCollection(BaseModel):
    """A collection of memories about the user.

    Attributes:
        memories (list[Memory]): A list of Memory objects containing user-related information
    """
    memories: list[Memory] = Field(description="A list of memories about the user.")


## Utilities

# SOURCE: https://github.com/langchain-ai/langchain-academy/blob/main/module-5/memory_agent.ipynb
# Visibility into Trustcall updates
# Trustcall creates and updates JSON schemas.
# What if we want visibility into the specific changes made by Trustcall?
# For example, we saw before that Trustcall has some of its own tools to:
# Self-correct from validation failures -- see trace example here
# Update existing documents -- see trace example here
# Visibility into these tools can be useful for the agent we're going to build.
# Below, we'll show how to do this!


# ---------------------------------------------------------------------------------------
# We can add a listener to the Trustcall extractor.
# This will pass runs from the extractor's execution to a class, Spy, that we will define.
# Our Spy class will extract information about what tool calls were made by Trustcall.
# ---------------------------------------------------------------------------------------
# Inspect the tool calls for Trustcall
class Spy:
    """A class to monitor and collect tool calls made by the Trustcall extractor.

    This class acts as a listener for the Trustcall extractor's execution runs,
    collecting information about what tool calls were made during execution.

    Attributes:
        called_tools (list): A list to store tool calls made during execution.
    """

    def __init__(self) -> None:
        """Initialize the Spy with an empty list of called tools."""
        self.called_tools: list = []

    def __call__(self, run: Any) -> None:
        """Process a run and extract tool calls from chat model outputs.

        Traverses the run tree and collects tool calls from chat model outputs.

        Args:
            run: The run object containing execution information.
        """
        rich.print(f"Spy: {run}")
        rich.print(f"Spy type: {type(run)}")
        q: list = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generation"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# DISABLED: from module-5
# # Extract information from tool calls for both patches and new memories in Trustcall
# def extract_tool_info(tool_calls: list[list[dict[str, Any]]], schema_name: str = "Memory") -> str:
#     """Extract information from tool calls for both patches and new memories.

#     This function processes tool calls to extract information about document updates
#     and new memory creation. It formats the extracted information into a human-readable
#     string.

#     Args:
#         tool_calls: List of tool call groups, where each group contains tool call
#             dictionaries with information about patches or new memory creation
#         schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")

#     Returns:
#         A formatted string containing information about all document updates and
#         new memory creations
#     """
#     # Initialize list of changes
#     changes: list[dict[str, Any]] = []

#     for call_group in tool_calls:
#         for call in call_group:
#             if call['name'] == 'PatchDoc':
#                 changes.append({
#                     'type': 'update',
#                     'doc_id': call['args']['json_doc_id'],
#                     'planned_edits': call['args']['planned_edits'],
#                     'value': call['args']['patches'][0]['value']
#                 })
#             elif call['name'] == schema_name:
#                 changes.append({
#                     'type': 'new',
#                     'value': call['args']
#                 })

#     # Format results as a single string
#     result_parts: list[str] = []
#     for change in changes:
#         if change['type'] == 'update':
#             result_parts.append(
#                 f"Document {change['doc_id']} updated:\n"
#                 f"Plan: {change['planned_edits']}\n"
#                 f"Added content: {change['value']}"
#             )
#         else:
#             result_parts.append(
#                 f"New {schema_name} created:\n"
#                 f"Content: {change['value']}"
#             )

#     return "\n\n".join(result_parts)


# module-6
# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls: list[list[dict[str, Any]]], schema_name: str = "Memory") -> str:
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """
    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Check if there are any patches
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    # Handle case where no changes were needed
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)

## Schema definitions

# Creating an agent
# There are many different agent architectures to choose from.

# Here, we'll implement something simple, a ReAct agent.

# This agent will be a helpful companion for creating and managing a ToDo list.

# This agent can make a decision to update three types of long-term memory:

# (a) Create or update a user profile with general user information

# (b) Add or update items in a ToDo list collection

# (c) Update its own instructions on how to update items to the ToDo list

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: str | None = Field(description="The user's name", default=None)
    location: str | None = Field(description="The user's location", default=None)
    job: str | None = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has",
        default_factory=list
    )

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: int | None = Field(description="Estimated time to complete the task (minutes).")
    deadline: datetime | None = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """Decision on what memory type to update.

    Attributes:
        update_type (Literal['user', 'todo', 'instructions']): The type of memory to update
    """
    update_type: Literal['user', 'todo', 'instructions']

# Initialize the model

# model = ChatOpenAI(model="gpt-4o", temperature=0)
# model: BaseChatModel = init_chat_model("gpt-4o", model_provider=aiosettings.llm_provider, temperature=0.0) # pyright: ignore[reportUndefinedVariable]
# TODO: Use this to get embeddings
# tokenizer = tiktoken.encoding_for_model("gpt-4o")

# -----------------------------------------------------------------------------------
# async nodes
# -----------------------------------------------------------------------------------




## Node definitions

async def aio_tasks_democracy_ai(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[BaseMessage]]:
    """Load memories from the store and use them to personalize the chatbot's response.

    This function retrieves user profile, todo list, and custom instructions from the store
    and uses them to generate a personalized chatbot response.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing the list of messages with the chatbot's response
        Format: {"messages": [response]}
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    # DISABLED: # memories = store.search(namespace)

    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve people memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    # system_msg = configurable.system_prompt.format(user_profile=user_profile, todo=todo, instructions=instructions)
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)
    rich.print(f"system_msg: {system_msg}")
    # ---------------------------------------------------------------------------------------
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # ---------------------------------------------------------------------------------------

    # Respond using memory as well as the chat history
    # response = await model.bind_tools([UpdateMemory], parallel_tool_calls=False).ainvoke([SystemMessage(content=system_msg)]+state["messages"])

    # # Respond using memory as well as the chat history
    # response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    # Respond using memory as well as the chat history
    response = await model.bind_tools([UpdateMemory], parallel_tool_calls=False).ainvoke([SystemMessage(content=system_msg)]+state["messages"], config=config)


    return {"messages": [response]}

async def aio_update_profile(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the user profile in memory.

    This function processes the chat history to extract and update user profile information
    in the store using the Trustcall extractor.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message confirming the profile update
        Format: {
            "messages": [{
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )


    # Define the namespace for the memories
    namespace: tuple[str, str] = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name: str = "Profile"
    existing_memories: list[tuple[str, str, Any]] | None = (
        [(existing_item.key, tool_name, existing_item.value)
         for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    trustcall_instruction_formatted: str = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages: list[BaseMessage] = list(
        merge_message_runs(
            messages=[SystemMessage(content=trustcall_instruction_formatted)] + state["messages"][:-1]
        )
    )

    # ---------------------------------------------------------------------------------------
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # response = await model.ainvoke(messages, config)
    # Invoke the extractor
    # ---------------------------------------------------------------------------------------
    # # Invoke the extractor
    # result = profile_extractor.invoke({
    #     "messages": updated_messages,
    #     "existing": existing_memories
    # })

    # Invoke the extractor
    result = await profile_extractor.ainvoke({
        "messages": updated_messages,
        "existing": existing_memories
    }, config=config)

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"], strict=False):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    tool_calls = state['messages'][-1].tool_calls


    # Return tool message with update verification
    return {
        "messages": [{
            "role": "tool",
            "content": "updated profile",
            "tool_call_id": tool_calls[0]['id']
        }]
    }

async def aio_update_todos(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the todo list in memory.

    This function processes the chat history to extract and update todo items
    in the store using the Trustcall extractor. It also tracks changes made
    using a Spy instance.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message with update details
        Format: {
            "messages": [{
                "role": "tool",
                "content": str,  # Contains details of updates made
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )


    # Define the namespace for the memories
    namespace: tuple[str, str] = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name: str = "ToDo"
    existing_memories: list[tuple[str, str, Any]] | None = (
        [(existing_item.key, tool_name, existing_item.value)
         for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    trustcall_instruction_formatted: str = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages: list[BaseMessage] = list(
        merge_message_runs(
            messages=[SystemMessage(content=trustcall_instruction_formatted)] + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy: Spy = Spy()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor: Runnable[InputsLike, ExtractionOutputs] = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # # Invoke the extractor
    # result = todo_extractor.invoke({
    #     "messages": updated_messages,
    #     "existing": existing_memories
    # })

    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # response = await model.ainvoke(messages, config)
    # Invoke the extractor
    result = await todo_extractor.ainvoke({
        "messages": updated_messages,
        "existing": existing_memories
    }, config=config)

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"], strict=False):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Respond to the tool call made in tasks_democracy_ai, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add to the ToolMessage returned to tasks_democracy_ai
    todo_update_msg: str = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [{
            "role": "tool",
            "content": todo_update_msg,
            "tool_call_id": tool_calls[0]['id']
        }]
    }

async def aio_update_instructions(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the instructions in memory.

    This function processes the chat history to extract and update user-specified
    preferences for managing the todo list. It stores these instructions for future
    reference.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message confirming the instructions update
        Format: {
            "messages": [{
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

    namespace: tuple[str, str] = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg: str = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )

    # # # Respond using memory as well as the chat history
    # new_memory: BaseMessage = model.invoke(
    #     [SystemMessage(content=system_msg)] +
    #     state['messages'][:-1] +
    #     [HumanMessage(content="Please update the instructions based on the conversation")]
    # )
    # # Respond using memory as well as the chat history
    new_memory: BaseMessage = await model.ainvoke(
        [SystemMessage(content=system_msg)] +
        state['messages'][:-1] +
        [HumanMessage(content="Please update the instructions based on the conversation")],
        config=config
    )

    # Overwrite the existing memory in the store
    key: str = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})

    tool_calls = state['messages'][-1].tool_calls

    # Return tool message with update verification
    return {
        "messages": [{
            "role": "tool",
            "content": "updated instructions",
            "tool_call_id": tool_calls[0]['id']
        }]
    }


# -----------------------------------------------------------------------------------
# sync nodes
# -----------------------------------------------------------------------------------




## Node definitions

def tasks_democracy_ai(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[BaseMessage]]:
    """Load memories from the store and use them to personalize the chatbot's response.

    This function retrieves user profile, todo list, and custom instructions from the store
    and uses them to generate a personalized chatbot response.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing the list of messages with the chatbot's response
        Format: {"messages": [response]}
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    # DISABLED: # memories = store.search(namespace)

    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve people memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    # system_msg = configurable.system_prompt.format(user_profile=user_profile, todo=todo, instructions=instructions)
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)

    # ---------------------------------------------------------------------------------------
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # ---------------------------------------------------------------------------------------

    # Respond using memory as well as the chat history
    # response = await model.bind_tools([UpdateMemory], parallel_tool_calls=False).ainvoke([SystemMessage(content=system_msg)]+state["messages"])

    # # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"], config=config)

    # Respond using memory as well as the chat history
    # response = await model.bind_tools([UpdateMemory], parallel_tool_calls=False).ainvoke([SystemMessage(content=system_msg)]+state["messages"], config=config)

    return {"messages": [response]}

def update_profile(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the user profile in memory.

    This function processes the chat history to extract and update user profile information
    in the store using the Trustcall extractor.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message confirming the profile update
        Format: {
            "messages": [{
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )


    # Define the namespace for the memories
    namespace: tuple[str, str] = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name: str = "Profile"
    existing_memories: list[tuple[str, str, Any]] | None = (
        [(existing_item.key, tool_name, existing_item.value)
         for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    trustcall_instruction_formatted: str = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages: list[BaseMessage] = list(
        merge_message_runs(
            messages=[SystemMessage(content=trustcall_instruction_formatted)] + state["messages"][:-1]
        )
    )

    # ---------------------------------------------------------------------------------------
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # response = await model.ainvoke(messages, config)
    # Invoke the extractor
    # ---------------------------------------------------------------------------------------
    # # Invoke the extractor
    result = profile_extractor.invoke({
        "messages": updated_messages,
        "existing": existing_memories
    }, config=config)

    # # Invoke the extractor
    # result = await profile_extractor.ainvoke({
    #     "messages": updated_messages,
    #     "existing": existing_memories
    # }, config=config)

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"], strict=False):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {
        "messages": [{
            "role": "tool",
            "content": "updated profile",
            "tool_call_id": tool_calls[0]['id']
        }]
    }

def update_todos(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the todo list in memory.

    This function processes the chat history to extract and update todo items
    in the store using the Trustcall extractor. It also tracks changes made
    using a Spy instance.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message with update details
        Format: {
            "messages": [{
                "role": "tool",
                "content": str,  # Contains details of updates made
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )


    # Define the namespace for the memories
    namespace: tuple[str, str] = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name: str = "ToDo"
    existing_memories: list[tuple[str, str, Any]] | None = (
        [(existing_item.key, tool_name, existing_item.value)
         for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    trustcall_instruction_formatted: str = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages: list[BaseMessage] = list(
        merge_message_runs(
            messages=[SystemMessage(content=trustcall_instruction_formatted)] + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy: Spy = Spy()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor: Runnable[InputsLike, ExtractionOutputs] = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({
        "messages": updated_messages,
        "existing": existing_memories
    }, config=config)

    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    # response = await model.ainvoke(messages, config)
    # Invoke the extractor
    # result = await todo_extractor.ainvoke({
    #     "messages": updated_messages,
    #     "existing": existing_memories
    # }, config=config)

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"], strict=False):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Respond to the tool call made in tasks_democracy_ai, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add to the ToolMessage returned to tasks_democracy_ai
    todo_update_msg: str = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [{
            "role": "tool",
            "content": todo_update_msg,
            "tool_call_id": tool_calls[0]['id']
        }]
    }

def update_instructions(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> dict[str, list[dict[str, str]]]:
    """Reflect on the chat history and update the instructions in memory.

    This function processes the chat history to extract and update user-specified
    preferences for managing the todo list. It stores these instructions for future
    reference.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Dict containing a tool message confirming the instructions update
        Format: {
            "messages": [{
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": str
            }]
        }
    """
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    model = _utils.make_chat_model(configurable.model) # pylint: disable=no-member

    ## Create the Trustcall extractors for updating the user profile and ToDo list
    profile_extractor = create_extractor(
        model,
        tools=[Profile],
        tool_choice="Profile",
    )

    namespace: tuple[str, str] = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg: str = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )

    # # # Respond using memory as well as the chat history
    new_memory: BaseMessage = model.invoke(
        [SystemMessage(content=system_msg)] +
        state['messages'][:-1] +
        [HumanMessage(content="Please update the instructions based on the conversation")],
        config=config
    )
    # # Respond using memory as well as the chat history
    # new_memory: BaseMessage = await model.ainvoke(
    #     [SystemMessage(content=system_msg)] +
    #     state['messages'][:-1] +
    #     [HumanMessage(content="Please update the instructions based on the conversation")],
    #     config=config
    # )

    # Overwrite the existing memory in the store
    key: str = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})

    tool_calls = state['messages'][-1].tool_calls

    # Return tool message with update verification
    return {
        "messages": [{
            "role": "tool",
            "content": "updated instructions",
            "tool_call_id": tool_calls[0]['id']
        }]
    }

# Conditional edge
def route_message(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore
) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Route messages to appropriate memory update functions based on tool call type.

    This function examines the latest message in the state and determines which memory
    update function should handle it based on the tool call's update_type.

    Args:
        state: Current message state containing chat history
        config: Configuration object containing user settings and preferences
        store: Storage interface for accessing and managing memories

    Returns:
        Literal indicating which node should process the message next:
        - END: No tool calls present
        - "update_todos": Route to todo list update
        - "update_instructions": Route to instructions update
        - "update_profile": Route to profile update

    Raises:
        ValueError: If the tool call's update_type is not recognized
    """
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError("Unknown update_type in tool call")

# SOURCE: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/create-react-agent-memory.ipynb
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(tasks_democracy_ai)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow
builder.add_edge(START, "tasks_democracy_ai")
builder.add_conditional_edges("tasks_democracy_ai", route_message)
builder.add_edge("update_todos", "tasks_democracy_ai")
builder.add_edge("update_profile", "tasks_democracy_ai")
builder.add_edge("update_instructions", "tasks_democracy_ai")


# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph
# graph: CompiledStateGraph = builder.compile()
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory, debug=True)
graph.name = "DemocracyExeAI"

print(graph.get_graph().print_ascii())

if __name__ == "__main__":  # pragma: no cover
    import rich
    # config = {}
    # configurable = Configuration.from_runnable_config(config)
    # rich.print(f"configurable: {configurable}")
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="Hi, my name is Heron and I like apple pie")]

    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
