"""
This type stub file was generated by pyright.
"""

from collections.abc import Sequence
from typing import Annotated, Callable, Optional, Type, TypeVar, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph._api.deprecation import deprecated_parameter
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    ...


StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]
STATE_MODIFIER_RUNNABLE_NAME = ...
MessagesModifier = Union[SystemMessage, str, Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]], Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],]
StateModifier = Union[SystemMessage, str, Callable[[StateSchema], Sequence[BaseMessage]], Runnable[StateSchema, Sequence[BaseMessage]],]
@deprecated_parameter("messages_modifier", "0.1.9", "state_modifier", removal="0.3.0")
def create_react_agent(model: LanguageModelLike, tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode], *, state_schema: Optional[StateSchemaType] = ..., messages_modifier: Optional[MessagesModifier] = ..., state_modifier: Optional[StateModifier] = ..., checkpointer: Optional[Checkpointer] = ..., store: Optional[BaseStore] = ..., interrupt_before: Optional[list[str]] = ..., interrupt_after: Optional[list[str]] = ..., debug: bool = ...) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools, a ToolExecutor, or a ToolNode instance.
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `is_last_step` keys.
            Defaults to `AgentState` that defines those two keys.
        messages_modifier: An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
            !!! Warning
                `messages_modifier` parameter is deprecated as of version 0.1.9 and will be removed in 0.2.0
        state_modifier: An optional
            state modifier. This takes full graph state BEFORE the LLM is called and prepares the input to LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "agent", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "agent", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.

    Returns:
        A compiled LangChain runnable that can be used for chat interactions.

    The resulting graph looks like this:

    ``` mermaid
    stateDiagram-v2
        [*] --> Start
        Start --> Agent
        Agent --> Tools : continue
        Tools --> Agent
        Agent --> End : end
        End --> [*]

        classDef startClass fill:#ffdfba;
        classDef endClass fill:#baffc9;
        classDef otherClass fill:#fad7de;

        class Start startClass
        class End endClass
        class Agent,Tools otherClass
    ```

    The "agent" node calls the language model with the messages list (after applying the messages modifier).
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][langgraph.prebuilt.tool_node.ToolNode].
    The "tools" node executes the tools (1 tool per `tool_call`) and adds the responses to the messages list
    as `ToolMessage` objects. The agent node then calls the language model again.
    The process repeats until no more `tool_calls` are present in the response.
    The agent then returns the full list of messages as a dictionary containing the key "messages".

    ``` mermaid
        sequenceDiagram
            participant U as User
            participant A as Agent (LLM)
            participant T as Tools
            U->>A: Initial input
            Note over A: Messages modifier + LLM
            loop while tool_calls present
                A->>T: Execute tools
                T-->>A: ToolMessage for each tool_calls
            end
            A->>U: Return final state
    ```

    Examples:
        Use with a simple tool:

        ```pycon
        >>> from datetime import datetime
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent


        ... def check_weather(location: str, at_time: datetime | None = None) -> str:
        ...     '''Return the weather forecast for the specified location.'''
        ...     return f"It's always sunny in {location}"
        >>>
        >>> tools = [check_weather]
        >>> model = ChatOpenAI(model="gpt-4o")
        >>> graph = create_react_agent(model, tools=tools)
        >>> inputs = {"messages": [("user", "what is the weather in sf")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ('user', 'what is the weather in sf')
        ================================== Ai Message ==================================
        Tool Calls:
        check_weather (call_LUzFvKJRuaWQPeXvBOzwhQOu)
        Call ID: call_LUzFvKJRuaWQPeXvBOzwhQOu
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is sunny.
        ```
        Add a system prompt for the LLM:

        ```pycon
        >>> system_prompt = "You are a helpful bot named Fred."
        >>> graph = create_react_agent(model, tools, state_modifier=system_prompt)
        >>> inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ('user', "What's your name? And what's the weather in SF?")
        ================================== Ai Message ==================================
        Hi, my name is Fred. Let me check the weather in San Francisco for you.
        Tool Calls:
        check_weather (call_lqhj4O0hXYkW9eknB4S41EXk)
        Call ID: call_lqhj4O0hXYkW9eknB4S41EXk
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is currently sunny. If you need any more details or have other questions, feel free to ask!
        ```

        Add a more complex prompt for the LLM:

        ```pycon
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> prompt = ChatPromptTemplate.from_messages([
        ...     ("system", "You are a helpful bot named Fred."),
        ...     ("placeholder", "{messages}"),
        ...     ("user", "Remember, always be polite!"),
        ... ])
        >>> def format_for_model(state: AgentState):
        ...     # You can do more complex modifications here
        ...     return prompt.invoke({"messages": state["messages"]})
        >>>
        >>> graph = create_react_agent(model, tools, state_modifier=format_for_model)
        >>> inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add complex prompt with custom graph state:

        ```pycon
        >>> from typing import TypedDict
        >>> prompt = ChatPromptTemplate.from_messages(
        ...     [
        ...         ("system", "Today is {today}"),
        ...         ("placeholder", "{messages}"),
        ...     ]
        ... )
        >>>
        >>> class CustomState(TypedDict):
        ...     today: str
        ...     messages: Annotated[list[BaseMessage], add_messages]
        ...     is_last_step: str
        >>>
        >>> graph = create_react_agent(model, tools, state_schema=CustomState, state_modifier=prompt)
        >>> inputs = {"messages": [("user", "What's today's date? And what's the weather in SF?")], "today": "July 16, 2004"}
        >>> for s in graph.stream(inputs, stream_mode="values"):
        ...     message = s["messages"][-1]
        ...     if isinstance(message, tuple):
        ...         print(message)
        ...     else:
        ...         message.pretty_print()
        ```

        Add thread-level "chat memory" to the graph:

        ```pycon
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> graph = create_react_agent(model, tools, checkpointer=MemorySaver())
        >>> config = {"configurable": {"thread_id": "thread-1"}}
        >>> def print_stream(graph, inputs, config):
        ...     for s in graph.stream(inputs, config, stream_mode="values"):
        ...         message = s["messages"][-1]
        ...         if isinstance(message, tuple):
        ...             print(message)
        ...         else:
        ...             message.pretty_print()
        >>> inputs = {"messages": [("user", "What's the weather in SF?")]}
        >>> print_stream(graph, inputs, config)
        >>> inputs2 = {"messages": [("user", "Cool, so then should i go biking today?")]}
        >>> print_stream(graph, inputs2, config)
        ('user', "What's the weather in SF?")
        ================================== Ai Message ==================================
        Tool Calls:
        check_weather (call_ChndaktJxpr6EMPEB5JfOFYc)
        Call ID: call_ChndaktJxpr6EMPEB5JfOFYc
        Args:
            location: San Francisco
        ================================= Tool Message =================================
        Name: check_weather
        It's always sunny in San Francisco
        ================================== Ai Message ==================================
        The weather in San Francisco is sunny. Enjoy your day!
        ================================ Human Message =================================
        Cool, so then should i go biking today?
        ================================== Ai Message ==================================
        Since the weather in San Francisco is sunny, it sounds like a great day for biking! Enjoy your ride!
        ```

        Add an interrupt to let the user confirm before taking an action:

        ```pycon
        >>> graph = create_react_agent(
        ...     model, tools, interrupt_before=["tools"], checkpointer=MemorySaver()
        >>> )
        >>> config = {"configurable": {"thread_id": "thread-1"}}

        >>> inputs = {"messages": [("user", "What's the weather in SF?")]}
        >>> print_stream(graph, inputs, config)
        >>> snapshot = graph.get_state(config)
        >>> print("Next step: ", snapshot.next)
        >>> print_stream(graph, None, config)
        ```

        Add cross-thread memory to the graph:

        ```pycon
        >>> from langgraph.prebuilt import InjectedStore
        >>> from langgraph.store.base import BaseStore

        >>> def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
        ...     '''Save the given memory for the current user.'''
        ...     # This is a **tool** the model can use to save memories to storage
        ...     user_id = config.get("configurable", {}).get("user_id")
        ...     namespace = ("memories", user_id)
        ...     store.put(namespace, f"memory_{len(store.search(namespace))}", {"data": memory})
        ...     return f"Saved memory: {memory}"

        >>> def prepare_model_inputs(state: AgentState, config: RunnableConfig, store: BaseStore):
        ...     # Retrieve user memories and add them to the system message
        ...     # This function is called **every time** the model is prompted. It converts the state to a prompt
        ...     user_id = config.get("configurable", {}).get("user_id")
        ...     namespace = ("memories", user_id)
        ...     memories = [m.value["data"] for m in store.search(namespace)]
        ...     system_msg = f"User memories: {', '.join(memories)}"
        ...     return [{"role": "system", "content": system_msg)] + state["messages"]

        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> from langgraph.store.memory import InMemoryStore
        >>> store = InMemoryStore()
        >>> graph = create_react_agent(model, [save_memory], state_modifier=prepare_model_inputs, store=store, checkpointer=MemorySaver())
        >>> config = {"configurable": {"thread_id": "thread-1", "user_id": "1"}}

        >>> inputs = {"messages": [("user", "Hey I'm Will, how's it going?")]}
        >>> print_stream(graph, inputs, config)
        ('user', "Hey I'm Will, how's it going?")
        ================================== Ai Message ==================================
        Hello Will! It's nice to meet you. I'm doing well, thank you for asking. How are you doing today?

        >>> inputs2 = {"messages": [("user", "I like to bike")]}
        >>> print_stream(graph, inputs2, config)
        ================================ Human Message =================================
        I like to bike
        ================================== Ai Message ==================================
        That's great to hear, Will! Biking is an excellent hobby and form of exercise. It's a fun way to stay active and explore your surroundings. Do you have any favorite biking routes or trails you enjoy? Or perhaps you're into a specific type of biking, like mountain biking or road cycling?

        >>> config = {"configurable": {"thread_id": "thread-2", "user_id": "1"}}
        >>> inputs3 = {"messages": [("user", "Hi there! Remember me?")]}
        >>> print_stream(graph, inputs3, config)
        ================================ Human Message =================================
        Hi there! Remember me?
        ================================== Ai Message ==================================
        User memories:
        Hello! Of course, I remember you, Will! You mentioned earlier that you like to bike. It's great to hear from you again. How have you been? Have you been on any interesting bike rides lately?
        ```

        Add a timeout for a given step:

        ```pycon
        >>> import time
        ... def check_weather(location: str, at_time: datetime | None = None) -> float:
        ...     '''Return the weather forecast for the specified location.'''
        ...     time.sleep(2)
        ...     return f"It's always sunny in {location}"
        >>>
        >>> tools = [check_weather]
        >>> graph = create_react_agent(model, tools)
        >>> graph.step_timeout = 1 # Seconds
        >>> for s in graph.stream({"messages": [("user", "what is the weather in sf")]}):
        ...     print(s)
        TimeoutError: Timed out at step 2
        ```
    """
    ...

create_tool_calling_executor = ...
__all__ = ["create_react_agent", "create_tool_calling_executor", "AgentState"]
