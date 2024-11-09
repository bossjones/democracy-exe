"""
This type stub file was generated by pyright.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Callable, Dict, Optional, Type, Union, overload

from langchain_core.runnables import Runnable
from langchain_core.runnables.base import Input, Output
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.write import ChannelWrite
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer, StateSnapshot, StreamMode
from pydantic import BaseModel
from typing_extensions import Self

WriteValue = Union[Callable[[Input], Output], Any]
class Channel:
    @overload
    @classmethod
    def subscribe_to(cls, channels: str, *, key: Optional[str] = ..., tags: Optional[list[str]] = ...) -> PregelNode:
        ...

    @overload
    @classmethod
    def subscribe_to(cls, channels: Sequence[str], *, key: None = ..., tags: Optional[list[str]] = ...) -> PregelNode:
        ...

    @classmethod
    def subscribe_to(cls, channels: Union[str, Sequence[str]], *, key: Optional[str] = ..., tags: Optional[list[str]] = ...) -> PregelNode:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        ...

    @classmethod
    def write_to(cls, *channels: str, **kwargs: WriteValue) -> ChannelWrite:
        """Writes to channels the result of the lambda, or None to skip writing."""
        ...



class Pregel(Runnable[Union[dict[str, Any], Any], Union[dict[str, Any], Any]]):
    nodes: dict[str, PregelNode]
    channels: dict[str, Union[BaseChannel, ManagedValueSpec]]
    stream_mode: StreamMode = ...
    output_channels: Union[str, Sequence[str]]
    stream_channels: Optional[Union[str, Sequence[str]]] = ...
    interrupt_after_nodes: Union[All, Sequence[str]]
    interrupt_before_nodes: Union[All, Sequence[str]]
    input_channels: Union[str, Sequence[str]]
    step_timeout: Optional[float] = ...
    debug: bool
    checkpointer: Checkpointer = ...
    store: Optional[BaseStore] = ...
    retry_policy: Optional[RetryPolicy] = ...
    config_type: Optional[Type[Any]] = ...
    config: Optional[RunnableConfig] = ...
    name: str = ...
    def __init__(self, *, nodes: dict[str, PregelNode], channels: Optional[dict[str, Union[BaseChannel, ManagedValueSpec]]], auto_validate: bool = ..., stream_mode: StreamMode = ..., output_channels: Union[str, Sequence[str]], stream_channels: Optional[Union[str, Sequence[str]]] = ..., interrupt_after_nodes: Union[All, Sequence[str]] = ..., interrupt_before_nodes: Union[All, Sequence[str]] = ..., input_channels: Union[str, Sequence[str]], step_timeout: Optional[float] = ..., debug: Optional[bool] = ..., checkpointer: Optional[BaseCheckpointSaver] = ..., store: Optional[BaseStore] = ..., retry_policy: Optional[RetryPolicy] = ..., config_type: Optional[Type[Any]] = ..., config: Optional[RunnableConfig] = ..., name: str = ...) -> None:
        ...

    def copy(self, update: dict[str, Any] | None = ...) -> Self:
        ...

    def with_config(self, config: RunnableConfig | None = ..., **kwargs: Any) -> Self:
        ...

    def validate(self) -> Self:
        ...

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        ...

    @property
    def InputType(self) -> Any:
        ...

    def get_input_schema(self, config: Optional[RunnableConfig] = ...) -> Type[BaseModel]:
        ...

    def get_input_jsonschema(self, config: Optional[RunnableConfig] = ...) -> Dict[All, Any]:
        ...

    @property
    def OutputType(self) -> Any:
        ...

    def get_output_schema(self, config: Optional[RunnableConfig] = ...) -> Type[BaseModel]:
        ...

    def get_output_jsonschema(self, config: Optional[RunnableConfig] = ...) -> Dict[All, Any]:
        ...

    @property
    def stream_channels_list(self) -> Sequence[str]:
        ...

    @property
    def stream_channels_asis(self) -> Union[str, Sequence[str]]:
        ...

    def get_subgraphs(self, *, namespace: Optional[str] = ..., recurse: bool = ...) -> Iterator[tuple[str, Pregel]]:
        ...

    async def aget_subgraphs(self, *, namespace: Optional[str] = ..., recurse: bool = ...) -> AsyncIterator[tuple[str, Pregel]]:
        ...

    def get_state(self, config: RunnableConfig, *, subgraphs: bool = ...) -> StateSnapshot:
        """Get the current state of the graph."""
        ...

    async def aget_state(self, config: RunnableConfig, *, subgraphs: bool = ...) -> StateSnapshot:
        """Get the current state of the graph."""
        ...

    def get_state_history(self, config: RunnableConfig, *, filter: Optional[Dict[str, Any]] = ..., before: Optional[RunnableConfig] = ..., limit: Optional[int] = ...) -> Iterator[StateSnapshot]:
        """Get the history of the state of the graph."""
        ...

    async def aget_state_history(self, config: RunnableConfig, *, filter: Optional[Dict[str, Any]] = ..., before: Optional[RunnableConfig] = ..., limit: Optional[int] = ...) -> AsyncIterator[StateSnapshot]:
        """Get the history of the state of the graph."""
        ...

    def update_state(self, config: RunnableConfig, values: Optional[Union[dict[str, Any], Any]], as_node: Optional[str] = ...) -> RunnableConfig:
        """Update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        ...

    async def aupdate_state(self, config: RunnableConfig, values: dict[str, Any] | Any, as_node: Optional[str] = ...) -> RunnableConfig:
        ...

    def stream(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = ..., *, stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = ..., output_keys: Optional[Union[str, Sequence[str]]] = ..., interrupt_before: Optional[Union[All, Sequence[str]]] = ..., interrupt_after: Optional[Union[All, Sequence[str]]] = ..., debug: Optional[bool] = ..., subgraphs: bool = ...) -> Iterator[Union[dict[str, Any], Any]]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.
            subgraphs: Whether to stream subgraphs, defaults to False.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.

        Examples:
            Using different stream modes with a graph:
            ```pycon
            >>> import operator
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph
            >>> from langgraph.constants import START
            ...
            >>> class State(TypedDict):
            ...     alist: Annotated[list, operator.add]
            ...     another_list: Annotated[list, operator.add]
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", lambda _state: {"another_list": ["hi"]})
            >>> builder.add_node("b", lambda _state: {"alist": ["there"]})
            >>> builder.add_edge("a", "b")
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ```
            With stream_mode="values":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
            ...     print(event)
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': []}, 'triggers': ['start:a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'result': [('another_list', ['hi'])]}}
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': ['hi']}, 'triggers': ['a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'result': [('alist', ['there'])]}}
            ```
        """
        ...

    async def astream(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = ..., *, stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = ..., output_keys: Optional[Union[str, Sequence[str]]] = ..., interrupt_before: Optional[Union[All, Sequence[str]]] = ..., interrupt_after: Optional[Union[All, Sequence[str]]] = ..., debug: Optional[bool] = ..., subgraphs: bool = ...) -> AsyncIterator[Union[dict[str, Any], Any]]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.
            subgraphs: Whether to stream subgraphs, defaults to False.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.

        Examples:
            Using different stream modes with a graph:
            ```pycon
            >>> import operator
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph
            >>> from langgraph.constants import START
            ...
            >>> class State(TypedDict):
            ...     alist: Annotated[list, operator.add]
            ...     another_list: Annotated[list, operator.add]
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", lambda _state: {"another_list": ["hi"]})
            >>> builder.add_node("b", lambda _state: {"alist": ["there"]})
            >>> builder.add_edge("a", "b")
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ```
            With stream_mode="values":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
            ...     print(event)
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': []}, 'triggers': ['start:a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'result': [('another_list', ['hi'])]}}
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': ['hi']}, 'triggers': ['a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'result': [('alist', ['there'])]}}
            ```
        """
        ...

    def invoke(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = ..., *, stream_mode: StreamMode = ..., output_keys: Optional[Union[str, Sequence[str]]] = ..., interrupt_before: Optional[Union[All, Sequence[str]]] = ..., interrupt_after: Optional[Union[All, Sequence[str]]] = ..., debug: Optional[bool] = ..., **kwargs: Any) -> Union[dict[str, Any], Any]:
        """Run the graph with a single input and config.

        Args:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            stream_mode: Optional[str]. The stream mode for the graph run. Default is "values".
            output_keys: Optional. The output keys to retrieve from the graph run.
            interrupt_before: Optional. The nodes to interrupt the graph run before.
            interrupt_after: Optional. The nodes to interrupt the graph run after.
            debug: Optional. Enable debug mode for the graph run.
            **kwargs: Additional keyword arguments to pass to the graph run.

        Returns:
            The output of the graph run. If stream_mode is "values", it returns the latest output.
            If stream_mode is not "values", it returns a list of output chunks.
        """
        ...

    async def ainvoke(self, input: Union[dict[str, Any], Any], config: Optional[RunnableConfig] = ..., *, stream_mode: StreamMode = ..., output_keys: Optional[Union[str, Sequence[str]]] = ..., interrupt_before: Optional[Union[All, Sequence[str]]] = ..., interrupt_after: Optional[Union[All, Sequence[str]]] = ..., debug: Optional[bool] = ..., **kwargs: Any) -> Union[dict[str, Any], Any]:
        """Asynchronously invoke the graph on a single input.

        Args:
            input: The input data for the computation. It can be a dictionary or any other type.
            config: Optional. The configuration for the computation.
            stream_mode: Optional. The stream mode for the computation. Default is "values".
            output_keys: Optional. The output keys to include in the result. Default is None.
            interrupt_before: Optional. The nodes to interrupt before. Default is None.
            interrupt_after: Optional. The nodes to interrupt after. Default is None.
            debug: Optional. Whether to enable debug mode. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the computation. If stream_mode is "values", it returns the latest value.
            If stream_mode is "chunks", it returns a list of chunks.
        """
        ...
