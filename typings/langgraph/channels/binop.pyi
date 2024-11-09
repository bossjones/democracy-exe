"""
This type stub file was generated by pyright.
"""

from collections.abc import Sequence
from typing import Callable, Generic, Optional, Type

from langgraph.channels.base import BaseChannel, Value
from typing_extensions import Self

class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the result of applying a binary operator to the current value and each new value.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```
    """
    __slots__ = ...
    def __init__(self, typ: Type[Value], operator: Callable[[Value, Value], Value]) -> None:
        ...

    def __eq__(self, value: object) -> bool:
        ...

    @property
    def ValueType(self) -> Type[Value]:
        """The type of the value stored in the channel."""
        ...

    @property
    def UpdateType(self) -> Type[Value]:
        """The type of the update received by the channel."""
        ...

    def from_checkpoint(self, checkpoint: Optional[Value]) -> Self:
        ...

    def update(self, values: Sequence[Value]) -> bool:
        ...

    def get(self) -> Value:
        ...
