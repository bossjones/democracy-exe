"""
This type stub file was generated by pyright.
"""

from collections.abc import Sequence
from typing import Generic, Optional, Type

from langgraph.channels.base import BaseChannel, Value
from typing_extensions import Self

class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received, can receive at most one value per step."""
    __slots__ = ...
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
