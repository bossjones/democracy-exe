"""
This type stub file was generated by pyright.
"""

from typing import Any, Protocol, TypeVar

__all__ = ["NestedSequence", "SupportsBufferProtocol"]
_T_co = TypeVar("_T_co", covariant=True)
class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]:
        ...
    
    def __len__(self, /) -> int:
        ...
    


SupportsBufferProtocol = Any
Array = Any
Device = Any
