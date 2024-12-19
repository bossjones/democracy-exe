"""
This type stub file was generated by pyright.
"""

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T", bound=type)
def delegate_to_executor(*attrs: str) -> Callable[[_T], _T]:
    ...

def proxy_method_directly(*attrs: str) -> Callable[[_T], _T]:
    ...

def proxy_property_directly(*attrs: str) -> Callable[[_T], _T]:
    ...

def cond_delegate_to_executor(*attrs: str) -> Callable[[_T], _T]:
    ...

