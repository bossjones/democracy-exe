"""
This type stub file was generated by pyright.
"""

import contextlib
import contextvars
from types import FrameType
from typing import Any, Generator, Mapping
from .typing import BindableLogger, EventDict, WrappedLogger

"""
Primitives to deal with a concurrency supporting context, as introduced in
Python 3.7 as :mod:`contextvars`.

.. versionadded:: 20.1.0
.. versionchanged:: 21.1.0
   Reimplemented without using a single dict as context carrier for improved
   isolation. Every key-value pair is a separate `contextvars.ContextVar` now.
.. versionchanged:: 23.3.0
   Callsite parameters are now also collected under asyncio.

See :doc:`contextvars`.
"""
STRUCTLOG_KEY_PREFIX = ...
STRUCTLOG_KEY_PREFIX_LEN = ...
_ASYNC_CALLING_STACK: contextvars.ContextVar[FrameType] = ...
_CONTEXT_VARS: dict[str, contextvars.ContextVar[Any]] = ...
def get_contextvars() -> dict[str, Any]:
    """
    Return a copy of the *structlog*-specific context-local context.

    .. versionadded:: 21.2.0
    """
    ...

def get_merged_contextvars(bound_logger: BindableLogger) -> dict[str, Any]:
    """
    Return a copy of the current context-local context merged with the context
    from *bound_logger*.

    .. versionadded:: 21.2.0
    """
    ...

def merge_contextvars(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    A processor that merges in a global (context-local) context.

    Use this as your first processor in :func:`structlog.configure` to ensure
    context-local context is included in all log calls.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0 See toplevel note.
    """
    ...

def clear_contextvars() -> None:
    """
    Clear the context-local context.

    The typical use-case for this function is to invoke it early in request-
    handling code.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0 See toplevel note.
    """
    ...

def bind_contextvars(**kw: Any) -> Mapping[str, contextvars.Token[Any]]:
    r"""
    Put keys and values into the context-local context.

    Use this instead of :func:`~structlog.BoundLogger.bind` when you want some
    context to be global (context-local).

    Return the mapping of `contextvars.Token`\s resulting
    from setting the backing :class:`~contextvars.ContextVar`\s.
    Suitable for passing to :func:`reset_contextvars`.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0 Return the `contextvars.Token` mapping
        rather than None. See also the toplevel note.
    """
    ...

def reset_contextvars(**kw: contextvars.Token[Any]) -> None:
    r"""
    Reset contextvars corresponding to the given Tokens.

    .. versionadded:: 21.1.0
    """
    ...

def unbind_contextvars(*keys: str) -> None:
    """
    Remove *keys* from the context-local context if they are present.

    Use this instead of :func:`~structlog.BoundLogger.unbind` when you want to
    remove keys from a global (context-local) context.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0 See toplevel note.
    """
    ...

@contextlib.contextmanager
def bound_contextvars(**kw: Any) -> Generator[None, None, None]:
    """
    Bind *kw* to the current context-local context. Unbind or restore *kw*
    afterwards. Do **not** affect other keys.

    Can be used as a context manager or decorator.

    .. versionadded:: 21.4.0
    """
    ...

