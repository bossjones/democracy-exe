"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Any, Iterable, Tuple, Union
from .typing import ExcInfo

"""
Extract a structured traceback from an exception.

Based on work by Will McGugan
<https://github.com/hynek/structlog/pull/407#issuecomment-1150926246>`_ from
`rich.traceback
<https://github.com/Textualize/rich/blob/972dedff/rich/traceback.py>`_.
"""
__all__ = ["ExceptionDictTransformer", "Frame", "Stack", "SyntaxError_", "Trace", "extract", "safe_str", "to_repr"]
SHOW_LOCALS = ...
LOCALS_MAX_LENGTH = ...
LOCALS_MAX_STRING = ...
MAX_FRAMES = ...
OptExcInfo = Union[ExcInfo, Tuple[None, None, None]]
@dataclass
class Frame:
    """
    Represents a single stack frame.
    """
    filename: str
    lineno: int
    name: str
    locals: dict[str, str] | None = ...


@dataclass
class SyntaxError_:
    """
    Contains detailed information about :exc:`SyntaxError` exceptions.
    """
    offset: int
    filename: str
    line: str
    lineno: int
    msg: str
    ...


@dataclass
class Stack:
    """
    Represents an exception and a list of stack frames.
    """
    exc_type: str
    exc_value: str
    syntax_error: SyntaxError_ | None = ...
    is_cause: bool = ...
    frames: list[Frame] = ...


@dataclass
class Trace:
    """
    Container for a list of stack traces.
    """
    stacks: list[Stack]
    ...


def safe_str(_object: Any) -> str:
    """Don't allow exceptions from __str__ to propagate."""
    ...

def to_repr(obj: Any, max_length: int | None = ..., max_string: int | None = ..., use_rich: bool = ...) -> str:
    """
    Get repr string for an object, but catch errors.

    :func:`repr()` is used for strings, too, so that secret wrappers that
    inherit from :func:`str` and overwrite ``__repr__()`` are handled correctly
    (i.e. secrets are not logged in plain text).

    Args:
        obj: Object to get a string representation for.

        max_length: Maximum length of containers before abbreviating, or
            ``None`` for no abbreviation.

        max_string: Maximum length of string before truncating, or ``None`` to
            disable truncating.

        use_rich: If ``True`` (the default), use rich_ to compute the repr.
            If ``False`` or if rich_ is not installed, fall back to a simpler
            algorithm.

    Returns:
        The string representation of *obj*.

    .. versionchanged:: 24.3.0
       Added *max_length* argument.  Use :program:`rich` to render locals if it
       is available.  Call :func:`repr()` on strings in fallback
       implementation.
    """
    ...

def extract(exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType | None, *, show_locals: bool = ..., locals_max_length: int = ..., locals_max_string: int = ..., locals_hide_dunder: bool = ..., locals_hide_sunder: bool = ..., use_rich: bool = ...) -> Trace:
    """
    Extract traceback information.

    Args:
        exc_type: Exception type.

        exc_value: Exception value.

        traceback: Python Traceback object.

        show_locals: Enable display of local variables. Defaults to False.

        locals_max_length:
            Maximum length of containers before abbreviating, or ``None`` for
            no abbreviation.

        locals_max_string:
            Maximum length of string before truncating, or ``None`` to disable
            truncating.

        locals_hide_dunder:
            Hide locals prefixed with double underscore.
            Defaults to True.

        locals_hide_sunder:
            Hide locals prefixed with single underscore.
            This implies hiding *locals_hide_dunder*.
            Defaults to False.

        use_rich: If ``True`` (the default), use rich_ to compute the repr.
            If ``False`` or if rich_ is not installed, fall back to a simpler
            algorithm.

    Returns:
        A Trace instance with structured information about all exceptions.

    .. versionadded:: 22.1.0
    .. versionchanged:: 24.3.0
       Added *locals_max_length*, *locals_hide_sunder*, *locals_hide_dunder*
       and *use_rich* arguments.
    """
    ...

class ExceptionDictTransformer:
    """
    Return a list of exception stack dictionaries for an exception.

    These dictionaries are based on :class:`Stack` instances generated by
    :func:`extract()` and can be dumped to JSON.

    Args:
        show_locals:
            Whether or not to include the values of a stack frame's local
            variables.

        locals_max_length:
            Maximum length of containers before abbreviating, or ``None`` for
            no abbreviation.

        locals_max_string:
            Maximum length of string before truncating, or ``None`` to disable
            truncating.

        locals_hide_dunder:
            Hide locals prefixed with double underscore.
            Defaults to True.

        locals_hide_sunder:
            Hide locals prefixed with single underscore.
            This implies hiding *locals_hide_dunder*.
            Defaults to False.

        suppress:
            Optional sequence of modules or paths for which to suppress the
            display of locals even if *show_locals* is ``True``.

        max_frames:
            Maximum number of frames in each stack.  Frames are removed from
            the inside out.  The idea is, that the first frames represent your
            code responsible for the exception and last frames the code where
            the exception actually happened.  With larger web frameworks, this
            does not always work, so you should stick with the default.

        use_rich: If ``True`` (the default), use rich_ to compute the repr of
            locals.  If ``False`` or if rich_ is not installed, fall back to
            a simpler algorithm.

    .. seealso::
        :doc:`exceptions` for a broader explanation of *structlog*'s exception
        features.

    .. versionchanged:: 24.3.0
       Added *locals_max_length*, *locals_hide_sunder*, *locals_hide_dunder*,
       *suppress* and *use_rich* arguments.
    """
    def __init__(self, *, show_locals: bool = ..., locals_max_length: int = ..., locals_max_string: int = ..., locals_hide_dunder: bool = ..., locals_hide_sunder: bool = ..., suppress: Iterable[str | ModuleType] = ..., max_frames: int = ..., use_rich: bool = ...) -> None:
        ...
    
    def __call__(self, exc_info: ExcInfo) -> list[dict[str, Any]]:
        ...
    


