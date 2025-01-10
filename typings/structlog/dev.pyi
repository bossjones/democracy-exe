"""
This type stub file was generated by pyright.
"""

import colorama
import rich
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Literal, Protocol, Sequence, TextIO, Type, Union
from .typing import EventDict, ExcInfo, ExceptionRenderer, WrappedLogger

"""
Helpers that make development with *structlog* more pleasant.

See also the narrative documentation in `console-output`.
"""
__all__ = ["ConsoleRenderer", "plain_traceback", "rich_traceback", "better_traceback"]
_IS_WINDOWS = ...
_MISSING = ...
_EVENT_WIDTH = ...
if colorama is not None:
    RESET_ALL = ...
    BRIGHT = ...
    DIM = ...
    RED = ...
    BLUE = ...
    CYAN = ...
    MAGENTA = ...
    YELLOW = ...
    GREEN = ...
    RED_BACK = ...
else:
    RESET_ALL = ...
    BRIGHT = ...
    DIM = ...
    RED = ...
    BLUE = ...
    CYAN = ...
    MAGENTA = ...
    YELLOW = ...
    GREEN = ...
    RED_BACK = ...
_has_colors = ...
_use_colors = ...
class _Styles(Protocol):
    reset: str
    bright: str
    level_critical: str
    level_exception: str
    level_error: str
    level_warn: str
    level_info: str
    level_debug: str
    level_notset: str
    timestamp: str
    logger_name: str
    kv_key: str
    kv_value: str
    ...


Styles = Union[_Styles, Type[_Styles]]
class _ColorfulStyles:
    reset = ...
    bright = ...
    level_critical = ...
    level_exception = ...
    level_error = ...
    level_warn = ...
    level_info = ...
    level_debug = ...
    level_notset = ...
    timestamp = ...
    logger_name = ...
    kv_key = ...
    kv_value = ...


class _PlainStyles:
    reset = ...
    bright = ...
    level_critical = ...
    level_exception = ...
    level_error = ...
    level_warn = ...
    level_info = ...
    level_debug = ...
    level_notset = ...
    timestamp = ...
    logger_name = ...
    kv_key = ...
    kv_value = ...


class ColumnFormatter(Protocol):
    """
    :class:`~typing.Protocol` for column formatters.

    See `KeyValueColumnFormatter` and `LogLevelColumnFormatter` for examples.

    .. versionadded:: 23.3.0
    """
    def __call__(self, key: str, value: object) -> str:
        """
        Format *value* for *key*.

        This method is responsible for formatting, *key*, the ``=``, and the
        *value*. That means that it can use any string instead of the ``=`` and
        it can leave out both the *key* or the *value*.

        If it returns an empty string, the column is omitted completely.
        """
        ...
    


@dataclass
class Column:
    """
    A column defines the way a key-value pair is formatted, and, by it's
    position to the *columns* argument of `ConsoleRenderer`, the order in which
    it is rendered.

    Args:
        key:
            The key for which this column is responsible. Leave empty to define
            it as the default formatter.

        formatter: The formatter for columns with *key*.

    .. versionadded:: 23.3.0
    """
    key: str
    formatter: ColumnFormatter
    ...


@dataclass
class KeyValueColumnFormatter:
    """
    Format a key-value pair.

    Args:
        key_style: The style to apply to the key. If None, the key is omitted.

        value_style: The style to apply to the value.

        reset_style: The style to apply whenever a style is no longer needed.

        value_repr:
            A callable that returns the string representation of the value.

        width: The width to pad the value to. If 0, no padding is done.

        prefix:
            A string to prepend to the formatted key-value pair. May contain
            styles.

        postfix:
            A string to append to the formatted key-value pair. May contain
            styles.

    .. versionadded:: 23.3.0
    """
    key_style: str | None
    value_style: str
    reset_style: str
    value_repr: Callable[[object], str]
    width: int = ...
    prefix: str = ...
    postfix: str = ...
    def __call__(self, key: str, value: object) -> str:
        ...
    


class LogLevelColumnFormatter:
    """
    Format a log level according to *level_styles*.

    The width is padded to the longest level name (if *level_styles* is passed
    -- otherwise there's no way to know the lengths of all levels).

    Args:
        level_styles:
            A dictionary of level names to styles that are applied to it. If
            None, the level is formatted as a plain ``[level]``.

        reset_style:
            What to use to reset the style after the level name. Ignored if
            if *level_styles* is None.

        width:
            The width to pad the level to. If 0, no padding is done.

    .. versionadded:: 23.3.0
    .. versionadded:: 24.2.0 *width*
    """
    level_styles: dict[str, str] | None
    reset_style: str
    width: int
    def __init__(self, level_styles: dict[str, str], reset_style: str, width: int | None = ...) -> None:
        ...
    
    def __call__(self, key: str, value: object) -> str:
        ...
    


_NOTHING = ...
def plain_traceback(sio: TextIO, exc_info: ExcInfo) -> None:
    """
    "Pretty"-print *exc_info* to *sio* using our own plain formatter.

    To be passed into `ConsoleRenderer`'s ``exception_formatter`` argument.

    Used by default if neither Rich nor *better-exceptions* are present.

    .. versionadded:: 21.2.0
    """
    ...

@dataclass
class RichTracebackFormatter:
    """
    A Rich traceback renderer with the given options.

    Pass an instance as `ConsoleRenderer`'s ``exception_formatter`` argument.

    See :class:`rich.traceback.Traceback` for details on the arguments.

    If a *width* of -1 is passed, the terminal width is used. If the width
    can't be determined, fall back to 80.

    .. versionadded:: 23.2.0
    """
    color_system: Literal["auto", "standard", "256", "truecolor", "windows"] = ...
    show_locals: bool = ...
    max_frames: int = ...
    theme: str | None = ...
    word_wrap: bool = ...
    extra_lines: int = ...
    width: int = ...
    indent_guides: bool = ...
    locals_max_length: int = ...
    locals_max_string: int = ...
    locals_hide_dunder: bool = ...
    locals_hide_sunder: bool = ...
    suppress: Sequence[str | ModuleType] = ...
    def __call__(self, sio: TextIO, exc_info: ExcInfo) -> None:
        ...
    


rich_traceback = ...
def better_traceback(sio: TextIO, exc_info: ExcInfo) -> None:
    """
    Pretty-print *exc_info* to *sio* using the *better-exceptions* package.

    To be passed into `ConsoleRenderer`'s ``exception_formatter`` argument.

    Used by default if *better-exceptions* is installed and Rich is absent.

    .. versionadded:: 21.2.0
    """
    ...

if rich is not None:
    default_exception_formatter = ...
else:
    default_exception_formatter = ...
    default_exception_formatter = ...
class ConsoleRenderer:
    r"""
    Render ``event_dict`` nicely aligned, possibly in colors, and ordered.

    If ``event_dict`` contains a true-ish ``exc_info`` key, it will be rendered
    *after* the log line. If Rich_ or better-exceptions_ are present, in colors
    and with extra context.

    Args:
        columns:
            A list of `Column` objects defining both the order and format of
            the key-value pairs in the output. If passed, most other arguments
            become meaningless.

            **Must** contain a column with ``key=''`` that defines the default
            formatter.

            .. seealso:: `columns-config`

        pad_event:
            Pad the event to this many characters. Ignored if *columns* are
            passed.

        colors:
            Use colors for a nicer output. `True` by default. On Windows only
            if Colorama_ is installed. Ignored if *columns* are passed.

        force_colors:
            Force colors even for non-tty destinations. Use this option if your
            logs are stored in a file that is meant to be streamed to the
            console. Only meaningful on Windows. Ignored if *columns* are
            passed.

        repr_native_str:
            When `True`, `repr` is also applied to ``str``\ s. The ``event``
            key is *never* `repr` -ed. Ignored if *columns* are passed.

        level_styles:
            When present, use these styles for colors. This must be a dict from
            level names (strings) to Colorama styles. The default can be
            obtained by calling `ConsoleRenderer.get_default_level_styles`.
            Ignored when *columns* are passed.

        exception_formatter:
            A callable to render ``exc_infos``. If Rich_ or better-exceptions_
            are installed, they are used for pretty-printing by default (rich_
            taking precedence). You can also manually set it to
            `plain_traceback`, `better_traceback`, an instance of
            `RichTracebackFormatter` like `rich_traceback`, or implement your
            own.

        sort_keys:
            Whether to sort keys when formatting. `True` by default. Ignored if
            *columns* are passed.

        event_key:
            The key to look for the main log message. Needed when you rename it
            e.g. using `structlog.processors.EventRenamer`. Ignored if
            *columns* are passed.

        timestamp_key:
            The key to look for timestamp of the log message. Needed when you
            rename it e.g. using `structlog.processors.EventRenamer`. Ignored
            if *columns* are passed.

        pad_level:
            Whether to pad log level with blanks to the longest amongst all
            level label.

    Requires the Colorama_ package if *colors* is `True` **on Windows**.

    Raises:
        ValueError: If there's not exactly one default column formatter.

    .. _Colorama: https://pypi.org/project/colorama/
    .. _better-exceptions: https://pypi.org/project/better-exceptions/
    .. _Rich: https://pypi.org/project/rich/

    .. versionadded:: 16.0.0
    .. versionadded:: 16.1.0 *colors*
    .. versionadded:: 17.1.0 *repr_native_str*
    .. versionadded:: 18.1.0 *force_colors*
    .. versionadded:: 18.1.0 *level_styles*
    .. versionchanged:: 19.2.0
       Colorama now initializes lazily to avoid unwanted initializations as
       ``ConsoleRenderer`` is used by default.
    .. versionchanged:: 19.2.0 Can be pickled now.
    .. versionchanged:: 20.1.0
       Colorama does not initialize lazily on Windows anymore because it breaks
       rendering.
    .. versionchanged:: 21.1.0
       It is additionally possible to set the logger name using the
       ``logger_name`` key in the ``event_dict``.
    .. versionadded:: 21.2.0 *exception_formatter*
    .. versionchanged:: 21.2.0
       `ConsoleRenderer` now handles the ``exc_info`` event dict key itself. Do
       **not** use the `structlog.processors.format_exc_info` processor
       together with `ConsoleRenderer` anymore! It will keep working, but you
       can't have customize exception formatting and a warning will be raised
       if you ask for it.
    .. versionchanged:: 21.2.0
       The colors keyword now defaults to True on non-Windows systems, and
       either True or False in Windows depending on whether Colorama is
       installed.
    .. versionadded:: 21.3.0 *sort_keys*
    .. versionadded:: 22.1.0 *event_key*
    .. versionadded:: 23.2.0 *timestamp_key*
    .. versionadded:: 23.3.0 *columns*
    .. versionadded:: 24.2.0 *pad_level*
    """
    def __init__(self, pad_event: int = ..., colors: bool = ..., force_colors: bool = ..., repr_native_str: bool = ..., level_styles: Styles | None = ..., exception_formatter: ExceptionRenderer = ..., sort_keys: bool = ..., event_key: str = ..., timestamp_key: str = ..., columns: list[Column] | None = ..., pad_level: bool = ...) -> None:
        ...
    
    def __call__(self, logger: WrappedLogger, name: str, event_dict: EventDict) -> str:
        ...
    
    @staticmethod
    def get_default_level_styles(colors: bool = ...) -> Any:
        """
        Get the default styles for log levels

        This is intended to be used with `ConsoleRenderer`'s ``level_styles``
        parameter.  For example, if you are adding custom levels in your
        home-grown :func:`~structlog.stdlib.add_log_level` you could do::

            my_styles = ConsoleRenderer.get_default_level_styles()
            my_styles["EVERYTHING_IS_ON_FIRE"] = my_styles["critical"] renderer
            = ConsoleRenderer(level_styles=my_styles)

        Args:
            colors:
                Whether to use colorful styles. This must match the *colors*
                parameter to `ConsoleRenderer`. Default: `True`.
        """
        ...
    


_SENTINEL = ...
def set_exc_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Set ``event_dict["exc_info"] = True`` if *method_name* is ``"exception"``.

    Do nothing if the name is different or ``exc_info`` is already set.
    """
    ...

