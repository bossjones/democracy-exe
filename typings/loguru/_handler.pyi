"""
This type stub file was generated by pyright.
"""

def prepare_colored_format(format_, ansi_level): # -> tuple[ColoredFormat, Any | Literal['']]:
    ...

def prepare_stripped_format(format_): # -> Literal['']:
    ...

def memoize(function): # -> _lru_cache_wrapper[Any]:
    ...

class Message(str):
    __slots__ = ...


class Handler:
    def __init__(self, *, sink, name, levelno, formatter, is_formatter_dynamic, filter_, colorize, serialize, enqueue, multiprocessing_context, error_interceptor, exception_formatter, id_, levels_ansi_codes) -> None:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def emit(self, record, level_id, from_decorator, is_raw, colored_message):
        ...
    
    def stop(self): # -> None:
        ...
    
    def complete_queue(self): # -> None:
        ...
    
    def tasks_to_complete(self): # -> list[Any]:
        ...
    
    def update_format(self, level_id): # -> None:
        ...
    
    @property
    def levelno(self): # -> Any:
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    

