"""
This type stub file was generated by pyright.
"""

from .formatter import THEME

"""Beautiful and helpful exceptions

Just set your `BETTER_EXCEPTIONS` environment variable. It handles the rest.


   Name: better_exceptions
 Author: Josh Junon
  Email: josh@junon.me
    URL: github.com/qix-/better-exceptions
License: Copyright (c) 2017 Josh Junon, licensed under the MIT license
"""
__version__ = ...
THEME = ...
def write_stream(data, stream=...): # -> None:
    ...

def format_exception(exc, value, tb): # -> list[Any | str]:
    ...

def excepthook(exc, value, tb): # -> None:
    ...

def hook(): # -> None:
    ...
