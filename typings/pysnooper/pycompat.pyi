"""
This type stub file was generated by pyright.
"""

import abc
import os
import sys

'''Python 2/3 compatibility'''
PY3 = ...
PY2 = ...
if hasattr(abc, 'ABC'):
    ABC = ...
else:
    class ABC:
        """Helper class that provides a standard way to create an ABC using
        inheritance.
        """
        __metaclass__ = abc.ABCMeta
        __slots__ = ...
    
    
if hasattr(os, 'PathLike'):
    PathLike = ...
else:
    class PathLike(ABC):
        """Abstract base class for implementing the file system path protocol."""
        @abc.abstractmethod
        def __fspath__(self):
            """Return the file system path representation of the object."""
            ...
        
        @classmethod
        def __subclasshook__(cls, subclass): # -> bool:
            ...
        
    
    
iscoroutinefunction = ...
isasyncgenfunction = ...
if PY3:
    string_types = ...
    text_type = ...
    binary_type = ...
else:
    string_types = ...
    text_type = ...
    binary_type = ...
if sys.version_info[: 2] >= (3, 6):
    time_isoformat = ...
else:
    def time_isoformat(time, timespec=...): # -> str:
        ...
    
def timedelta_format(timedelta): # -> str:
    ...

def timedelta_parse(s): # -> timedelta:
    ...

