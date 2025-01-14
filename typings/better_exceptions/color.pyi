"""
This type stub file was generated by pyright.
"""

import os

"""Checks if the current terminal supports colors.

Also specifies the stream to write to. On Windows, this is a wrapped
stream.
"""
STREAM = ...
SHOULD_ENCODE = ...
SUPPORTS_COLOR = ...
def get_terminfo_file(): # -> BufferedReader | None:
    ...

class ProxyBufferStreamWrapper:
    def __init__(self, wrapped) -> None:
        ...
    
    def __getattr__(self, name): # -> Any:
        ...
    
    def write(self, text): # -> None:
        ...
    


if os.name == 'nt':
    ...
else:
    ...
