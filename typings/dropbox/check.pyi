"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

class EchoArg(bb.Struct):
    """
    Contains the arguments to be sent to the Dropbox servers.

    :ivar check.EchoArg.query: The string that you'd like to be echoed back to
        you.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, query=...) -> None:
        ...
    
    query = ...


EchoArg_validator = ...
class EchoResult(bb.Struct):
    """
    EchoResult contains the result returned from the Dropbox servers.

    :ivar check.EchoResult.result: If everything worked correctly, this would be
        the same as query.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, result=...) -> None:
        ...
    
    result = ...


EchoResult_validator = ...
app = ...
user = ...
ROUTES = ...
