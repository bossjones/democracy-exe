"""
This type stub file was generated by pyright.
"""

import collections
import collections.abc
import ipaddress
import numbers
import re
from string import hexdigits

"""RFC 3986 compliant, scheme-agnostic replacement for `urllib.parse`.

This module defines RFC 3986 compliant replacements for the most
commonly used functions of the Python Standard Library
:mod:`urllib.parse` module.

"""
__all__ = ("GEN_DELIMS", "RESERVED", "SUB_DELIMS", "UNRESERVED", "isabspath", "isabsuri", "isnetpath", "isrelpath", "issamedoc", "isuri", "uricompose", "uridecode", "uridefrag", "uriencode", "urijoin", "urisplit", "uriunsplit")
__version__ = ...
GEN_DELIMS = ...
SUB_DELIMS = ...
RESERVED = ...
UNRESERVED = ...
_unreserved = ...
_encoded = ...
_decoded = ...
def uriencode(uristring, safe=..., encoding=..., errors=...): # -> bytes:
    """Encode a URI string or string component."""
    ...

def uridecode(uristring, encoding=..., errors=...): # -> str | bytes:
    """Decode a URI string or string component."""
    ...

class DefragResult(collections.namedtuple("DefragResult", "uri fragment")):
    """Class to hold :func:`uridefrag` results."""
    __slots__ = ...
    def geturi(self):
        """Return the recombined version of the original URI as a string."""
        ...
    
    def getfragment(self, default=..., encoding=..., errors=...): # -> str | bytes | None:
        """Return the decoded fragment identifier, or `default` if the
        original URI did not contain a fragment component.

        """
        ...
    


class SplitResult(collections.namedtuple("SplitResult", "scheme authority path query fragment")):
    """Base class to hold :func:`urisplit` results."""
    __slots__ = ...
    @property
    def userinfo(self): # -> None:
        ...
    
    @property
    def host(self): # -> None:
        ...
    
    @property
    def port(self): # -> None:
        ...
    
    def geturi(self):
        """Return the re-combined version of the original URI reference as a
        string.

        """
        ...
    
    def getscheme(self, default=...): # -> str | None:
        """Return the URI scheme in canonical (lowercase) form, or `default`
        if the original URI reference did not contain a scheme component.

        """
        ...
    
    def getauthority(self, default=..., encoding=..., errors=...): # -> tuple[Any | str | bytes | None, Any | IPv6Address | IPv4Address | bytes | str | None, int | Any | None]:
        """Return the decoded userinfo, host and port subcomponents of the URI
        authority as a three-item tuple.

        """
        ...
    
    def getuserinfo(self, default=..., encoding=..., errors=...): # -> str | bytes | None:
        """Return the decoded userinfo subcomponent of the URI authority, or
        `default` if the original URI reference did not contain a
        userinfo field.

        """
        ...
    
    def gethost(self, default=..., errors=...): # -> IPv6Address | IPv4Address | bytes | str | None:
        """Return the decoded host subcomponent of the URI authority as a
        string or an :mod:`ipaddress` address object, or `default` if
        the original URI reference did not contain a host.

        """
        ...
    
    def getport(self, default=...): # -> int | None:
        """Return the port subcomponent of the URI authority as an
        :class:`int`, or `default` if the original URI reference did
        not contain a port or if the port was empty.

        """
        ...
    
    def getpath(self, encoding=..., errors=...): # -> str | bytes:
        """Return the normalized decoded URI path."""
        ...
    
    def getquery(self, default=..., encoding=..., errors=...): # -> str | bytes | None:
        """Return the decoded query string, or `default` if the original URI
        reference did not contain a query component.

        """
        ...
    
    def getquerydict(self, sep=..., encoding=..., errors=...): # -> defaultdict[Any, list[Any]]:
        """Split the query component into individual `name=value` pairs
        separated by `sep` and return a dictionary of query variables.
        The dictionary keys are the unique query variable names and
        the values are lists of values for each name.

        """
        ...
    
    def getquerylist(self, sep=..., encoding=..., errors=...): # -> list[Any]:
        """Split the query component into individual `name=value` pairs
        separated by `sep`, and return a list of `(name, value)`
        tuples.

        """
        ...
    
    def getfragment(self, default=..., encoding=..., errors=...): # -> str | bytes | None:
        """Return the decoded fragment identifier, or `default` if the
        original URI reference did not contain a fragment component.

        """
        ...
    
    def isuri(self): # -> bool:
        """Return :const:`True` if this is a URI."""
        ...
    
    def isabsuri(self): # -> bool:
        """Return :const:`True` if this is an absolute URI."""
        ...
    
    def isnetpath(self): # -> bool:
        """Return :const:`True` if this is a network-path reference."""
        ...
    
    def isabspath(self): # -> Literal[False]:
        """Return :const:`True` if this is an absolute-path reference."""
        ...
    
    def isrelpath(self): # -> bool:
        """Return :const:`True` if this is a relative-path reference."""
        ...
    
    def issamedoc(self): # -> bool:
        """Return :const:`True` if this is a same-document reference."""
        ...
    
    def transform(self, ref, strict=...): # -> Self:
        """Transform a URI reference relative to `self` into a
        :class:`SplitResult` representing its target URI.

        """
        ...
    


class SplitResultBytes(SplitResult):
    __slots__ = ...
    RE = ...
    DIGITS = ...


class SplitResultString(SplitResult):
    __slots__ = ...
    RE = ...
    DIGITS = ...


def uridefrag(uristring): # -> DefragResult:
    """Remove an existing fragment component from a URI reference string."""
    ...

def urisplit(uristring): # -> SplitResultBytes | SplitResultString:
    """Split a well-formed URI reference string into a tuple with five
    components corresponding to a URI's general structure::

      <scheme>://<authority>/<path>?<query>#<fragment>

    """
    ...

def uriunsplit(parts):
    """Combine the elements of a five-item iterable into a URI reference's
    string representation.

    """
    ...

def urijoin(base, ref, strict=...):
    """Convert a URI reference relative to a base URI to its target URI
    string.

    """
    ...

def isuri(uristring): # -> bool:
    """Return :const:`True` if `uristring` is a URI."""
    ...

def isabsuri(uristring): # -> bool:
    """Return :const:`True` if `uristring` is an absolute URI."""
    ...

def isnetpath(uristring): # -> bool:
    """Return :const:`True` if `uristring` is a network-path reference."""
    ...

def isabspath(uristring): # -> Literal[False]:
    """Return :const:`True` if `uristring` is an absolute-path reference."""
    ...

def isrelpath(uristring): # -> bool:
    """Return :const:`True` if `uristring` is a relative-path reference."""
    ...

def issamedoc(uristring): # -> bool:
    """Return :const:`True` if `uristring` is a same-document reference."""
    ...

_SCHEME_RE = ...
_AUTHORITY_RE_BYTES = ...
_AUTHORITY_RE_STR = ...
_SAFE_USERINFO = ...
_SAFE_HOST = ...
_SAFE_PATH = ...
_SAFE_QUERY = ...
_SAFE_FRAGMENT = ...
def uricompose(scheme=..., authority=..., path=..., query=..., fragment=..., userinfo=..., host=..., port=..., querysep=..., encoding=...):
    """Compose a URI reference string from its individual components."""
    ...

