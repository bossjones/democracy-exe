"""
This type stub file was generated by pyright.
"""

import wrapt

log = ...
class CassetteContextDecorator:
    """Context manager/decorator that handles installing the cassette and
    removing cassettes.

    This class defers the creation of a new cassette instance until
    the point at which it is installed by context manager or
    decorator. The fact that a new cassette is used with each
    application prevents the state of any cassette from interfering
    with another.

    Instances of this class are NOT reentrant as context managers.
    However, functions that are decorated by
    ``CassetteContextDecorator`` instances ARE reentrant. See the
    implementation of ``__call__`` on this class for more details.
    There is also a guard against attempts to reenter instances of
    this class as a context manager in ``__exit__``.
    """
    _non_cassette_arguments = ...
    @classmethod
    def from_args(cls, cassette_class, **kwargs): # -> Self:
        ...
    
    def __init__(self, cls, args_getter) -> None:
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *exc_info): # -> None:
        ...
    
    @wrapt.decorator
    def __call__(self, function, instance, args, kwargs): # -> Coroutine[Any, Any, Any] | Generator[Any, Any, Any]:
        ...
    
    @staticmethod
    def get_function_name(function):
        ...
    


class Cassette:
    """A container for recorded requests and responses"""
    @classmethod
    def load(cls, **kwargs): # -> Self:
        """Instantiate and load the cassette stored at the specified path."""
        ...
    
    @classmethod
    def use_arg_getter(cls, arg_getter): # -> CassetteContextDecorator:
        ...
    
    @classmethod
    def use(cls, **kwargs): # -> CassetteContextDecorator:
        ...
    
    def __init__(self, path, serializer=..., persister=..., record_mode=..., match_on=..., before_record_request=..., before_record_response=..., custom_patches=..., inject=..., allow_playback_repeats=...) -> None:
        ...
    
    @property
    def play_count(self): # -> int:
        ...
    
    @property
    def all_played(self): # -> bool:
        """Returns True if all responses have been played, False otherwise."""
        ...
    
    @property
    def requests(self): # -> list[Any]:
        ...
    
    @property
    def responses(self): # -> list[Any]:
        ...
    
    @property
    def write_protected(self): # -> bool:
        ...
    
    def append(self, request, response): # -> None:
        """Add a request, response pair to this cassette"""
        ...
    
    def filter_request(self, request):
        ...
    
    def can_play_response_for(self, request): # -> bool:
        ...
    
    def play_response(self, request):
        """
        Get the response corresponding to a request, but only if it
        hasn't been played back before, and mark it as played
        """
        ...
    
    def responses_of(self, request): # -> list[Any]:
        """
        Find the responses corresponding to a request.
        This function isn't actually used by VCR internally, but is
        provided as an external API.
        """
        ...
    
    def rewind(self): # -> None:
        ...
    
    def find_requests_with_most_matches(self, request): # -> list[Any]:
        """
        Get the most similar request(s) stored in the cassette
        of a given request as a list of tuples like this:
        - the request object
        - the successful matchers as string
        - the failed matchers and the related assertion message with the difference details as strings tuple

        This is useful when a request failed to be found,
        we can get the similar request(s) in order to know what have changed in the request parts.
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    def __len__(self): # -> int:
        """Return the number of request,response pairs stored in here"""
        ...
    
    def __contains__(self, request): # -> bool:
        """Return whether or not a request has been stored"""
        ...
    


