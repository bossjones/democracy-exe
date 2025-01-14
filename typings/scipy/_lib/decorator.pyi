"""
This type stub file was generated by pyright.
"""

from contextlib import _GeneratorContextManager

"""
Decorator module, see https://pypi.python.org/pypi/decorator
for the documentation.
"""
__version__ = ...
def get_init(cls):
    ...

ArgSpec = ...
def getargspec(f): # -> ArgSpec:
    """A replacement for inspect.getargspec"""
    ...

DEF = ...
class FunctionMaker:
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict, and
    methods update and make.
    """
    _compile_count = ...
    def __init__(self, func=..., name=..., signature=..., defaults=..., doc=..., module=..., funcdict=...) -> None:
        ...
    
    def update(self, func, **kw): # -> None:
        "Update the signature of func with the data in self"
        ...
    
    def make(self, src_templ, evaldict=..., addsource=..., **attrs):
        "Make a new function from a given template and update the signature"
        ...
    
    @classmethod
    def create(cls, obj, body, evaldict, defaults=..., doc=..., module=..., addsource=..., **attrs):
        """
        Create a function from the strings name, signature, and body.
        evaldict is the evaluation dictionary. If addsource is true, an
        attribute __source__ is added to the result. The attributes attrs
        are added, if any.
        """
        ...
    


def decorate(func, caller):
    """
    decorate(func, caller) decorates a function using a caller.
    """
    ...

def decorator(caller, _func=...):
    """decorator(caller) converts a caller function into a decorator"""
    ...

class ContextManager(_GeneratorContextManager):
    def __call__(self, func):
        """Context manager decorator"""
        ...
    


init = ...
n_args = ...
if n_args == 2 and not init.varargs:
    def __init__(self, g, *a, **k) -> None:
        ...
    
else:
    ...
contextmanager = ...
def append(a, vancestors): # -> None:
    """
    Append ``a`` to the list of the virtual ancestors, unless it is already
    included.
    """
    ...

def dispatch_on(*dispatch_args): # -> Callable[..., Any]:
    """
    Factory of decorators turning a function into a generic function
    dispatching on the given arguments.
    """
    ...

