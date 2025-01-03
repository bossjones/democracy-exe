"""
This type stub file was generated by pyright.
"""

""" Helper to enable simple lazy module import.

    'Lazy' means the actual import is deferred until an attribute is
    requested from the module's namespace. This has the advantage of
    allowing all imports to be done at the top of a script (in a
    prominent and visible place) without having a great impact
    on startup time.

    Copyright (c) 1999-2005, Marc-Andre Lemburg; mailto:mal@lemburg.com
    See the documentation for further information on copyrights,
    or contact the author. All Rights Reserved.
"""
_debug = ...
class LazyModule:
    """Lazy module class.

    Lazy modules are imported into the given namespaces whenever a
    non-special attribute (there are some attributes like __doc__
    that class instances handle without calling __getattr__) is
    requested. The module is then registered under the given name
    in locals usually replacing the import wrapper instance. The
    import itself is done using globals as global namespace.

    Example of creating a lazy load module:

    ISO = LazyModule('ISO',locals(),globals())

    Later, requesting an attribute from ISO will load the module
    automatically into the locals() namespace, overriding the
    LazyModule instance:

    t = ISO.Week(1998,1,1)

    """
    __lazymodule_init = ...
    __lazymodule_name = ...
    __lazymodule_loaded = ...
    __lazymodule_locals = ...
    __lazymodule_globals = ...
    def __init__(self, name, locals, globals=...) -> None:
        """Create a LazyModule instance wrapping module name.

        The module will later on be registered in locals under the
        given module name.

        globals is optional and defaults to locals.

        """
        ...
    
    def __getattr__(self, name): # -> Any:
        """Import the module on demand and get the attribute."""
        ...
    
    def __setattr__(self, name, value): # -> None:
        """Import the module on demand and set the attribute."""
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    


