"""
This type stub file was generated by pyright.
"""

from uarray import *

from ._uarray import *

"""`uarray` provides functions for generating multimethods that dispatch to
multiple different backends

This should be imported, rather than `_uarray` so that an installed version could
be used instead, if available. This means that users can call
`uarray.set_backend` directly instead of going through SciPy.

"""
if _has_uarray:
    ...
else:
    ...
