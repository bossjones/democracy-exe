"""
This type stub file was generated by pyright.
"""

"""
Generic test utilities.

"""
__all__ = ['PytestTester', 'check_free_memory', '_TestPythranFunc', 'IS_MUSL']
IS_MUSL = ...
_v = ...
if 'musl' in _v:
    IS_MUSL = ...
IS_EDITABLE = ...
class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    ...


class PytestTester:
    """
    Run tests for this namespace

    ``scipy.test()`` runs tests for all of SciPy, with the default settings.
    When used from a submodule (e.g., ``scipy.cluster.test()``, only the tests
    for that namespace are run.

    Parameters
    ----------
    label : {'fast', 'full'}, optional
        Whether to run only the fast tests, or also those marked as slow.
        Default is 'fast'.
    verbose : int, optional
        Test output verbosity. Default is 1.
    extra_argv : list, optional
        Arguments to pass through to Pytest.
    doctests : bool, optional
        Whether to run doctests or not. Default is False.
    coverage : bool, optional
        Whether to run tests with code coverage measurements enabled.
        Default is False.
    tests : list of str, optional
        List of module names to run tests for. By default, uses the module
        from which the ``test`` function is called.
    parallel : int, optional
        Run tests in parallel with pytest-xdist, if number given is larger than
        1. Default is 1.

    """
    def __init__(self, module_name) -> None:
        ...
    
    def __call__(self, label=..., verbose=..., extra_argv=..., doctests=..., coverage=..., tests=..., parallel=...): # -> bool:
        ...
    


class _TestPythranFunc:
    '''
    These are situations that can be tested in our pythran tests:
    - A function with multiple array arguments and then
      other positional and keyword arguments.
    - A function with array-like keywords (e.g. `def somefunc(x0, x1=None)`.
    Note: list/tuple input is not yet tested!

    `self.arguments`: A dictionary which key is the index of the argument,
                      value is tuple(array value, all supported dtypes)
    `self.partialfunc`: A function used to freeze some non-array argument
                        that of no interests in the original function
    '''
    ALL_INTEGER = ...
    ALL_FLOAT = ...
    ALL_COMPLEX = ...
    def setup_method(self): # -> None:
        ...
    
    def get_optional_args(self, func): # -> dict[Any, Any]:
        ...
    
    def get_max_dtype_list_length(self): # -> int:
        ...
    
    def get_dtype(self, dtype_list, dtype_idx):
        ...
    
    def test_all_dtypes(self): # -> None:
        ...
    
    def test_views(self): # -> None:
        ...
    
    def test_strided(self): # -> None:
        ...
    


def check_free_memory(free_mb): # -> None:
    """
    Check *free_mb* of memory is available, otherwise do pytest.skip
    """
    ...

