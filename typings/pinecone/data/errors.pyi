"""
This type stub file was generated by pyright.
"""

class VectorDictionaryMissingKeysError(ValueError):
    def __init__(self, item) -> None:
        ...



class VectorDictionaryExcessKeysError(ValueError):
    def __init__(self, item) -> None:
        ...



class VectorTupleLengthError(ValueError):
    def __init__(self, item) -> None:
        ...



class SparseValuesTypeError(ValueError, TypeError):
    def __init__(self) -> None:
        ...



class SparseValuesMissingKeysError(ValueError):
    def __init__(self, sparse_values_dict) -> None:
        ...



class SparseValuesDictionaryExpectedError(ValueError, TypeError):
    def __init__(self, sparse_values_dict) -> None:
        ...



class MetadataDictionaryExpectedError(ValueError, TypeError):
    def __init__(self, item) -> None:
        ...
