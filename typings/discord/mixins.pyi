"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
__all__ = ('EqualityComparable', 'Hashable')
class EqualityComparable:
    __slots__ = ...
    id: int
    def __eq__(self, other: object) -> bool:
        ...
    


class Hashable(EqualityComparable):
    __slots__ = ...
    def __hash__(self) -> int:
        ...
    


