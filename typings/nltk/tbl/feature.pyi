"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod

class Feature(metaclass=ABCMeta):
    """
    An abstract base class for Features. A Feature is a combination of
    a specific property-computing method and a list of relative positions
    to apply that method to.

    The property-computing method, M{extract_property(tokens, index)},
    must be implemented by every subclass. It extracts or computes a specific
    property for the token at the current index. Typical extract_property()
    methods return features such as the token text or tag; but more involved
    methods may consider the entire sequence M{tokens} and
    for instance compute the length of the sentence the token belongs to.

    In addition, the subclass may have a PROPERTY_NAME, which is how
    it will be printed (in Rules and Templates, etc). If not given, defaults
    to the classname.

    """
    json_tag = ...
    PROPERTY_NAME = ...
    def __init__(self, positions, end=...) -> None:
        """
        Construct a Feature which may apply at C{positions}.

        >>> # For instance, importing some concrete subclasses (Feature is abstract)
        >>> from nltk.tag.brill import Word, Pos

        >>> # Feature Word, applying at one of [-2, -1]
        >>> Word([-2,-1])
        Word([-2, -1])

        >>> # Positions need not be contiguous
        >>> Word([-2,-1, 1])
        Word([-2, -1, 1])

        >>> # Contiguous ranges can alternatively be specified giving the
        >>> # two endpoints (inclusive)
        >>> Pos(-3, -1)
        Pos([-3, -2, -1])

        >>> # In two-arg form, start <= end is enforced
        >>> Pos(2, 1)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "nltk/tbl/template.py", line 306, in __init__
            raise TypeError
        ValueError: illegal interval specification: (start=2, end=1)

        :type positions: list of int
        :param positions: the positions at which this features should apply
        :raises ValueError: illegal position specifications

        An alternative calling convention, for contiguous positions only,
        is Feature(start, end):

        :type start: int
        :param start: start of range where this feature should apply
        :type end: int
        :param end: end of range (NOTE: inclusive!) where this feature should apply
        """
        ...

    def encode_json_obj(self): # -> tuple[int, ...] | None:
        ...

    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...

    def __repr__(self): # -> str:
        ...

    @classmethod
    def expand(cls, starts, winlens, excludezero=...): # -> list[Self]:
        """
        Return a list of features, one for each start point in starts
        and for each window length in winlen. If excludezero is True,
        no Features containing 0 in its positions will be generated
        (many tbl trainers have a special representation for the
        target feature at [0])

        For instance, importing a concrete subclass (Feature is abstract)

        >>> from nltk.tag.brill import Word

        First argument gives the possible start positions, second the
        possible window lengths

        >>> Word.expand([-3,-2,-1], [1])
        [Word([-3]), Word([-2]), Word([-1])]

        >>> Word.expand([-2,-1], [1])
        [Word([-2]), Word([-1])]

        >>> Word.expand([-3,-2,-1], [1,2])
        [Word([-3]), Word([-2]), Word([-1]), Word([-3, -2]), Word([-2, -1])]

        >>> Word.expand([-2,-1], [1])
        [Word([-2]), Word([-1])]

        A third optional argument excludes all Features whose positions contain zero

        >>> Word.expand([-2,-1,0], [1,2], excludezero=False)
        [Word([-2]), Word([-1]), Word([0]), Word([-2, -1]), Word([-1, 0])]

        >>> Word.expand([-2,-1,0], [1,2], excludezero=True)
        [Word([-2]), Word([-1]), Word([-2, -1])]

        All window lengths must be positive

        >>> Word.expand([-2,-1], [0])
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "nltk/tag/tbl/template.py", line 371, in expand
            :param starts: where to start looking for Feature
        ValueError: non-positive window length in [0]

        :param starts: where to start looking for Feature
        :type starts: list of ints
        :param winlens: window lengths where to look for Feature
        :type starts: list of ints
        :param excludezero: do not output any Feature with 0 in any of its positions.
        :type excludezero: bool
        :returns: list of Features
        :raises ValueError: for non-positive window lengths
        """
        ...

    def issuperset(self, other): # -> bool:
        """
        Return True if this Feature always returns True when other does

        More precisely, return True if this feature refers to the same property as other;
        and this Feature looks at all positions that other does (and possibly
        other positions in addition).

        #For instance, importing a concrete subclass (Feature is abstract)
        >>> from nltk.tag.brill import Word, Pos

        >>> Word([-3,-2,-1]).issuperset(Word([-3,-2]))
        True

        >>> Word([-3,-2,-1]).issuperset(Word([-3,-2, 0]))
        False

        #Feature subclasses must agree
        >>> Word([-3,-2,-1]).issuperset(Pos([-3,-2]))
        False

        :param other: feature with which to compare
        :type other: (subclass of) Feature
        :return: True if this feature is superset, otherwise False
        :rtype: bool


        """
        ...

    def intersects(self, other): # -> bool:
        """
        Return True if the positions of this Feature intersects with those of other

        More precisely, return True if this feature refers to the same property as other;
        and there is some overlap in the positions they look at.

        #For instance, importing a concrete subclass (Feature is abstract)
        >>> from nltk.tag.brill import Word, Pos

        >>> Word([-3,-2,-1]).intersects(Word([-3,-2]))
        True

        >>> Word([-3,-2,-1]).intersects(Word([-3,-2, 0]))
        True

        >>> Word([-3,-2,-1]).intersects(Word([0]))
        False

        #Feature subclasses must agree
        >>> Word([-3,-2,-1]).intersects(Pos([-3,-2]))
        False

        :param other: feature with which to compare
        :type other: (subclass of) Feature
        :return: True if feature classes agree and there is some overlap in the positions they look at
        :rtype: bool
        """
        ...

    def __eq__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    def __gt__(self, other) -> bool:
        ...

    def __ge__(self, other) -> bool:
        ...

    def __le__(self, other) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def extract_property(tokens, index): # -> None:
        """
        Any subclass of Feature must define static method extract_property(tokens, index)

        :param tokens: the sequence of tokens
        :type tokens: list of tokens
        :param index: the current index
        :type index: int
        :return: feature value
        :rtype: any (but usually scalar)
        """
        ...
