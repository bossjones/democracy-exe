"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader.api import CorpusReader

"""
CorpusReader for PanLex Lite, a stripped down version of PanLex distributed
as an SQLite database. See the README.txt in the panlex_lite corpus directory
for more information on PanLex Lite.
"""
class PanLexLiteCorpusReader(CorpusReader):
    MEANING_Q = ...
    TRANSLATION_Q = ...
    def __init__(self, root) -> None:
        ...
    
    def language_varieties(self, lc=...): # -> list[Any]:
        """
        Return a list of PanLex language varieties.

        :param lc: ISO 639 alpha-3 code. If specified, filters returned varieties
            by this code. If unspecified, all varieties are returned.
        :return: the specified language varieties as a list of tuples. The first
            element is the language variety's seven-character uniform identifier,
            and the second element is its default name.
        :rtype: list(tuple)
        """
        ...
    
    def meanings(self, expr_uid, expr_tt): # -> list[Meaning]:
        """
        Return a list of meanings for an expression.

        :param expr_uid: the expression's language variety, as a seven-character
            uniform identifier.
        :param expr_tt: the expression's text.
        :return: a list of Meaning objects.
        :rtype: list(Meaning)
        """
        ...
    
    def translations(self, from_uid, from_tt, to_uid): # -> list[Any]:
        """
        Return a list of translations for an expression into a single language
        variety.

        :param from_uid: the source expression's language variety, as a
            seven-character uniform identifier.
        :param from_tt: the source expression's text.
        :param to_uid: the target language variety, as a seven-character
            uniform identifier.
        :return: a list of translation tuples. The first element is the expression
            text and the second element is the translation quality.
        :rtype: list(tuple)
        """
        ...
    


class Meaning(dict):
    """
    Represents a single PanLex meaning. A meaning is a translation set derived
    from a single source.
    """
    def __init__(self, mn, attr) -> None:
        ...
    
    def id(self):
        """
        :return: the meaning's id.
        :rtype: int
        """
        ...
    
    def quality(self):
        """
        :return: the meaning's source's quality (0=worst, 9=best).
        :rtype: int
        """
        ...
    
    def source(self):
        """
        :return: the meaning's source id.
        :rtype: int
        """
        ...
    
    def source_group(self):
        """
        :return: the meaning's source group id.
        :rtype: int
        """
        ...
    
    def expressions(self):
        """
        :return: the meaning's expressions as a dictionary whose keys are language
            variety uniform identifiers and whose values are lists of expression
            texts.
        :rtype: dict
        """
        ...
    


