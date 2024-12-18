"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView

"""
Corpus reader for the SemCor Corpus.
"""
__docformat__ = ...
class SemcorCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the SemCor Corpus.
    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.
    """
    def __init__(self, root, fileids, wordnet, lazy=...) -> None:
        ...
    
    def words(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        ...
    
    def chunks(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of chunks,
            each of which is a list of words and punctuation symbols
            that form a unit.
        :rtype: list(list(str))
        """
        ...
    
    def tagged_chunks(self, fileids=..., tag=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of tagged chunks, represented
            in tree form.
        :rtype: list(Tree)

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        ...
    
    def sents(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of word strings.
        :rtype: list(list(str))
        """
        ...
    
    def chunk_sents(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of chunks.
        :rtype: list(list(list(str)))
        """
        ...
    
    def tagged_sents(self, fileids=..., tag=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list[Any] | tuple[()] | Element:
        """
        :return: the given file(s) as a list of sentences. Each sentence
            is represented as a list of tagged chunks (in tree form).
        :rtype: list(list(Tree))

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        ...
    


class SemcorSentence(list):
    """
    A list of words, augmented by an attribute ``num`` used to record
    the sentence identifier (the ``n`` attribute from the XML).
    """
    def __init__(self, num, items) -> None:
        ...
    


class SemcorWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """
    def __init__(self, fileid, unit, bracket_sent, pos_tag, sem_tag, wordnet) -> None:
        """
        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma
            and OOV named entity status.
        """
        ...
    
    def handle_elt(self, elt, context): # -> SemcorSentence | tuple[Any | Literal[''], ...] | list[LiteralString] | Tree | LiteralString | list[Tree] | Literal['']:
        ...
    
    def handle_word(self, elt): # -> tuple[Any | Literal[''], ...] | list[LiteralString] | Tree | LiteralString | list[Tree] | Literal['']:
        ...
    
    def handle_sent(self, elt): # -> SemcorSentence:
        ...
    


