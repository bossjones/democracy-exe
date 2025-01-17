"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *

"""
Read CoNLL-style chunk fileids.
"""
class ConllCorpusReader(CorpusReader):
    """
    A corpus reader for CoNLL-style files.  These files consist of a
    series of sentences, separated by blank lines.  Each sentence is
    encoded using a table (or "grid") of values, where each line
    corresponds to a single word, and each column corresponds to an
    annotation type.  The set of columns used by CoNLL-style files can
    vary from corpus to corpus; the ``ConllCorpusReader`` constructor
    therefore takes an argument, ``columntypes``, which is used to
    specify the columns that are used by a given corpus. By default
    columns are split by consecutive whitespaces, with the
    ``separator`` argument you can set a string to split by (e.g.
    ``\'\t\'``).


    @todo: Add support for reading from corpora where different
        parallel files contain different columns.
    @todo: Possibly add caching of the grid corpus view?  This would
        allow the same grid view to be used by different data access
        methods (eg words() and parsed_sents() could both share the
        same grid corpus view object).
    @todo: Better support for -DOCSTART-.  Currently, we just ignore
        it, but it could be used to define methods that retrieve a
        document at a time (eg parsed_documents()).
    """
    WORDS = ...
    POS = ...
    TREE = ...
    CHUNK = ...
    NE = ...
    SRL = ...
    IGNORE = ...
    COLUMN_TYPES = ...
    def __init__(self, root, fileids, columntypes, chunk_types=..., root_label=..., pos_in_tree=..., srl_includes_roleset=..., encoding=..., tree_class=..., tagset=..., separator=...) -> None:
        ...
    
    def words(self, fileids=...): # -> LazyConcatenation:
        ...
    
    def sents(self, fileids=...): # -> LazyMap:
        ...
    
    def tagged_words(self, fileids=..., tagset=...): # -> LazyConcatenation:
        ...
    
    def tagged_sents(self, fileids=..., tagset=...): # -> LazyMap:
        ...
    
    def chunked_words(self, fileids=..., chunk_types=..., tagset=...): # -> LazyConcatenation:
        ...
    
    def chunked_sents(self, fileids=..., chunk_types=..., tagset=...): # -> LazyMap:
        ...
    
    def parsed_sents(self, fileids=..., pos_in_tree=..., tagset=...): # -> LazyMap:
        ...
    
    def srl_spans(self, fileids=...): # -> LazyMap:
        ...
    
    def srl_instances(self, fileids=..., pos_in_tree=..., flatten=...): # -> LazyConcatenation | LazyMap:
        ...
    
    def iob_words(self, fileids=..., tagset=...): # -> LazyConcatenation:
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        ...
    
    def iob_sents(self, fileids=..., tagset=...): # -> LazyMap:
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        ...
    


class ConllSRLInstance:
    """
    An SRL instance from a CoNLL corpus, which identifies and
    providing labels for the arguments of a single verb.
    """
    def __init__(self, tree, verb_head, verb_stem, roleset, tagged_spans) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def pprint(self): # -> str:
        ...
    


class ConllSRLInstanceList(list):
    """
    Set of instances for a single sentence
    """
    def __init__(self, tree, instances=...) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def pprint(self, include_tree=...): # -> LiteralString | str:
        ...
    


class ConllChunkCorpusReader(ConllCorpusReader):
    """
    A ConllCorpusReader whose data file contains three columns: words,
    pos, and chunk.
    """
    def __init__(self, root, fileids, chunk_types, encoding=..., tagset=..., separator=...) -> None:
        ...
    


