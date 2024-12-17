"""
This type stub file was generated by pyright.
"""

from functools import total_ordering
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *

class PropbankCorpusReader(CorpusReader):
    """
    Corpus reader for the propbank corpus, which augments the Penn
    Treebank with information about the predicate argument structure
    of every verb instance.  The corpus consists of two parts: the
    predicate-argument annotations themselves, and a set of "frameset
    files" which define the argument labels used by the annotations,
    on a per-verb basis.  Each "frameset file" contains one or more
    predicates, such as ``'turn'`` or ``'turn_on'``, each of which is
    divided into coarse-grained word senses called "rolesets".  For
    each "roleset", the frameset file provides descriptions of the
    argument roles, along with examples.
    """
    def __init__(self, root, propfile, framefiles=..., verbsfile=..., parse_fileid_xform=..., parse_corpus=..., encoding=...) -> None:
        """
        :param root: The root directory for this corpus.
        :param propfile: The name of the file containing the predicate-
            argument annotations (relative to ``root``).
        :param framefiles: A list or regexp specifying the frameset
            fileids for this corpus.
        :param parse_fileid_xform: A transform that should be applied
            to the fileids in this corpus.  This should be a function
            of one argument (a fileid) that returns a string (the new
            fileid).
        :param parse_corpus: The corpus containing the parse trees
            corresponding to this corpus.  These parse trees are
            necessary to resolve the tree pointers used by propbank.
        """
        ...
    
    def instances(self, baseform=...): # -> StreamBackedCorpusView:
        """
        :return: a corpus view that acts as a list of
            ``PropBankInstance`` objects, one for each noun in the corpus.
        """
        ...
    
    def lines(self): # -> StreamBackedCorpusView:
        """
        :return: a corpus view that acts as a list of strings, one for
            each line in the predicate-argument annotation file.
        """
        ...
    
    def roleset(self, roleset_id): # -> Element | Any:
        """
        :return: the xml description for the given roleset.
        """
        ...
    
    def rolesets(self, baseform=...): # -> LazyConcatenation:
        """
        :return: list of xml descriptions for rolesets.
        """
        ...
    
    def verbs(self): # -> StreamBackedCorpusView:
        """
        :return: a corpus view that acts as a list of all verb lemmas
            in this corpus (from the verbs.txt file).
        """
        ...
    


class PropbankInstance:
    def __init__(self, fileid, sentnum, wordnum, tagger, roleset, inflection, predicate, arguments, parse_corpus=...) -> None:
        ...
    
    @property
    def baseform(self):
        """The baseform of the predicate."""
        ...
    
    @property
    def sensenumber(self):
        """The sense number of the predicate."""
        ...
    
    @property
    def predid(self): # -> Literal['rel']:
        """Identifier of the predicate."""
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def __str__(self) -> str:
        ...
    
    tree = ...
    @staticmethod
    def parse(s, parse_fileid_xform=..., parse_corpus=...): # -> PropbankInstance:
        ...
    


class PropbankPointer:
    """
    A pointer used by propbank to identify one or more constituents in
    a parse tree.  ``PropbankPointer`` is an abstract base class with
    three concrete subclasses:

      - ``PropbankTreePointer`` is used to point to single constituents.
      - ``PropbankSplitTreePointer`` is used to point to 'split'
        constituents, which consist of a sequence of two or more
        ``PropbankTreePointer`` pointers.
      - ``PropbankChainTreePointer`` is used to point to entire trace
        chains in a tree.  It consists of a sequence of pieces, which
        can be ``PropbankTreePointer`` or ``PropbankSplitTreePointer`` pointers.
    """
    def __init__(self) -> None:
        ...
    


class PropbankChainTreePointer(PropbankPointer):
    def __init__(self, pieces) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def select(self, tree): # -> Tree:
        ...
    


class PropbankSplitTreePointer(PropbankPointer):
    def __init__(self, pieces) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def select(self, tree): # -> Tree:
        ...
    


@total_ordering
class PropbankTreePointer(PropbankPointer):
    """
    wordnum:height*wordnum:height*...
    wordnum:height,

    """
    def __init__(self, wordnum, height) -> None:
        ...
    
    @staticmethod
    def parse(s): # -> PropbankChainTreePointer | PropbankSplitTreePointer | PropbankTreePointer:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __ne__(self, other) -> bool:
        ...
    
    def __lt__(self, other) -> bool:
        ...
    
    def select(self, tree):
        ...
    
    def treepos(self, tree): # -> tuple[Any, ...]:
        """
        Convert this pointer to a standard 'tree position' pointer,
        given that it points to the given tree.
        """
        ...
    


class PropbankInflection:
    INFINITIVE = ...
    GERUND = ...
    PARTICIPLE = ...
    FINITE = ...
    FUTURE = ...
    PAST = ...
    PRESENT = ...
    PERFECT = ...
    PROGRESSIVE = ...
    PERFECT_AND_PROGRESSIVE = ...
    THIRD_PERSON = ...
    ACTIVE = ...
    PASSIVE = ...
    NONE = ...
    def __init__(self, form=..., tense=..., aspect=..., person=..., voice=...) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    _VALIDATE = ...
    @staticmethod
    def parse(s): # -> PropbankInflection:
        ...
    


