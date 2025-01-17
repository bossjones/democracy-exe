"""
This type stub file was generated by pyright.
"""

from nltk.internals import deprecated
from nltk.parse import ParserI

class ChunkParserI(ParserI):
    """
    A processing interface for identifying non-overlapping groups in
    unrestricted text.  Typically, chunk parsers are used to find base
    syntactic constituents, such as base noun phrases.  Unlike
    ``ParserI``, ``ChunkParserI`` guarantees that the ``parse()`` method
    will always generate a parse.
    """
    def parse(self, tokens):
        """
        Return the best chunk structure for the given tokens
        and return a tree.

        :param tokens: The list of (word, tag) tokens to be chunked.
        :type tokens: list(tuple)
        :rtype: Tree
        """
        ...
    
    @deprecated("Use accuracy(gold) instead.")
    def evaluate(self, gold): # -> ChunkScore:
        ...
    
    def accuracy(self, gold): # -> ChunkScore:
        """
        Score the accuracy of the chunker against the gold standard.
        Remove the chunking the gold standard text, rechunk it using
        the chunker, and return a ``ChunkScore`` object
        reflecting the performance of this chunk parser.

        :type gold: list(Tree)
        :param gold: The list of chunked sentences to score the chunker on.
        :rtype: ChunkScore
        """
        ...
    


