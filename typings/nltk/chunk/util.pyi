"""
This type stub file was generated by pyright.
"""

def accuracy(chunker, gold): # -> float:
    """
    Score the accuracy of the chunker against the gold standard.
    Strip the chunk information from the gold standard and rechunk it using
    the chunker, then compute the accuracy score.

    :type chunker: ChunkParserI
    :param chunker: The chunker being evaluated.
    :type gold: tree
    :param gold: The chunk structures to score the chunker on.
    :rtype: float
    """
    ...

class ChunkScore:
    """
    A utility class for scoring chunk parsers.  ``ChunkScore`` can
    evaluate a chunk parser's output, based on a number of statistics
    (precision, recall, f-measure, misssed chunks, incorrect chunks).
    It can also combine the scores from the parsing of multiple texts;
    this makes it significantly easier to evaluate a chunk parser that
    operates one sentence at a time.

    Texts are evaluated with the ``score`` method.  The results of
    evaluation can be accessed via a number of accessor methods, such
    as ``precision`` and ``f_measure``.  A typical use of the
    ``ChunkScore`` class is::

        >>> chunkscore = ChunkScore()           # doctest: +SKIP
        >>> for correct in correct_sentences:   # doctest: +SKIP
        ...     guess = chunkparser.parse(correct.leaves())   # doctest: +SKIP
        ...     chunkscore.score(correct, guess)              # doctest: +SKIP
        >>> print('F Measure:', chunkscore.f_measure())       # doctest: +SKIP
        F Measure: 0.823

    :ivar kwargs: Keyword arguments:

        - max_tp_examples: The maximum number actual examples of true
          positives to record.  This affects the ``correct`` member
          function: ``correct`` will not return more than this number
          of true positive examples.  This does *not* affect any of
          the numerical metrics (precision, recall, or f-measure)

        - max_fp_examples: The maximum number actual examples of false
          positives to record.  This affects the ``incorrect`` member
          function and the ``guessed`` member function: ``incorrect``
          will not return more than this number of examples, and
          ``guessed`` will not return more than this number of true
          positive examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - max_fn_examples: The maximum number actual examples of false
          negatives to record.  This affects the ``missed`` member
          function and the ``correct`` member function: ``missed``
          will not return more than this number of examples, and
          ``correct`` will not return more than this number of true
          negative examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - chunk_label: A regular expression indicating which chunks
          should be compared.  Defaults to ``'.*'`` (i.e., all chunks).

    :type _tp: list(Token)
    :ivar _tp: List of true positives
    :type _fp: list(Token)
    :ivar _fp: List of false positives
    :type _fn: list(Token)
    :ivar _fn: List of false negatives

    :type _tp_num: int
    :ivar _tp_num: Number of true positives
    :type _fp_num: int
    :ivar _fp_num: Number of false positives
    :type _fn_num: int
    :ivar _fn_num: Number of false negatives.
    """
    def __init__(self, **kwargs) -> None:
        ...
    
    def score(self, correct, guessed): # -> None:
        """
        Given a correctly chunked sentence, score another chunked
        version of the same sentence.

        :type correct: chunk structure
        :param correct: The known-correct ("gold standard") chunked
            sentence.
        :type guessed: chunk structure
        :param guessed: The chunked sentence to be scored.
        """
        ...
    
    def accuracy(self): # -> float | Literal[1]:
        """
        Return the overall tag-based accuracy for all text that have
        been scored by this ``ChunkScore``, using the IOB (conll2000)
        tag encoding.

        :rtype: float
        """
        ...
    
    def precision(self): # -> float | Literal[0]:
        """
        Return the overall precision for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        ...
    
    def recall(self): # -> float | Literal[0]:
        """
        Return the overall recall for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        ...
    
    def f_measure(self, alpha=...): # -> float | Literal[0]:
        """
        Return the overall F measure for all texts that have been
        scored by this ``ChunkScore``.

        :param alpha: the relative weighting of precision and recall.
            Larger alpha biases the score towards the precision value,
            while smaller alpha biases the score towards the recall
            value.  ``alpha`` should have a value in the range [0,1].
        :type alpha: float
        :rtype: float
        """
        ...
    
    def missed(self): # -> list[Any]:
        """
        Return the chunks which were included in the
        correct chunk structures, but not in the guessed chunk
        structures, listed in input order.

        :rtype: list of chunks
        """
        ...
    
    def incorrect(self): # -> list[Any]:
        """
        Return the chunks which were included in the guessed chunk structures,
        but not in the correct chunk structures, listed in input order.

        :rtype: list of chunks
        """
        ...
    
    def correct(self): # -> list[Any]:
        """
        Return the chunks which were included in the correct
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        ...
    
    def guessed(self): # -> list[Any]:
        """
        Return the chunks which were included in the guessed
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __repr__(self): # -> str:
        """
        Return a concise representation of this ``ChunkScoring``.

        :rtype: str
        """
        ...
    
    def __str__(self) -> str:
        """
        Return a verbose representation of this ``ChunkScoring``.
        This representation includes the precision, recall, and
        f-measure scores.  For other information about the score,
        use the accessor methods (e.g., ``missed()`` and ``incorrect()``).

        :rtype: str
        """
        ...
    


def tagstr2tree(s, chunk_label=..., root_label=..., sep=..., source_tagset=..., target_tagset=...): # -> Tree:
    """
    Divide a string of bracketted tagged text into
    chunks and unchunked tokens, and produce a Tree.
    Chunks are marked by square brackets (``[...]``).  Words are
    delimited by whitespace, and each word should have the form
    ``text/tag``.  Words that do not contain a slash are
    assigned a ``tag`` of None.

    :param s: The string to be converted
    :type s: str
    :param chunk_label: The label to use for chunk nodes
    :type chunk_label: str
    :param root_label: The label to use for the root of the tree
    :type root_label: str
    :rtype: Tree
    """
    ...

_LINE_RE = ...
def conllstr2tree(s, chunk_types=..., root_label=...): # -> Tree:
    """
    Return a chunk structure for a single sentence
    encoded in the given CONLL 2000 style string.
    This function converts a CoNLL IOB string into a tree.
    It uses the specified chunk types
    (defaults to NP, PP and VP), and creates a tree rooted at a node
    labeled S (by default).

    :param s: The CoNLL string to be converted.
    :type s: str
    :param chunk_types: The chunk types to be converted.
    :type chunk_types: tuple
    :param root_label: The node label to use for the root.
    :type root_label: str
    :rtype: Tree
    """
    ...

def tree2conlltags(t): # -> list[Any]:
    """
    Return a list of 3-tuples containing ``(word, tag, IOB-tag)``.
    Convert a tree to the CoNLL IOB tag format.

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: list(tuple)
    """
    ...

def conlltags2tree(sentence, chunk_types=..., root_label=..., strict=...): # -> Tree:
    """
    Convert the CoNLL IOB format to a tree.
    """
    ...

def tree2conllstr(t): # -> str:
    """
    Return a multiline string where each line contains a word, tag and IOB tag.
    Convert a tree to the CoNLL IOB string format

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: str
    """
    ...

_IEER_DOC_RE = ...
_IEER_TYPE_RE = ...
def ieerstr2tree(s, chunk_types=..., root_label=...): # -> dict[str, Any] | list[Any] | Tree:
    """
    Return a chunk structure containing the chunked tagged text that is
    encoded in the given IEER style string.
    Convert a string of chunked tagged text in the IEER named
    entity format into a chunk structure.  Chunks are of several
    types, LOCATION, ORGANIZATION, PERSON, DURATION, DATE, CARDINAL,
    PERCENT, MONEY, and MEASURE.

    :rtype: Tree
    """
    ...

def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
