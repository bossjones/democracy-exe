"""
This type stub file was generated by pyright.
"""

from nltk.parse.api import ParserI

class ViterbiParser(ParserI):
    """
    A bottom-up ``PCFG`` parser that uses dynamic programming to find
    the single most likely parse for a text.  The ``ViterbiParser`` parser
    parses texts by filling in a "most likely constituent table".
    This table records the most probable tree representation for any
    given span and node value.  In particular, it has an entry for
    every start index, end index, and node value, recording the most
    likely subtree that spans from the start index to the end index,
    and has the given node value.

    The ``ViterbiParser`` parser fills in this table incrementally.  It starts
    by filling in all entries for constituents that span one element
    of text (i.e., entries where the end index is one greater than the
    start index).  After it has filled in all table entries for
    constituents that span one element of text, it fills in the
    entries for constitutants that span two elements of text.  It
    continues filling in the entries for constituents spanning larger
    and larger portions of the text, until the entire table has been
    filled.  Finally, it returns the table entry for a constituent
    spanning the entire text, whose node value is the grammar's start
    symbol.

    In order to find the most likely constituent with a given span and
    node value, the ``ViterbiParser`` parser considers all productions that
    could produce that node value.  For each production, it finds all
    children that collectively cover the span and have the node values
    specified by the production's right hand side.  If the probability
    of the tree formed by applying the production to the children is
    greater than the probability of the current entry in the table,
    then the table is updated with this new tree.

    A pseudo-code description of the algorithm used by
    ``ViterbiParser`` is:

    | Create an empty most likely constituent table, *MLC*.
    | For width in 1...len(text):
    |   For start in 1...len(text)-width:
    |     For prod in grammar.productions:
    |       For each sequence of subtrees [t[1], t[2], ..., t[n]] in MLC,
    |         where t[i].label()==prod.rhs[i],
    |         and the sequence covers [start:start+width]:
    |           old_p = MLC[start, start+width, prod.lhs]
    |           new_p = P(t[1])P(t[1])...P(t[n])P(prod)
    |           if new_p > old_p:
    |             new_tree = Tree(prod.lhs, t[1], t[2], ..., t[n])
    |             MLC[start, start+width, prod.lhs] = new_tree
    | Return MLC[0, len(text), start_symbol]

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """
    def __init__(self, grammar, trace=...) -> None:
        """
        Create a new ``ViterbiParser`` parser, that uses ``grammar`` to
        parse texts.

        :type grammar: PCFG
        :param grammar: The grammar used to parse texts.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        """
        ...
    
    def grammar(self): # -> Any:
        ...
    
    def trace(self, trace=...): # -> None:
        """
        Set the level of tracing output that should be generated when
        parsing a text.

        :type trace: int
        :param trace: The trace level.  A trace level of ``0`` will
            generate no tracing output; and higher trace levels will
            produce more verbose tracing output.
        :rtype: None
        """
        ...
    
    def parse(self, tokens): # -> Generator[Any, Any, None]:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    


def demo(): # -> None:
    """
    A demonstration of the probabilistic parsers.  The user is
    prompted to select which demo to run, and how many parses should
    be found; and then each parser is run on the same demo, and a
    summary of the results are displayed.
    """
    ...

if __name__ == "__main__":
    ...
