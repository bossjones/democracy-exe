"""
This type stub file was generated by pyright.
"""

"""
Code for extracting relational triples from the ieer and conll2002 corpora.

Relations are stored internally as dictionaries ('reldicts').

The two serialization outputs are "rtuple" and "clause".

- An rtuple is a tuple of the form ``(subj, filler, obj)``,
  where ``subj`` and ``obj`` are pairs of Named Entity mentions, and ``filler`` is the string of words
  occurring between ``sub`` and ``obj`` (with no intervening NEs). Strings are printed via ``repr()`` to
  circumvent locale variations in rendering utf-8 encoded strings.
- A clause is an atom of the form ``relsym(subjsym, objsym)``,
  where the relation, subject and object have been canonicalized to single strings.
"""
NE_CLASSES = ...
short2long = ...
long2short = ...
def class_abbrev(type): # -> str:
    """
    Abbreviate an NE class name.
    :type type: str
    :rtype: str
    """
    ...

def descape_entity(m, defs=...):
    """
    Translate one entity to its ISO Latin value.
    Inspired by example from effbot.org


    """
    ...

def list2sym(lst): # -> str:
    """
    Convert a list of strings into a canonical symbol.
    :type lst: list
    :return: a Unicode string without whitespace
    :rtype: unicode
    """
    ...

def tree2semi_rel(tree): # -> list[Any]:
    """
    Group a chunk structure into a list of 'semi-relations' of the form (list(str), ``Tree``).

    In order to facilitate the construction of (``Tree``, string, ``Tree``) triples, this
    identifies pairs whose first member is a list (possibly empty) of terminal
    strings, and whose second member is a ``Tree`` of the form (NE_label, terminals).

    :param tree: a chunk tree
    :return: a list of pairs (list(str), ``Tree``)
    :rtype: list of tuple
    """
    ...

def semi_rel2reldict(pairs, window=..., trace=...): # -> list[Any]:
    """
    Converts the pairs generated by ``tree2semi_rel`` into a 'reldict': a dictionary which
    stores information about the subject and object NEs plus the filler between them.
    Additionally, a left and right context of length =< window are captured (within
    a given input sentence).

    :param pairs: a pair of list(str) and ``Tree``, as generated by
    :param window: a threshold for the number of items to include in the left and right context
    :type window: int
    :return: 'relation' dictionaries whose keys are 'lcon', 'subjclass', 'subjtext', 'subjsym', 'filler', objclass', objtext', 'objsym' and 'rcon'
    :rtype: list(defaultdict)
    """
    ...

def extract_rels(subjclass, objclass, doc, corpus=..., pattern=..., window=...): # -> list[Any]:
    """
    Filter the output of ``semi_rel2reldict`` according to specified NE classes and a filler pattern.

    The parameters ``subjclass`` and ``objclass`` can be used to restrict the
    Named Entities to particular types (any of 'LOCATION', 'ORGANIZATION',
    'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE').

    :param subjclass: the class of the subject Named Entity.
    :type subjclass: str
    :param objclass: the class of the object Named Entity.
    :type objclass: str
    :param doc: input document
    :type doc: ieer document or a list of chunk trees
    :param corpus: name of the corpus to take as input; possible values are
        'ieer' and 'conll2002'
    :type corpus: str
    :param pattern: a regular expression for filtering the fillers of
        retrieved triples.
    :type pattern: SRE_Pattern
    :param window: filters out fillers which exceed this threshold
    :type window: int
    :return: see ``mk_reldicts``
    :rtype: list(defaultdict)
    """
    ...

def rtuple(reldict, lcon=..., rcon=...): # -> LiteralString:
    """
    Pretty print the reldict as an rtuple.
    :param reldict: a relation dictionary
    :type reldict: defaultdict
    """
    ...

def clause(reldict, relsym): # -> LiteralString:
    """
    Print the relation in clausal form.
    :param reldict: a relation dictionary
    :type reldict: defaultdict
    :param relsym: a label for the relation
    :type relsym: str
    """
    ...

def in_demo(trace=..., sql=...): # -> None:
    """
    Select pairs of organizations and locations whose mentions occur with an
    intervening occurrence of the preposition "in".

    If the sql parameter is set to True, then the entity pairs are loaded into
    an in-memory database, and subsequently pulled out using an SQL "SELECT"
    query.
    """
    ...

def roles_demo(trace=...): # -> None:
    ...

def ieer_headlines(): # -> None:
    ...

def conllned(trace=...): # -> None:
    """
    Find the copula+'van' relation ('of') in the Dutch tagged training corpus
    from CoNLL 2002.
    """
    ...

def conllesp(): # -> None:
    ...

def ne_chunked(): # -> None:
    ...

if __name__ == "__main__":
    ...
