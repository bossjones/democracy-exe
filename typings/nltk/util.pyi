"""
This type stub file was generated by pyright.
"""

from collections import defaultdict

from nltk.collections import *
from nltk.internals import deprecated

@deprecated("Use help(obj) instead.")
def usage(obj): # -> None:
    ...

def in_idle(): # -> bool:
    """
    Return True if this function is run within idle.  Tkinter
    programs that are run in idle should never call ``Tk.mainloop``; so
    this function should be used to gate all calls to ``Tk.mainloop``.

    :warning: This function works by checking ``sys.stdin``.  If the
        user has modified ``sys.stdin``, then it may return incorrect
        results.
    :rtype: bool
    """
    ...

def pr(data, start=..., end=...): # -> None:
    """
    Pretty print a sequence of data items

    :param data: the data stream to print
    :type data: sequence or iter
    :param start: the start position
    :type start: int
    :param end: the end position
    :type end: int
    """
    ...

def print_string(s, width=...): # -> None:
    """
    Pretty print a string, breaking lines on whitespace

    :param s: the string to print, consisting of words and spaces
    :type s: str
    :param width: the display width
    :type width: int
    """
    ...

def tokenwrap(tokens, separator=..., width=...): # -> str:
    """
    Pretty print a list of text tokens, breaking lines on whitespace

    :param tokens: the tokens to print
    :type tokens: list
    :param separator: the string to use to separate tokens
    :type separator: str
    :param width: the display width (default=70)
    :type width: int
    """
    ...

def cut_string(s, width=...): # -> Literal['']:
    """
    Cut off and return a given width of a string

    Return the same as s[:width] if width >= 0 or s[-width:] if
    width < 0, as long as s has no unicode combining characters.
    If it has combining characters make sure the returned string's
    visible width matches the called-for width.

    :param s: the string to cut
    :type s: str
    :param width: the display_width
    :type width: int
    """
    ...

class Index(defaultdict):
    def __init__(self, pairs) -> None:
        ...



def re_show(regexp, string, left=..., right=...): # -> None:
    """
    Return a string with markers surrounding the matched substrings.
    Search str for substrings matching ``regexp`` and wrap the matches
    with braces.  This is convenient for learning about regular expressions.

    :param regexp: The regular expression.
    :type regexp: str
    :param string: The string being matched.
    :type string: str
    :param left: The left delimiter (printed before the matched substring)
    :type left: str
    :param right: The right delimiter (printed after the matched substring)
    :type right: str
    :rtype: str
    """
    ...

def filestring(f): # -> str:
    ...

def breadth_first(tree, children=..., maxdepth=...): # -> Generator[Any, Any, None]:
    """Traverse the nodes of a tree in breadth-first order.
    (No check for cycles.)
    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    ...

def edge_closure(tree, children=..., maxdepth=..., verbose=...): # -> Generator[tuple, Any, None]:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node
    :param maxdepth: to limit the search depth
    :param verbose: to print warnings when cycles are discarded

    Yield the edges of a graph in breadth-first order,
    discarding eventual cycles.
    The first argument should be the start node;
    children should be a function taking as argument a graph node
    and returning an iterator of the node's children.

    >>> from nltk.util import edge_closure
    >>> print(list(edge_closure('A', lambda node:{'A':['B','C'], 'B':'C', 'C':'B'}[node])))
    [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]
    """
    ...

def edges2dot(edges, shapes=..., attr=...): # -> str:
    """
    :param edges: the set (or list) of edges of a directed graph.
    :param shapes: dictionary of strings that trigger a specified shape.
    :param attr: dictionary with global graph attributes
    :return: a representation of 'edges' as a string in the DOT graph language.

    Returns dot_string: a representation of 'edges' as a string in the DOT
    graph language, which can be converted to an image by the 'dot' program
    from the Graphviz package, or nltk.parse.dependencygraph.dot2img(dot_string).

    >>> import nltk
    >>> from nltk.util import edges2dot
    >>> print(edges2dot([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]))
    digraph G {
    "A" -> "B";
    "A" -> "C";
    "B" -> "C";
    "C" -> "B";
    }
    <BLANKLINE>
    """
    ...

def unweighted_minimum_spanning_digraph(tree, children=..., shapes=..., attr=...): # -> str:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node
    :param shapes: dictionary of strings that trigger a specified shape.
    :param attr: dictionary with global graph attributes

        Build a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    Return a representation of this MST as a string in the DOT graph language,
    which can be converted to an image by the 'dot' program from the Graphviz
    package, or nltk.parse.dependencygraph.dot2img(dot_string).

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> wn=nltk.corpus.wordnet
    >>> from nltk.util import unweighted_minimum_spanning_digraph as umsd
    >>> print(umsd(wn.synset('bound.a.01'), lambda s:sorted(s.also_sees())))
    digraph G {
    "Synset('bound.a.01')" -> "Synset('unfree.a.02')";
    "Synset('unfree.a.02')" -> "Synset('confined.a.02')";
    "Synset('unfree.a.02')" -> "Synset('dependent.a.01')";
    "Synset('unfree.a.02')" -> "Synset('restricted.a.01')";
    "Synset('restricted.a.01')" -> "Synset('classified.a.02')";
    }
    <BLANKLINE>
    """
    ...

def acyclic_breadth_first(tree, children=..., maxdepth=..., verbose=...): # -> Generator[Any, Any, None]:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node
    :param maxdepth: to limit the search depth
    :param verbose: to print warnings when cycles are discarded
    :return: the tree in breadth-first order

        Adapted from breadth_first() above, to discard cycles.
    Traverse the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    ...

def acyclic_depth_first(tree, children=..., depth=..., cut_mark=..., traversed=..., verbose=...): # -> list:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node
    :param depth: the maximum depth of the search
    :param cut_mark: the mark to add when cycles are truncated
    :param traversed: the set of traversed nodes
    :param verbose: to print warnings when cycles are discarded
    :return: the tree in depth-first order

    Traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within any branch,
    adding cut_mark (when specified) if cycles were truncated.
    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches all cycles:

    >>> import nltk
    >>> from nltk.util import acyclic_depth_first as acyclic_tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(acyclic_tree(wn.synset('dog.n.01'), lambda s:sorted(s.hypernyms()),cut_mark='...'))
    [Synset('dog.n.01'),
     [Synset('canine.n.02'),
      [Synset('carnivore.n.01'),
       [Synset('placental.n.01'),
        [Synset('mammal.n.01'),
         [Synset('vertebrate.n.01'),
          [Synset('chordate.n.01'),
           [Synset('animal.n.01'),
            [Synset('organism.n.01'),
             [Synset('living_thing.n.01'),
              [Synset('whole.n.02'),
               [Synset('object.n.01'),
                [Synset('physical_entity.n.01'),
                 [Synset('entity.n.01')]]]]]]]]]]]]],
     [Synset('domestic_animal.n.01'), "Cycle(Synset('animal.n.01'),-3,...)"]]
    """
    ...

def acyclic_branches_depth_first(tree, children=..., depth=..., cut_mark=..., traversed=..., verbose=...): # -> list:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node
    :param depth: the maximum depth of the search
    :param cut_mark: the mark to add when cycles are truncated
    :param traversed: the set of traversed nodes
    :param verbose: to print warnings when cycles are discarded
    :return: the tree in depth-first order

        Adapted from acyclic_depth_first() above, to
    traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within the same branch,
    but keep duplicate paths in different branches.
    Add cut_mark (when defined) if cycles were truncated.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches only only cycles within the same branch,
    but keeping cycles from different branches:

    >>> import nltk
    >>> from nltk.util import acyclic_branches_depth_first as tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(tree(wn.synset('certified.a.01'), lambda s:sorted(s.also_sees()), cut_mark='...', depth=4))
    [Synset('certified.a.01'),
     [Synset('authorized.a.01'),
      [Synset('lawful.a.01'),
       [Synset('legal.a.01'),
        "Cycle(Synset('lawful.a.01'),0,...)",
        [Synset('legitimate.a.01'), '...']],
       [Synset('straight.a.06'),
        [Synset('honest.a.01'), '...'],
        "Cycle(Synset('lawful.a.01'),0,...)"]],
      [Synset('legitimate.a.01'),
       "Cycle(Synset('authorized.a.01'),1,...)",
       [Synset('legal.a.01'),
        [Synset('lawful.a.01'), '...'],
        "Cycle(Synset('legitimate.a.01'),0,...)"],
       [Synset('valid.a.01'),
        "Cycle(Synset('legitimate.a.01'),0,...)",
        [Synset('reasonable.a.01'), '...']]],
      [Synset('official.a.01'), "Cycle(Synset('authorized.a.01'),1,...)"]],
     [Synset('documented.a.01')]]
    """
    ...

def acyclic_dic2tree(node, dic): # -> list:
    """
    :param node: the root node
    :param dic: the dictionary of children

    Convert acyclic dictionary 'dic', where the keys are nodes, and the
    values are lists of children, to output tree suitable for pprint(),
    starting at root 'node', with subtrees as nested lists."""
    ...

def unweighted_minimum_spanning_dict(tree, children=...): # -> dict:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node

            Output a dictionary representing a Minimum Spanning Tree (MST)
    of an unweighted graph, by traversing the nodes of a tree in
    breadth-first order, discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> from nltk.corpus import wordnet as wn
    >>> from nltk.util import unweighted_minimum_spanning_dict as umsd
    >>> from pprint import pprint
    >>> pprint(umsd(wn.synset('bound.a.01'), lambda s:sorted(s.also_sees())))
    {Synset('bound.a.01'): [Synset('unfree.a.02')],
     Synset('classified.a.02'): [],
     Synset('confined.a.02'): [],
     Synset('dependent.a.01'): [],
     Synset('restricted.a.01'): [Synset('classified.a.02')],
     Synset('unfree.a.02'): [Synset('confined.a.02'),
                             Synset('dependent.a.01'),
                             Synset('restricted.a.01')]}

    """
    ...

def unweighted_minimum_spanning_tree(tree, children=...): # -> list:
    """
    :param tree: the tree root
    :param children: a function taking as argument a tree node

       Output a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> from nltk.util import unweighted_minimum_spanning_tree as mst
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(mst(wn.synset('bound.a.01'), lambda s:sorted(s.also_sees())))
    [Synset('bound.a.01'),
     [Synset('unfree.a.02'),
      [Synset('confined.a.02')],
      [Synset('dependent.a.01')],
      [Synset('restricted.a.01'), [Synset('classified.a.02')]]]]
    """
    ...

def guess_encoding(data): # -> tuple[str | Any, str]:
    """
    Given a byte string, attempt to decode it.
    Tries the standard 'UTF8' and 'latin-1' encodings,
    Plus several gathered from locale information.

    The calling program *must* first call::

        locale.setlocale(locale.LC_ALL, '')

    If successful it returns ``(decoded_unicode, successful_encoding)``.
    If unsuccessful it raises a ``UnicodeError``.
    """
    ...

def unique_list(xs): # -> list:
    ...

def invert_dict(d): # -> defaultdict[Any, list]:
    ...

def transitive_closure(graph, reflexive=...): # -> dict[Any, set]:
    """
    Calculate the transitive closure of a directed graph,
    optionally the reflexive transitive closure.

    The algorithm is a slight modification of the "Marking Algorithm" of
    Ioannidis & Ramakrishnan (1998) "Efficient Transitive Closure Algorithms".

    :param graph: the initial graph, represented as a dictionary of sets
    :type graph: dict(set)
    :param reflexive: if set, also make the closure reflexive
    :type reflexive: bool
    :rtype: dict(set)
    """
    ...

def invert_graph(graph): # -> dict:
    """
    Inverts a directed graph.

    :param graph: the graph, represented as a dictionary of sets
    :type graph: dict(set)
    :return: the inverted graph
    :rtype: dict(set)
    """
    ...

def clean_html(html):
    ...

def clean_url(url):
    ...

def flatten(*args): # -> list:
    """
    Flatten a list.

        >>> from nltk.util import flatten
        >>> flatten(1, 2, ['b', 'a' , ['c', 'd']], 3)
        [1, 2, 'b', 'a', 'c', 'd', 3]

    :param args: items and lists to be combined into a single list
    :rtype: list
    """
    ...

def pad_sequence(sequence, n, pad_left=..., pad_right=..., left_pad_symbol=..., right_pad_symbol=...): # -> chain:
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    ...

def ngrams(sequence, n, **kwargs): # -> Generator[tuple, Any, None]:
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    ...

def bigrams(sequence, **kwargs): # -> Generator[tuple, Any, None]:
    """
    Return the bigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import bigrams
        >>> list(bigrams([1,2,3,4,5]))
        [(1, 2), (2, 3), (3, 4), (4, 5)]

    Use bigrams for a list version of this function.

    :param sequence: the source data to be converted into bigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """
    ...

def trigrams(sequence, **kwargs): # -> Generator[tuple, Any, None]:
    """
    Return the trigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import trigrams
        >>> list(trigrams([1,2,3,4,5]))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Use trigrams for a list version of this function.

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """
    ...

def everygrams(sequence, min_len=..., max_len=..., pad_left=..., pad_right=..., **kwargs): # -> Generator[tuple, Any, None]:
    """
    Returns all possible ngrams generated from a sequence of items, as an iterator.

        >>> sent = 'a b c'.split()

    New version outputs for everygrams.
        >>> list(everygrams(sent))
        [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]

    Old version outputs for everygrams.
        >>> sorted(everygrams(sent), key=len)
        [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]

        >>> list(everygrams(sent, max_len=2))
        [('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',)]

    :param sequence: the source data to be converted into ngrams. If max_len is
        not provided, this sequence will be loaded into memory
    :type sequence: sequence or iter
    :param min_len: minimum length of the ngrams, aka. n-gram order/degree of ngram
    :type  min_len: int
    :param max_len: maximum length of the ngrams (set to length of sequence by default)
    :type  max_len: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :rtype: iter(tuple)
    """
    ...

def skipgrams(sequence, n, k, **kwargs): # -> Generator[Any, Any, None]:
    """
    Returns all possible skipgrams generated from a sequence of items, as an iterator.
    Skipgrams are ngrams that allows tokens to be skipped.
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf

        >>> sent = "Insurgents killed in ongoing fighting".split()
        >>> list(skipgrams(sent, 2, 2))
        [('Insurgents', 'killed'), ('Insurgents', 'in'), ('Insurgents', 'ongoing'), ('killed', 'in'), ('killed', 'ongoing'), ('killed', 'fighting'), ('in', 'ongoing'), ('in', 'fighting'), ('ongoing', 'fighting')]
        >>> list(skipgrams(sent, 3, 2))
        [('Insurgents', 'killed', 'in'), ('Insurgents', 'killed', 'ongoing'), ('Insurgents', 'killed', 'fighting'), ('Insurgents', 'in', 'ongoing'), ('Insurgents', 'in', 'fighting'), ('Insurgents', 'ongoing', 'fighting'), ('killed', 'in', 'ongoing'), ('killed', 'in', 'fighting'), ('killed', 'ongoing', 'fighting'), ('in', 'ongoing', 'fighting')]

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param k: the skip distance
    :type  k: int
    :rtype: iter(tuple)
    """
    ...

def binary_search_file(file, key, cache=..., cacheDepth=...): # -> None:
    """
    Return the line from the file with first word key.
    Searches through a sorted file using the binary search algorithm.

    :type file: file
    :param file: the file to be searched through.
    :type key: str
    :param key: the identifier we are searching for.
    """
    ...

def set_proxy(proxy, user=..., password=...): # -> None:
    """
    Set the HTTP proxy for Python to download through.

    If ``proxy`` is None then tries to set proxy from environment or system
    settings.

    :param proxy: The HTTP proxy server to use. For example:
        'http://proxy.example.com:3128/'
    :param user: The username to authenticate with. Use None to disable
        authentication.
    :param password: The password to authenticate with.
    """
    ...

def elementtree_indent(elem, level=...): # -> None:
    """
    Recursive function to indent an ElementTree._ElementInterface
    used for pretty printing. Run indent on elem and then output
    in the normal way.

    :param elem: element to be indented. will be modified.
    :type elem: ElementTree._ElementInterface
    :param level: level of indentation for this element
    :type level: nonnegative integer
    :rtype:   ElementTree._ElementInterface
    :return:  Contents of elem indented to reflect its structure
    """
    ...

def choose(n, k): # -> int:
    """
    This function is a fast way to calculate binomial coefficients, commonly
    known as nCk, i.e. the number of combinations of n things taken k at a time.
    (https://en.wikipedia.org/wiki/Binomial_coefficient).

    This is the *scipy.special.comb()* with long integer computation but this
    approximation is faster, see https://github.com/nltk/nltk/issues/1181

        >>> choose(4, 2)
        6
        >>> choose(6, 2)
        15

    :param n: The number of things.
    :type n: int
    :param r: The number of times a thing is taken.
    :type r: int
    """
    ...

def pairwise(iterable): # -> zip[tuple]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    ...

def parallelize_preprocess(func, iterator, processes, progress_bar=...): # -> map | Generator[Any | None, Any, None] | list[Any | None]:
    ...
