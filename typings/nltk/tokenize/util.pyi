"""
This type stub file was generated by pyright.
"""

def string_span_tokenize(s, sep): # -> Generator[tuple[Any | int, Any] | tuple[Any | int, int], Any, None]:
    r"""
    Return the offsets of the tokens in *s*, as a sequence of ``(start, end)``
    tuples, by splitting the string at each occurrence of *sep*.

        >>> from nltk.tokenize.util import string_span_tokenize
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
        ... two of them.\n\nThanks.'''
        >>> list(string_span_tokenize(s, " ")) # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (5, 12), (13, 17), (18, 26), (27, 30), (31, 36), (37, 37),
        (38, 44), (45, 48), (49, 55), (56, 58), (59, 73)]

    :param s: the string to be tokenized
    :type s: str
    :param sep: the token separator
    :type sep: str
    :rtype: iter(tuple(int, int))
    """
    ...

def regexp_span_tokenize(s, regexp): # -> Generator[tuple[Any | Literal[0], Any] | tuple[Any | Literal[0], int], Any, None]:
    r"""
    Return the offsets of the tokens in *s*, as a sequence of ``(start, end)``
    tuples, by splitting the string at each successive match of *regexp*.

        >>> from nltk.tokenize.util import regexp_span_tokenize
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
        ... two of them.\n\nThanks.'''
        >>> list(regexp_span_tokenize(s, r'\s')) # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36),
        (38, 44), (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]

    :param s: the string to be tokenized
    :type s: str
    :param regexp: regular expression that matches token separators (must not be empty)
    :type regexp: str
    :rtype: iter(tuple(int, int))
    """
    ...

def spans_to_relative(spans): # -> Generator[tuple[Any, Any], Any, None]:
    r"""
    Return a sequence of relative spans, given a sequence of spans.

        >>> from nltk.tokenize import WhitespaceTokenizer
        >>> from nltk.tokenize.util import spans_to_relative
        >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me
        ... two of them.\n\nThanks.'''
        >>> list(spans_to_relative(WhitespaceTokenizer().span_tokenize(s))) # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (1, 7), (1, 4), (1, 5), (1, 2), (1, 3), (1, 5), (2, 6),
        (1, 3), (1, 2), (1, 3), (1, 2), (1, 5), (2, 7)]

    :param spans: a sequence of (start, end) offsets of the tokens
    :type spans: iter(tuple(int, int))
    :rtype: iter(tuple(int, int))
    """
    ...

class CJKChars:
    """
    An object that enumerates the code points of the CJK characters as listed on
    https://en.wikipedia.org/wiki/Basic_Multilingual_Plane#Basic_Multilingual_Plane

    This is a Python port of the CJK code point enumerations of Moses tokenizer:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl#L309
    """
    Hangul_Jamo = ...
    CJK_Radicals = ...
    Phags_Pa = ...
    Hangul_Syllables = ...
    CJK_Compatibility_Ideographs = ...
    CJK_Compatibility_Forms = ...
    Katakana_Hangul_Halfwidth = ...
    Supplementary_Ideographic_Plane = ...
    ranges = ...


def is_cjk(character): # -> bool:
    """
    Python port of Moses' code to check for CJK character.

    >>> CJKChars().ranges
    [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215), (63744, 64255), (65072, 65103), (65381, 65500), (131072, 196607)]
    >>> is_cjk(u'\u33fe')
    True
    >>> is_cjk(u'\uFE5F')
    False

    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    ...

def xml_escape(text): # -> str:
    """
    This function transforms the input text into an "escaped" version suitable
    for well-formed XML formatting.

    Note that the default xml.sax.saxutils.escape() function don't escape
    some characters that Moses does so we have to manually add them to the
    entities dictionary.

        >>> input_str = ''')| & < > ' " ] ['''
        >>> expected_output =  ''')| &amp; &lt; &gt; ' " ] ['''
        >>> escape(input_str) == expected_output
        True
        >>> xml_escape(input_str)
        ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'

    :param text: The text that needs to be escaped.
    :type text: str
    :rtype: str
    """
    ...

def xml_unescape(text): # -> str:
    """
    This function transforms the "escaped" version suitable
    for well-formed XML formatting into humanly-readable string.

    Note that the default xml.sax.saxutils.unescape() function don't unescape
    some characters that Moses does so we have to manually add them to the
    entities dictionary.

        >>> from xml.sax.saxutils import unescape
        >>> s = ')&#124; &amp; &lt; &gt; &apos; &quot; &#93; &#91;'
        >>> expected = ''')| & < > \' " ] ['''
        >>> xml_unescape(s) == expected
        True

    :param text: The text that needs to be unescaped.
    :type text: str
    :rtype: str
    """
    ...

def align_tokens(tokens, sentence): # -> list[Any]:
    """
    This module attempt to find the offsets of the tokens in *s*, as a sequence
    of ``(start, end)`` tuples, given the tokens and also the source string.

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> from nltk.tokenize.util import align_tokens
        >>> s = str("The plane, bound for St Petersburg, crashed in Egypt's "
        ... "Sinai desert just 23 minutes after take-off from Sharm el-Sheikh "
        ... "on Saturday.")
        >>> tokens = TreebankWordTokenizer().tokenize(s)
        >>> expected = [(0, 3), (4, 9), (9, 10), (11, 16), (17, 20), (21, 23),
        ... (24, 34), (34, 35), (36, 43), (44, 46), (47, 52), (52, 54),
        ... (55, 60), (61, 67), (68, 72), (73, 75), (76, 83), (84, 89),
        ... (90, 98), (99, 103), (104, 109), (110, 119), (120, 122),
        ... (123, 131), (131, 132)]
        >>> output = list(align_tokens(tokens, s))
        >>> len(tokens) == len(expected) == len(output)  # Check that length of tokens and tuples are the same.
        True
        >>> expected == list(align_tokens(tokens, s))  # Check that the output is as expected.
        True
        >>> tokens == [s[start:end] for start, end in output]  # Check that the slices of the string corresponds to the tokens.
        True

    :param tokens: The list of strings that are the result of tokenization
    :type tokens: list(str)
    :param sentence: The original string
    :type sentence: str
    :rtype: list(tuple(int,int))
    """
    ...

