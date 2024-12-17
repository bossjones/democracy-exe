"""
This type stub file was generated by pyright.
"""

def extract(f_start, f_end, e_start, e_end, alignment, f_aligned, srctext, trgtext, srclen, trglen, max_phrase_length): # -> dict[Any, Any] | set[Any]:
    """
    This function checks for alignment point consistency and extracts
    phrases using the chunk of consistent phrases.

    A phrase pair (e, f ) is consistent with an alignment A if and only if:

    (i) No English words in the phrase pair are aligned to words outside it.

           ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f

    (ii) No Foreign words in the phrase pair are aligned to words outside it.

            ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e

    (iii) The phrase pair contains at least one alignment point.

            ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A

    :type f_start: int
    :param f_start: Starting index of the possible foreign language phrases
    :type f_end: int
    :param f_end: End index of the possible foreign language phrases
    :type e_start: int
    :param e_start: Starting index of the possible source language phrases
    :type e_end: int
    :param e_end: End index of the possible source language phrases
    :type srctext: list
    :param srctext: The source language tokens, a list of string.
    :type trgtext: list
    :param trgtext: The target language tokens, a list of string.
    :type srclen: int
    :param srclen: The number of tokens in the source language tokens.
    :type trglen: int
    :param trglen: The number of tokens in the target language tokens.
    """
    ...

def phrase_extraction(srctext, trgtext, alignment, max_phrase_length=...): # -> set[Any]:
    """
    Phrase extraction algorithm extracts all consistent phrase pairs from
    a word-aligned sentence pair.

    The idea is to loop over all possible source language (e) phrases and find
    the minimal foreign phrase (f) that matches each of them. Matching is done
    by identifying all alignment points for the source phrase and finding the
    shortest foreign phrase that includes all the foreign counterparts for the
    source words.

    In short, a phrase alignment has to
    (a) contain all alignment points for all covered words
    (b) contain at least one alignment point

    >>> srctext = "michael assumes that he will stay in the house"
    >>> trgtext = "michael geht davon aus , dass er im haus bleibt"
    >>> alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9),
    ... (5,9), (6,7), (7,7), (8,8)]
    >>> phrases = phrase_extraction(srctext, trgtext, alignment)
    >>> for i in sorted(phrases):
    ...    print(i)
    ...
    ((0, 1), (0, 1), 'michael', 'michael')
    ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus')
    ((0, 2), (0, 5), 'michael assumes', 'michael geht davon aus ,')
    ((0, 3), (0, 6), 'michael assumes that', 'michael geht davon aus , dass')
    ((0, 4), (0, 7), 'michael assumes that he', 'michael geht davon aus , dass er')
    ((0, 9), (0, 10), 'michael assumes that he will stay in the house', 'michael geht davon aus , dass er im haus bleibt')
    ((1, 2), (1, 4), 'assumes', 'geht davon aus')
    ((1, 2), (1, 5), 'assumes', 'geht davon aus ,')
    ((1, 3), (1, 6), 'assumes that', 'geht davon aus , dass')
    ((1, 4), (1, 7), 'assumes that he', 'geht davon aus , dass er')
    ((1, 9), (1, 10), 'assumes that he will stay in the house', 'geht davon aus , dass er im haus bleibt')
    ((2, 3), (4, 6), 'that', ', dass')
    ((2, 3), (5, 6), 'that', 'dass')
    ((2, 4), (4, 7), 'that he', ', dass er')
    ((2, 4), (5, 7), 'that he', 'dass er')
    ((2, 9), (4, 10), 'that he will stay in the house', ', dass er im haus bleibt')
    ((2, 9), (5, 10), 'that he will stay in the house', 'dass er im haus bleibt')
    ((3, 4), (6, 7), 'he', 'er')
    ((3, 9), (6, 10), 'he will stay in the house', 'er im haus bleibt')
    ((4, 6), (9, 10), 'will stay', 'bleibt')
    ((4, 9), (7, 10), 'will stay in the house', 'im haus bleibt')
    ((6, 8), (7, 8), 'in the', 'im')
    ((6, 9), (7, 9), 'in the house', 'im haus')
    ((8, 9), (8, 9), 'house', 'haus')

    :type srctext: str
    :param srctext: The sentence string from the source language.
    :type trgtext: str
    :param trgtext: The sentence string from the target language.
    :type alignment: list(tuple)
    :param alignment: The word alignment outputs as list of tuples, where
        the first elements of tuples are the source words' indices and
        second elements are the target words' indices. This is also the output
        format of nltk.translate.ibm1
    :rtype: list(tuple)
    :return: A list of tuples, each element in a list is a phrase and each
        phrase is a tuple made up of (i) its source location, (ii) its target
        location, (iii) the source phrase and (iii) the target phrase. The phrase
        list of tuples represents all the possible phrases extracted from the
        word alignments.
    :type max_phrase_length: int
    :param max_phrase_length: maximal phrase length, if 0 or not specified
        it is set to a length of the longer sentence (srctext or trgtext).
    """
    ...

