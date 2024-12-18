"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader import CorpusReader

class LinThesaurusCorpusReader(CorpusReader):
    """Wrapper for the LISP-formatted thesauruses distributed by Dekang Lin."""
    _key_re = ...
    def __init__(self, root, badscore=...) -> None:
        """
        Initialize the thesaurus.

        :param root: root directory containing thesaurus LISP files
        :type root: C{string}
        :param badscore: the score to give to words which do not appear in each other's sets of synonyms
        :type badscore: C{float}
        """
        ...
    
    def similarity(self, ngram1, ngram2, fileid=...): # -> float | list[tuple[str | Any, float]] | list[tuple[str | Any, Any | float]]:
        """
        Returns the similarity score for two ngrams.

        :param ngram1: first ngram to compare
        :type ngram1: C{string}
        :param ngram2: second ngram to compare
        :type ngram2: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, just the score for the two ngrams; otherwise,
                 list of tuples of fileids and scores.
        """
        ...
    
    def scored_synonyms(self, ngram, fileid=...): # -> dict_items[Any, Any] | list[tuple[str | Any, dict_items[Any, Any]]]:
        """
        Returns a list of scored synonyms (tuples of synonyms and scores) for the current ngram

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, list of tuples of scores and synonyms; otherwise,
                 list of tuples of fileids and lists, where inner lists consist of tuples of
                 scores and synonyms.
        """
        ...
    
    def synonyms(self, ngram, fileid=...): # -> dict_keys[Any, Any] | list[tuple[str | Any, dict_keys[Any, Any]]]:
        """
        Returns a list of synonyms for the current ngram.

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :param fileid: thesaurus fileid to search in. If None, search all fileids.
        :type fileid: C{string}
        :return: If fileid is specified, list of synonyms; otherwise, list of tuples of fileids and
                 lists, where inner lists contain synonyms.
        """
        ...
    
    def __contains__(self, ngram): # -> bool:
        """
        Determines whether or not the given ngram is in the thesaurus.

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :return: whether the given ngram is in the thesaurus.
        """
        ...
    


def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
