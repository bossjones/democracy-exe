"""
This type stub file was generated by pyright.
"""

from nltk.stem.api import StemmerI

"""
ARLSTem2 Arabic Light Stemmer
The details about the implementation of this algorithm are described in:
K. Abainia and H. Rebbani, Comparing the Effectiveness of the Improved ARLSTem
Algorithm with Existing Arabic Light Stemmers, International Conference on
Theoretical and Applicative Aspects of Computer Science (ICTAACS'19), Skikda,
Algeria, December 15-16, 2019.
ARLSTem2 is an Arabic light stemmer based on removing the affixes from
the words (i.e. prefixes, suffixes and infixes). It is an improvement
of the previous Arabic light stemmer (ARLSTem). The new version was compared to
the original algorithm and several existing Arabic light stemmers, where the
results showed that the new version considerably improves the under-stemming
errors that are common to light stemmers. Both ARLSTem and ARLSTem2 can be run
online and do not use any dictionary.
"""
class ARLSTem2(StemmerI):
    """
    Return a stemmed Arabic word after removing affixes. This an improved
    version of the previous algorithm, which reduces under-stemming errors.
    Typically used in Arabic search engine, information retrieval and NLP.

        >>> from nltk.stem import arlstem2
        >>> stemmer = ARLSTem2()
        >>> word = stemmer.stem('يعمل')
        >>> print(word)
        عمل

    :param token: The input Arabic word (unicode) to be stemmed
    :type token: unicode
    :return: A unicode Arabic word
    """
    def __init__(self) -> None:
        ...

    def stem1(self, token):
        """
        call this function to get the first stem
        """
        ...

    def stem(self, token): # -> None:
        ...

    def norm(self, token): # -> str:
        """
        normalize the word by removing diacritics, replace hamzated Alif
        with Alif bare, replace AlifMaqsura with Yaa and remove Waaw at the
        beginning.
        """
        ...

    def pref(self, token): # -> None:
        """
        remove prefixes from the words' beginning.
        """
        ...

    def adjective(self, token): # -> None:
        """
        remove the infixes from adjectives
        """
        ...

    def suff(self, token):
        """
        remove the suffixes from the word's ending.
        """
        ...

    def fem2masc(self, token): # -> None:
        """
        transform the word from the feminine form to the masculine form.
        """
        ...

    def plur2sing(self, token): # -> None:
        """
        transform the word from the plural form to the singular form.
        """
        ...

    def verb(self, token):
        """
        stem the verb prefixes and suffixes or both
        """
        ...

    def verb_t1(self, token): # -> None:
        """
        stem the present tense co-occurred prefixes and suffixes
        """
        ...

    def verb_t2(self, token): # -> None:
        """
        stem the future tense co-occurred prefixes and suffixes
        """
        ...

    def verb_t3(self, token): # -> None:
        """
        stem the present tense suffixes
        """
        ...

    def verb_t4(self, token): # -> None:
        """
        stem the present tense prefixes
        """
        ...

    def verb_t5(self, token): # -> None:
        """
        stem the future tense prefixes
        """
        ...

    def verb_t6(self, token):
        """
        stem the imperative tense prefixes
        """
        ...
