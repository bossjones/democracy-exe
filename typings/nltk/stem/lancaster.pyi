"""
This type stub file was generated by pyright.
"""

from nltk.stem.api import StemmerI

"""
A word stemmer based on the Lancaster (Paice/Husk) stemming algorithm.
Paice, Chris D. "Another Stemmer." ACM SIGIR Forum 24.3 (1990): 56-61.
"""
class LancasterStemmer(StemmerI):
    """
    Lancaster Stemmer

        >>> from nltk.stem.lancaster import LancasterStemmer
        >>> st = LancasterStemmer()
        >>> st.stem('maximum')     # Remove "-um" when word is intact
        'maxim'
        >>> st.stem('presumably')  # Don't remove "-um" when word is not intact
        'presum'
        >>> st.stem('multiply')    # No action taken if word ends with "-ply"
        'multiply'
        >>> st.stem('provision')   # Replace "-sion" with "-j" to trigger "j" set of rules
        'provid'
        >>> st.stem('owed')        # Word starting with vowel must contain at least 2 letters
        'ow'
        >>> st.stem('ear')         # ditto
        'ear'
        >>> st.stem('saying')      # Words starting with consonant must contain at least 3
        'say'
        >>> st.stem('crying')      #     letters and one of those letters must be a vowel
        'cry'
        >>> st.stem('string')      # ditto
        'string'
        >>> st.stem('meant')       # ditto
        'meant'
        >>> st.stem('cement')      # ditto
        'cem'
        >>> st_pre = LancasterStemmer(strip_prefix_flag=True)
        >>> st_pre.stem('kilometer') # Test Prefix
        'met'
        >>> st_custom = LancasterStemmer(rule_tuple=("ssen4>", "s1t."))
        >>> st_custom.stem("ness") # Change s to t
        'nest'
    """
    default_rule_tuple = ...
    def __init__(self, rule_tuple=..., strip_prefix_flag=...) -> None:
        """Create an instance of the Lancaster stemmer."""
        ...
    
    def parseRules(self, rule_tuple=...): # -> None:
        """Validate the set of rules used in this stemmer.

        If this function is called as an individual method, without using stem
        method, rule_tuple argument will be compiled into self.rule_dictionary.
        If this function is called within stem, self._rule_tuple will be used.

        """
        ...
    
    def stem(self, word):
        """Stem a word using the Lancaster stemmer."""
        ...
    
    def __repr__(self): # -> Literal['<LancasterStemmer>']:
        ...
    


