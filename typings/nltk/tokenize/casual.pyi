"""
This type stub file was generated by pyright.
"""

import regex
from typing import List
from nltk.tokenize.api import TokenizerI

"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new
domains and tasks. The basic logic is this:

1. The tuple REGEXPS defines a list of regular expression
   strings.

2. The REGEXPS strings are put, in order, into a compiled
   regular expression object called WORD_RE, under the TweetTokenizer
   class.

3. The tokenization is done by WORD_RE.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   TweetTokenizer.

4. When instantiating Tokenizer objects, there are several options:
    * preserve_case. By default, it is set to True. If it is set to
      False, then the tokenizer will downcase everything except for
      emoticons.
    * reduce_len. By default, it is set to False. It specifies whether
      to replace repeated character sequences of length 3 or greater
      with sequences of length 3.
    * strip_handles. By default, it is set to False. It specifies
      whether to remove Twitter handles of text used in the
      `tokenize` method.
    * match_phone_numbers. By default, it is set to True. It indicates
      whether the `tokenize` method should look for phone numbers.
"""
EMOTICONS = ...
URLS = ...
FLAGS = ...
PHONE_REGEX = ...
REGEXPS = ...
REGEXPS_PHONE = ...
HANG_RE = ...
EMOTICON_RE = ...
ENT_RE = ...
HANDLES_RE = ...
class TweetTokenizer(TokenizerI):
    r"""
    Tokenizer for tweets.

        >>> from nltk.tokenize import TweetTokenizer
        >>> tknzr = TweetTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0) # doctest: +NORMALIZE_WHITESPACE
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->',
         '<--']

    Examples using `strip_handles` and `reduce_len parameters`:

        >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
        >>> tknzr.tokenize(s1)
        [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    """
    _WORD_RE = ...
    _PHONE_WORD_RE = ...
    def __init__(self, preserve_case=..., reduce_len=..., strip_handles=..., match_phone_numbers=...) -> None:
        """
        Create a `TweetTokenizer` instance with settings for use in the `tokenize` method.

        :param preserve_case: Flag indicating whether to preserve the casing (capitalisation)
            of text used in the `tokenize` method. Defaults to True.
        :type preserve_case: bool
        :param reduce_len: Flag indicating whether to replace repeated character sequences
            of length 3 or greater with sequences of length 3. Defaults to False.
        :type reduce_len: bool
        :param strip_handles: Flag indicating whether to remove Twitter handles of text used
            in the `tokenize` method. Defaults to False.
        :type strip_handles: bool
        :param match_phone_numbers: Flag indicating whether the `tokenize` method should look
            for phone numbers. Defaults to True.
        :type match_phone_numbers: bool
        """
        ...
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.

        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; joining this list returns\
        the original string if `preserve_case=False`.
        """
        ...
    
    @property
    def WORD_RE(self) -> regex.Pattern:
        """Core TweetTokenizer regex"""
        ...
    
    @property
    def PHONE_WORD_RE(self) -> regex.Pattern:
        """Secondary core TweetTokenizer regex"""
        ...
    


def reduce_lengthening(text): # -> str:
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    """
    ...

def remove_handles(text): # -> str:
    """
    Remove Twitter username handles from text.
    """
    ...

def casual_tokenize(text, preserve_case=..., reduce_len=..., strip_handles=..., match_phone_numbers=...): # -> List[str]:
    """
    Convenience function for wrapping the tokenizer.
    """
    ...

