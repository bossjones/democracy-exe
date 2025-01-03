"""
This type stub file was generated by pyright.
"""

import sys
import emoji.unicode_codes
from emoji.core import *
from emoji.unicode_codes import *

"""
emoji for Python
~~~~~~~~~~~~~~~~

emoji terminal output for Python.

    >>> import emoji
    >>> print(emoji.emojize('Python is :thumbsup:', language='alias'))
    Python is 👍
    >>> print(emoji.emojize('Python is :thumbs_up:'))
    Python is 👍
"""
if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    ...
else:
    ...
__all__ = ['emojize', 'demojize', 'get_emoji_regexp', 'emoji_count', 'emoji_lis', 'distinct_emoji_lis', 'replace_emoji', 'version', 'is_emoji', 'emoji_list', 'distinct_emoji_list', 'EMOJI_UNICODE_ENGLISH', 'EMOJI_UNICODE_SPANISH', 'EMOJI_UNICODE_PORTUGUESE', 'EMOJI_UNICODE_ITALIAN', 'EMOJI_UNICODE_FRENCH', 'EMOJI_UNICODE_GERMAN', 'UNICODE_EMOJI_ENGLISH', 'UNICODE_EMOJI_SPANISH', 'UNICODE_EMOJI_PORTUGUESE', 'UNICODE_EMOJI_ITALIAN', 'UNICODE_EMOJI_FRENCH', 'UNICODE_EMOJI_GERMAN', 'EMOJI_ALIAS_UNICODE_ENGLISH', 'UNICODE_EMOJI_ALIAS_ENGLISH', 'EMOJI_DATA']
__version__ = ...
__author__ = ...
__email__ = ...
__source__ = ...
__license__ = ...
_DEPRECATED = ...
def __getattr__(varname): # -> Any:
    ...

