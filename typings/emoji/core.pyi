"""
This type stub file was generated by pyright.
"""

"""
emoji.core
~~~~~~~~~~

Core components for emoji.

"""
__all__ = ['emojize', 'demojize', 'get_emoji_regexp', 'emoji_lis', 'distinct_emoji_lis', 'emoji_count', 'replace_emoji', 'is_emoji', 'version', 'emoji_list', 'distinct_emoji_list']
PY2 = ...
_EMOJI_REGEXP = ...
_SEARCH_TREE = ...
_DEFAULT_DELIMITER = ...
class _DeprecatedParameter:
    ...


def emojize(string, use_aliases=..., delimiters=..., variant=..., language=..., version=..., handle_version=...):
    """Replace emoji names in a string with unicode codes.
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbsup:", language='alias'))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun __thumbs_up__", delimiters = ("__", "__")))
        Python is fun 👍
        >>> print(emoji.emojize("Python is fun :red_heart:",variant="text_type"))
        Python is fun ❤
        >>> print(emoji.emojize("Python is fun :red_heart:",variant="emoji_type"))
        Python is fun ❤️ #red heart, not black heart

    :param string: String contains emoji names.
    :param use_aliases: (optional) Deprecated. Use language='alias' instead
    :param delimiters: (optional) Use delimiters other than _DEFAULT_DELIMITER
    :param variant: (optional) Choose variation selector between "base"(None), VS-15 ("text_type") and VS-16 ("emoji_type")
    :param language: Choose language of emoji name: language code 'es', 'de', etc. or 'alias'
        to use English aliases
    :param version: (optional) Max version. If set to an Emoji Version,
        all emoji above this version will be ignored.
    :param handle_version: (optional) Replace the emoji above ``version``
        instead of ignoring it. handle_version can be either a string or a
        callable; If it is a callable, it's passed the unicode emoji and the
        data dict from emoji.EMOJI_DATA and must return a replacement string
        to be used::

            handle_version(u'\\U0001F6EB', {
                'en' : ':airplane_departure:',
                'status' : fully_qualified,
                'E' : 1,
                'alias' : [u':flight_departure:'],
                'de': u':abflug:',
                'es': u':avión_despegando:',
                ...
            })

    :raises ValueError: if ``variant`` is neither None, 'text_type' or 'emoji_type'

    """
    ...

def demojize(string, use_aliases=..., delimiters=..., language=..., version=..., handle_version=...): # -> LiteralString:
    """Replace unicode emoji in a string with emoji shortcodes. Useful for storage.
        >>> import emoji
        >>> print(emoji.emojize("Python is fun :thumbs_up:"))
        Python is fun 👍
        >>> print(emoji.demojize(u"Python is fun 👍"))
        Python is fun :thumbs_up:
        >>> print(emoji.demojize(u"Unicode is tricky 😯", delimiters=("__", "__")))
        Unicode is tricky __hushed_face__

    :param string: String contains unicode characters. MUST BE UNICODE.
    :param use_aliases: (optional) Deprecated. Use language='alias' instead
    :param delimiters: (optional) User delimiters other than ``_DEFAULT_DELIMITER``
    :param language: Choose language of emoji name: language code 'es', 'de', etc. or 'alias'
        to use English aliases
    :param version: (optional) Max version. If set to an Emoji Version,
        all emoji above this version will be removed.
    :param handle_version: (optional) Replace the emoji above ``version``
        instead of removing it. handle_version can be either a string or a
        callable ``handle_version(emj: str, data: dict) -> str``; If it is
        a callable, it's passed the unicode emoji and the data dict from
        emoji.EMOJI_DATA and must return a replacement string  to be used.
        The passed data is in the form of::

            handle_version(u'\\U0001F6EB', {
                'en' : ':airplane_departure:',
                'status' : fully_qualified,
                'E' : 1,
                'alias' : [u':flight_departure:'],
                'de': u':abflug:',
                'es': u':avión_despegando:',
                ...
            })

    """
    ...

def replace_emoji(string, replace=..., language=..., version=...): # -> LiteralString:
    """Replace unicode emoji in a customizable string.

    :param string: String contains unicode characters. MUST BE UNICODE.
    :param replace: (optional) replace can be either a string or a callable;
        If it is a callable, it's passed the unicode emoji and the data dict from
        emoji.EMOJI_DATA and must return a replacement string to be used.
        replace(str, dict) -> str
    :param version: (optional) Max version. If set to an Emoji Version,
        only emoji above this version will be replaced.
    :param language: (optional) Deprecated and has no effect
    """
    ...

def get_emoji_regexp(language=...): # -> Pattern[str]:
    """Returns compiled regular expression that matches all emojis defined in
    ``emoji.EMOJI_DATA``. The regular expression is only compiled once.

    :param language: (optional) Parameter is no longer used
    """
    ...

def emoji_lis(string, language=...): # -> list[Any]:
    """Returns the location and emoji in list of dict format.
        >>> emoji.emoji_lis("Hi, I am fine. 😁")
        [{'location': 15, 'emoji': '😁'}]

    :param language: (optional) Deprecated and has no effect
    """
    ...

def emoji_list(string): # -> list[Any]:
    """Returns the location and emoji in list of dict format.
        >>> emoji.emoji_list("Hi, I am fine. 😁")
        [{'match_start': 15, 'match_end': 16, 'emoji': '😁'}]y

    """
    ...

def distinct_emoji_lis(string, language=...): # -> list[Any]:
    """Returns distinct list of emojis from the string.

    :param language: (optional) Deprecated and has no effect
    """
    ...

def distinct_emoji_list(string): # -> list[Any]:
    """Returns distinct list of emojis from the string.
    """
    ...

def emoji_count(string, unique=...): # -> int:
    """Returns the count of emojis in a string.

    :param unique: (optional) True if count only unique emojis
    """
    ...

def is_emoji(string): # -> bool:
    """Returns True if the string is an emoji"""
    ...

def version(string):
    """Returns the Emoji Version of the emoji.
      See http://www.unicode.org/reports/tr51/#Versioning for more information.
        >>> emoji.version("😁")
        >>> 0.6
        >>> emoji.version(":butterfly:")
        >>> 3

    :param string: An emoji or a text containig an emoji
    :raises ValueError: if ``string`` does not contain an emoji
    """
    ...

