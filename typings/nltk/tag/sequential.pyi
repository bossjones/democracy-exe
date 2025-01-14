"""
This type stub file was generated by pyright.
"""

from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.tag.api import FeaturesetTaggerI, TaggerI

"""
Classes for tagging sentences sequentially, left to right.  The
abstract base class SequentialBackoffTagger serves as the base
class for all the taggers in this module.  Tagging of individual words
is performed by the method ``choose_tag()``, which is defined by
subclasses of SequentialBackoffTagger.  If a tagger is unable to
determine a tag for the specified token, then its backoff tagger is
consulted instead.  Any SequentialBackoffTagger may serve as a
backoff tagger for any other SequentialBackoffTagger.
"""
class SequentialBackoffTagger(TaggerI):
    """
    An abstract base class for taggers that tags words sequentially,
    left to right.  Tagging of individual words is performed by the
    ``choose_tag()`` method, which should be defined by subclasses.  If
    a tagger is unable to determine a tag for the specified token,
    then its backoff tagger is consulted.

    :ivar _taggers: A list of all the taggers that should be tried to
        tag a token (i.e., self and its backoff taggers).
    """
    def __init__(self, backoff=...) -> None:
        ...
    
    @property
    def backoff(self): # -> Self | None:
        """The backoff tagger for this tagger."""
        ...
    
    def tag(self, tokens): # -> list[tuple[Any, Any]]:
        ...
    
    def tag_one(self, tokens, index, history): # -> None:
        """
        Determine an appropriate tag for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, then its backoff tagger is consulted.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """
        ...
    
    @abstractmethod
    def choose_tag(self, tokens, index, history): # -> None:
        """
        Decide which tag should be used for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, return None -- do not consult
        the backoff tagger.  This method should be overridden by
        subclasses of SequentialBackoffTagger.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """
        ...
    


class ContextTagger(SequentialBackoffTagger):
    """
    An abstract base class for sequential backoff taggers that choose
    a tag for a token based on the value of its "context".  Different
    subclasses are used to define different contexts.

    A ContextTagger chooses the tag for a token by calculating the
    token's context, and looking up the corresponding tag in a table.
    This table can be constructed manually; or it can be automatically
    constructed based on a training corpus, using the ``_train()``
    factory method.

    :ivar _context_to_tag: Dictionary mapping contexts to tags.
    """
    def __init__(self, context_to_tag, backoff=...) -> None:
        """
        :param context_to_tag: A dictionary mapping contexts to tags.
        :param backoff: The backoff tagger that should be used for this tagger.
        """
        ...
    
    @abstractmethod
    def context(self, tokens, index, history): # -> None:
        """
        :return: the context that should be used to look up the tag
            for the specified token; or None if the specified token
            should not be handled by this tagger.
        :rtype: (hashable)
        """
        ...
    
    def choose_tag(self, tokens, index, history): # -> None:
        ...
    
    def size(self): # -> int:
        """
        :return: The number of entries in the table used by this
            tagger to map from contexts to tags.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


@jsontags.register_tag
class DefaultTagger(SequentialBackoffTagger):
    """
    A tagger that assigns the same tag to every token.

        >>> from nltk.tag import DefaultTagger
        >>> default_tagger = DefaultTagger('NN')
        >>> list(default_tagger.tag('This is a test'.split()))
        [('This', 'NN'), ('is', 'NN'), ('a', 'NN'), ('test', 'NN')]

    This tagger is recommended as a backoff tagger, in cases where
    a more powerful tagger is unable to assign a tag to the word
    (e.g. because the word was not seen during training).

    :param tag: The tag to assign to each token
    :type tag: str
    """
    json_tag = ...
    def __init__(self, tag) -> None:
        ...
    
    def encode_json_obj(self): # -> Any:
        ...
    
    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...
    
    def choose_tag(self, tokens, index, history): # -> Any:
        ...
    
    def __repr__(self): # -> str:
        ...
    


@jsontags.register_tag
class NgramTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on its word string and
    on the preceding n word's tags.  In particular, a tuple
    (tags[i-n:i-1], words[i]) is looked up in a table, and the
    corresponding tag is returned.  N-gram taggers are typically
    trained on a tagged corpus.

    Train a new NgramTagger using the given training data or
    the supplied model.  In particular, construct a new tagger
    whose table maps from each context (tag[i-n:i-1], word[i])
    to the most frequent tag for that context.  But exclude any
    contexts that are already tagged perfectly by the backoff
    tagger.

    :param train: A tagged corpus consisting of a list of tagged
        sentences, where each sentence is a list of (word, tag) tuples.
    :param backoff: A backoff tagger, to be used by the new
        tagger if it encounters an unknown context.
    :param cutoff: If the most likely tag for a context occurs
        fewer than *cutoff* times, then exclude it from the
        context-to-tag table for the new tagger.
    """
    json_tag = ...
    def __init__(self, n, train=..., model=..., backoff=..., cutoff=..., verbose=...) -> None:
        ...
    
    def encode_json_obj(self): # -> tuple[Any, dict[str, Any], Self | Any | None] | tuple[dict[str, Any], Self | Any | None]:
        ...
    
    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...
    
    def context(self, tokens, index, history): # -> tuple[tuple[Any, ...], Any]:
        ...
    


@jsontags.register_tag
class UnigramTagger(NgramTagger):
    """
    Unigram Tagger

    The UnigramTagger finds the most likely tag for each word in a training
    corpus, and then uses that information to assign tags to new tokens.

        >>> from nltk.corpus import brown
        >>> from nltk.tag import UnigramTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> unigram_tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
        >>> for tok, tag in unigram_tagger.tag(test_sent):
        ...     print("({}, {}), ".format(tok, tag)) # doctest: +NORMALIZE_WHITESPACE
        (The, AT), (Fulton, NP-TL), (County, NN-TL), (Grand, JJ-TL),
        (Jury, NN-TL), (said, VBD), (Friday, NR), (an, AT),
        (investigation, NN), (of, IN), (Atlanta's, NP$), (recent, JJ),
        (primary, NN), (election, NN), (produced, VBD), (``, ``),
        (no, AT), (evidence, NN), ('', ''), (that, CS), (any, DTI),
        (irregularities, NNS), (took, VBD), (place, NN), (., .),

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """
    json_tag = ...
    def __init__(self, train=..., model=..., backoff=..., cutoff=..., verbose=...) -> None:
        ...
    
    def context(self, tokens, index, history):
        ...
    


@jsontags.register_tag
class BigramTagger(NgramTagger):
    """
    A tagger that chooses a token's tag based its word string and on
    the preceding words' tag.  In particular, a tuple consisting
    of the previous tag and the word is looked up in a table, and
    the corresponding tag is returned.

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """
    json_tag = ...
    def __init__(self, train=..., model=..., backoff=..., cutoff=..., verbose=...) -> None:
        ...
    


@jsontags.register_tag
class TrigramTagger(NgramTagger):
    """
    A tagger that chooses a token's tag based its word string and on
    the preceding two words' tags.  In particular, a tuple consisting
    of the previous two tags and the word is looked up in a table, and
    the corresponding tag is returned.

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """
    json_tag = ...
    def __init__(self, train=..., model=..., backoff=..., cutoff=..., verbose=...) -> None:
        ...
    


@jsontags.register_tag
class AffixTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on a leading or trailing
    substring of its word string.  (It is important to note that these
    substrings are not necessarily "true" morphological affixes).  In
    particular, a fixed-length substring of the word is looked up in a
    table, and the corresponding tag is returned.  Affix taggers are
    typically constructed by training them on a tagged corpus.

    Construct a new affix tagger.

    :param affix_length: The length of the affixes that should be
        considered during training and tagging.  Use negative
        numbers for suffixes.
    :param min_stem_length: Any words whose length is less than
        min_stem_length+abs(affix_length) will be assigned a
        tag of None by this tagger.
    """
    json_tag = ...
    def __init__(self, train=..., model=..., affix_length=..., min_stem_length=..., backoff=..., cutoff=..., verbose=...) -> None:
        ...
    
    def encode_json_obj(self): # -> tuple[int, int, Any | dict[Any, Any], Self | Any | None]:
        ...
    
    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...
    
    def context(self, tokens, index, history): # -> None:
        ...
    


@jsontags.register_tag
class RegexpTagger(SequentialBackoffTagger):
    r"""
    Regular Expression Tagger

    The RegexpTagger assigns tags to tokens by comparing their
    word strings to a series of regular expressions.  The following tagger
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag:

        >>> from nltk.corpus import brown
        >>> from nltk.tag import RegexpTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> regexp_tagger = RegexpTagger(
        ...     [(r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
        ...      (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        ...      (r'.*able$', 'JJ'),                # adjectives
        ...      (r'.*ness$', 'NN'),                # nouns formed from adjectives
        ...      (r'.*ly$', 'RB'),                  # adverbs
        ...      (r'.*s$', 'NNS'),                  # plural nouns
        ...      (r'.*ing$', 'VBG'),                # gerunds
        ...      (r'.*ed$', 'VBD'),                 # past tense verbs
        ...      (r'.*', 'NN')                      # nouns (default)
        ... ])
        >>> regexp_tagger
        <Regexp Tagger: size=9>
        >>> regexp_tagger.tag(test_sent) # doctest: +NORMALIZE_WHITESPACE
        [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'), ('Grand', 'NN'), ('Jury', 'NN'),
        ('said', 'NN'), ('Friday', 'NN'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
        ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'), ('election', 'NN'),
        ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'), ('evidence', 'NN'), ("''", 'NN'),
        ('that', 'NN'), ('any', 'NN'), ('irregularities', 'NNS'), ('took', 'NN'),
        ('place', 'NN'), ('.', 'NN')]

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, tag)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be tagged with ``tag``.  The pairs will be evaluated in
        order.  If none of the regexps match a word, then the
        optional backoff tagger is invoked, else it is
        assigned the tag None.
    """
    json_tag = ...
    def __init__(self, regexps: List[Tuple[str, str]], backoff: Optional[TaggerI] = ...) -> None:
        ...
    
    def encode_json_obj(self): # -> tuple[list[tuple[Any, Any]], Self | Any | None]:
        ...
    
    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...
    
    def choose_tag(self, tokens, index, history): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class ClassifierBasedTagger(SequentialBackoffTagger, FeaturesetTaggerI):
    """
    A sequential tagger that uses a classifier to choose the tag for
    each token in a sentence.  The featureset input for the classifier
    is generated by a feature detector function::

        feature_detector(tokens, index, history) -> featureset

    Where tokens is the list of unlabeled tokens in the sentence;
    index is the index of the token for which feature detection
    should be performed; and history is list of the tags for all
    tokens before index.

    Construct a new classifier-based sequential tagger.

    :param feature_detector: A function used to generate the
        featureset input for the classifier::
        feature_detector(tokens, index, history) -> featureset

    :param train: A tagged corpus consisting of a list of tagged
        sentences, where each sentence is a list of (word, tag) tuples.

    :param backoff: A backoff tagger, to be used by the new tagger
        if it encounters an unknown context.

    :param classifier_builder: A function used to train a new
        classifier based on the data in *train*.  It should take
        one argument, a list of labeled featuresets (i.e.,
        (featureset, label) tuples).

    :param classifier: The classifier that should be used by the
        tagger.  This is only useful if you want to manually
        construct the classifier; normally, you would use *train*
        instead.

    :param backoff: A backoff tagger, used if this tagger is
        unable to determine a tag for a given token.

    :param cutoff_prob: If specified, then this tagger will fall
        back on its backoff tagger if the probability of the most
        likely tag is less than *cutoff_prob*.
    """
    def __init__(self, feature_detector=..., train=..., classifier_builder=..., classifier=..., backoff=..., cutoff_prob=..., verbose=...) -> None:
        ...
    
    def choose_tag(self, tokens, index, history): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def feature_detector(self, tokens, index, history):
        """
        Return the feature detector that this tagger uses to generate
        featuresets for its classifier.  The feature detector is a
        function with the signature::

          feature_detector(tokens, index, history) -> featureset

        See ``classifier()``
        """
        ...
    
    def classifier(self): # -> None:
        """
        Return the classifier that this tagger uses to choose a tag
        for each word in a sentence.  The input for this classifier is
        generated using this tagger's feature detector.
        See ``feature_detector()``
        """
        ...
    


class ClassifierBasedPOSTagger(ClassifierBasedTagger):
    """
    A classifier based part of speech tagger.
    """
    def feature_detector(self, tokens, index, history): # -> dict[str, Any]:
        ...
    


