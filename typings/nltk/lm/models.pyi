"""
This type stub file was generated by pyright.
"""

from nltk.lm.api import LanguageModel

"""Language Models"""
class MLE(LanguageModel):
    """Class for providing MLE ngram model scores.

    Inherits initialization from BaseNgramModel.
    """
    def unmasked_score(self, word, context=...): # -> float | Literal[0]:
        """Returns the MLE score for a word given a context.

        Args:
        - word is expected to be a string
        - context is expected to be something reasonably convertible to a tuple
        """
        ...
    


class Lidstone(LanguageModel):
    """Provides Lidstone-smoothed scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """
    def __init__(self, gamma, *args, **kwargs) -> None:
        ...
    
    def unmasked_score(self, word, context=...): # -> float:
        """Add-one smoothing: Lidstone or Laplace.

        To see what kind, look at `gamma` attribute on the class.

        """
        ...
    


class Laplace(Lidstone):
    """Implements Laplace (add one) smoothing.

    Initialization identical to BaseNgramModel because gamma is always 1.
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class StupidBackoff(LanguageModel):
    """Provides StupidBackoff scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a parameter alpha with which we scale the lower order probabilities.
    Note that this is not a true probability distribution as scores for ngrams
    of the same order do not sum up to unity.
    """
    def __init__(self, alpha=..., *args, **kwargs) -> None:
        ...
    
    def unmasked_score(self, word, context=...): # -> float | Literal[0]:
        ...
    


class InterpolatedLanguageModel(LanguageModel):
    """Logic common to all interpolated language models.

    The idea to abstract this comes from Chen & Goodman 1995.
    Do not instantiate this class directly!
    """
    def __init__(self, smoothing_cls, order, **kwargs) -> None:
        ...
    
    def unmasked_score(self, word, context=...):
        ...
    


class WittenBellInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Witten-Bell smoothing."""
    def __init__(self, order, **kwargs) -> None:
        ...
    


class AbsoluteDiscountingInterpolated(InterpolatedLanguageModel):
    """Interpolated version of smoothing with absolute discount."""
    def __init__(self, order, discount=..., **kwargs) -> None:
        ...
    


class KneserNeyInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Kneser-Ney smoothing."""
    def __init__(self, order, discount=..., **kwargs) -> None:
        ...
    


