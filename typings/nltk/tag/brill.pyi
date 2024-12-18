"""
This type stub file was generated by pyright.
"""

from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature

@jsontags.register_tag
class Word(Feature):
    """
    Feature which examines the text (word) of nearby tokens.
    """
    json_tag = ...
    @staticmethod
    def extract_property(tokens, index):
        """@return: The given token's text."""
        ...
    


@jsontags.register_tag
class Pos(Feature):
    """
    Feature which examines the tags of nearby tokens.
    """
    json_tag = ...
    @staticmethod
    def extract_property(tokens, index):
        """@return: The given token's tag."""
        ...
    


def nltkdemo18(): # -> list[Template]:
    """
    Return 18 templates, from the original nltk demo, in multi-feature syntax
    """
    ...

def nltkdemo18plus(): # -> list[Template]:
    """
    Return 18 templates, from the original nltk demo, and additionally a few
    multi-feature ones (the motivation is easy comparison with nltkdemo18)
    """
    ...

def fntbl37(): # -> list[Template]:
    """
    Return 37 templates taken from the postagging task of the
    fntbl distribution https://www.cs.jhu.edu/~rflorian/fntbl/
    (37 is after excluding a handful which do not condition on Pos[0];
    fntbl can do that but the current nltk implementation cannot.)
    """
    ...

def brill24(): # -> list[Template]:
    """
    Return 24 templates of the seminal TBL paper, Brill (1995)
    """
    ...

def describe_template_sets(): # -> None:
    """
    Print the available template sets in this demo, with a short description"
    """
    ...

@jsontags.register_tag
class BrillTagger(TaggerI):
    """
    Brill's transformational rule-based tagger.  Brill taggers use an
    initial tagger (such as ``tag.DefaultTagger``) to assign an initial
    tag sequence to a text; and then apply an ordered list of
    transformational rules to correct the tags of individual tokens.
    These transformation rules are specified by the ``TagRule``
    interface.

    Brill taggers can be created directly, from an initial tagger and
    a list of transformational rules; but more often, Brill taggers
    are created by learning rules from a training corpus, using one
    of the TaggerTrainers available.
    """
    json_tag = ...
    def __init__(self, initial_tagger, rules, training_stats=...) -> None:
        """
        :param initial_tagger: The initial tagger
        :type initial_tagger: TaggerI

        :param rules: An ordered list of transformation rules that
            should be used to correct the initial tagging.
        :type rules: list(TagRule)

        :param training_stats: A dictionary of statistics collected
            during training, for possible later use
        :type training_stats: dict

        """
        ...
    
    def encode_json_obj(self): # -> tuple[Any, tuple[Any, ...], Any | None]:
        ...
    
    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...
    
    def rules(self): # -> tuple[Any, ...]:
        """
        Return the ordered list of  transformation rules that this tagger has learnt

        :return: the ordered list of transformation rules that correct the initial tagging
        :rtype: list of Rules
        """
        ...
    
    def train_stats(self, statistic=...): # -> None:
        """
        Return a named statistic collected during training, or a dictionary of all
        available statistics if no name given

        :param statistic: name of statistic
        :type statistic: str
        :return: some statistic collected during training of this tagger
        :rtype: any (but usually a number)
        """
        ...
    
    def tag(self, tokens):
        ...
    
    def print_template_statistics(self, test_stats=..., printunused=...): # -> None:
        """
        Print a list of all templates, ranked according to efficiency.

        If test_stats is available, the templates are ranked according to their
        relative contribution (summed for all rules created from a given template,
        weighted by score) to the performance on the test set. If no test_stats, then
        statistics collected during training are used instead. There is also
        an unweighted measure (just counting the rules). This is less informative,
        though, as many low-score rules will appear towards end of training.

        :param test_stats: dictionary of statistics collected during testing
        :type test_stats: dict of str -> any (but usually numbers)
        :param printunused: if True, print a list of all unused templates
        :type printunused: bool
        :return: None
        :rtype: None
        """
        ...
    
    def batch_tag_incremental(self, sequences, gold): # -> tuple[list[Any], dict[Any, Any]]:
        """
        Tags by applying each rule to the entire corpus (rather than all rules to a
        single sequence). The point is to collect statistics on the test set for
        individual rules.

        NOTE: This is inefficient (does not build any index, so will traverse the entire
        corpus N times for N rules) -- usually you would not care about statistics for
        individual rules and thus use batch_tag() instead

        :param sequences: lists of token sequences (sentences, in some applications) to be tagged
        :type sequences: list of list of strings
        :param gold: the gold standard
        :type gold: list of list of strings
        :returns: tuple of (tagged_sequences, ordered list of rule scores (one for each rule))
        """
        ...
    


