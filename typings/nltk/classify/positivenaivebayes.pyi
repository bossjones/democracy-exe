"""
This type stub file was generated by pyright.
"""

from nltk.classify.naivebayes import NaiveBayesClassifier

"""
A variant of the Naive Bayes Classifier that performs binary classification with
partially-labeled training sets. In other words, assume we want to build a classifier
that assigns each example to one of two complementary classes (e.g., male names and
female names).
If we have a training set with labeled examples for both classes, we can use a
standard Naive Bayes Classifier. However, consider the case when we only have labeled
examples for one of the classes, and other, unlabeled, examples.
Then, assuming a prior distribution on the two labels, we can use the unlabeled set
to estimate the frequencies of the various features.

Let the two possible labels be 1 and 0, and let's say we only have examples labeled 1
and unlabeled examples. We are also given an estimate of P(1).

We compute P(feature|1) exactly as in the standard case.

To compute P(feature|0), we first estimate P(feature) from the unlabeled set (we are
assuming that the unlabeled examples are drawn according to the given prior distribution)
and then express the conditional probability as:

|                  P(feature) - P(feature|1) * P(1)
|  P(feature|0) = ----------------------------------
|                               P(0)

Example:

    >>> from nltk.classify import PositiveNaiveBayesClassifier

Some sentences about sports:

    >>> sports_sentences = [ 'The team dominated the game',
    ...                      'They lost the ball',
    ...                      'The game was intense',
    ...                      'The goalkeeper catched the ball',
    ...                      'The other team controlled the ball' ]

Mixed topics, including sports:

    >>> various_sentences = [ 'The President did not comment',
    ...                       'I lost the keys',
    ...                       'The team won the game',
    ...                       'Sara has two kids',
    ...                       'The ball went off the court',
    ...                       'They had the ball for the whole game',
    ...                       'The show is over' ]

The features of a sentence are simply the words it contains:

    >>> def features(sentence):
    ...     words = sentence.lower().split()
    ...     return dict(('contains(%s)' % w, True) for w in words)

We use the sports sentences as positive examples, the mixed ones ad unlabeled examples:

    >>> positive_featuresets = map(features, sports_sentences)
    >>> unlabeled_featuresets = map(features, various_sentences)
    >>> classifier = PositiveNaiveBayesClassifier.train(positive_featuresets,
    ...                                                 unlabeled_featuresets)

Is the following sentence about sports?

    >>> classifier.classify(features('The cat is on the table'))
    False

What about this one?

    >>> classifier.classify(features('My team lost the game'))
    True
"""
class PositiveNaiveBayesClassifier(NaiveBayesClassifier):
    @staticmethod
    def train(positive_featuresets, unlabeled_featuresets, positive_prob_prior=..., estimator=...): # -> PositiveNaiveBayesClassifier:
        """
        :param positive_featuresets: An iterable of featuresets that are known as positive
            examples (i.e., their label is ``True``).

        :param unlabeled_featuresets: An iterable of featuresets whose label is unknown.

        :param positive_prob_prior: A prior estimate of the probability of the label
            ``True`` (default 0.5).
        """
        ...
    


def demo(): # -> None:
    ...

