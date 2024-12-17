"""
This type stub file was generated by pyright.
"""

from nltk.classify import Senna

"""
Senna POS tagger, NER Tagger, Chunk Tagger

The input is:

- path to the directory that contains SENNA executables. If the path is incorrect,
  SennaTagger will automatically search for executable file specified in SENNA environment variable
- (optionally) the encoding of the input data (default:utf-8)

Note: Unit tests for this module can be found in test/unit/test_senna.py

>>> from nltk.tag import SennaTagger
>>> tagger = SennaTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> tagger.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
[('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'),
('of', 'IN'), ('an', 'DT'), ('unladen', 'NN'), ('swallow', 'NN'), ('?', '.')]

>>> from nltk.tag import SennaChunkTagger
>>> chktagger = SennaChunkTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> chktagger.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
[('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'),
('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'),
('?', 'O')]

>>> from nltk.tag import SennaNERTagger
>>> nertagger = SennaNERTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
>>> nertagger.tag('Shakespeare theatre was in London .'.split()) # doctest: +SKIP
[('Shakespeare', 'B-PER'), ('theatre', 'O'), ('was', 'O'), ('in', 'O'),
('London', 'B-LOC'), ('.', 'O')]
>>> nertagger.tag('UN headquarters are in NY , USA .'.split()) # doctest: +SKIP
[('UN', 'B-ORG'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'),
('NY', 'B-LOC'), (',', 'O'), ('USA', 'B-LOC'), ('.', 'O')]
"""
class SennaTagger(Senna):
    def __init__(self, path, encoding=...) -> None:
        ...
    
    def tag_sents(self, sentences): # -> list[list[Any]]:
        """
        Applies the tag method over a list of sentences. This method will return
        for each sentence a list of tuples of (word, tag).
        """
        ...
    


class SennaChunkTagger(Senna):
    def __init__(self, path, encoding=...) -> None:
        ...
    
    def tag_sents(self, sentences): # -> list[list[Any]]:
        """
        Applies the tag method over a list of sentences. This method will return
        for each sentence a list of tuples of (word, tag).
        """
        ...
    
    def bio_to_chunks(self, tagged_sent, chunk_type): # -> Generator[tuple[LiteralString, str], Any, None]:
        """
        Extracts the chunks in a BIO chunk-tagged sentence.

        >>> from nltk.tag import SennaChunkTagger
        >>> chktagger = SennaChunkTagger('/usr/share/senna-v3.0')  # doctest: +SKIP
        >>> sent = 'What is the airspeed of an unladen swallow ?'.split()
        >>> tagged_sent = chktagger.tag(sent)  # doctest: +SKIP
        >>> tagged_sent  # doctest: +SKIP
        [('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'),
        ('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'),
        ('?', 'O')]
        >>> list(chktagger.bio_to_chunks(tagged_sent, chunk_type='NP'))  # doctest: +SKIP
        [('What', '0'), ('the airspeed', '2-3'), ('an unladen swallow', '5-6-7')]

        :param tagged_sent: A list of tuples of word and BIO chunk tag.
        :type tagged_sent: list(tuple)
        :param tagged_sent: The chunk tag that users want to extract, e.g. 'NP' or 'VP'
        :type tagged_sent: str

        :return: An iterable of tuples of chunks that users want to extract
          and their corresponding indices.
        :rtype: iter(tuple(str))
        """
        ...
    


class SennaNERTagger(Senna):
    def __init__(self, path, encoding=...) -> None:
        ...
    
    def tag_sents(self, sentences): # -> list[list[Any]]:
        """
        Applies the tag method over a list of sentences. This method will return
        for each sentence a list of tuples of (word, tag).
        """
        ...
    


