"""
This type stub file was generated by pyright.
"""

from nltk.tag.api import TaggerI

"""
A general interface to the SENNA pipeline that supports any of the
operations specified in SUPPORTED_OPERATIONS.

Applying multiple operations at once has the speed advantage. For example,
Senna will automatically determine POS tags if you are extracting named
entities. Applying both of the operations will cost only the time of
extracting the named entities.

The SENNA pipeline has a fixed maximum size of the sentences that it can read.
By default it is 1024 token/sentence. If you have larger sentences, changing
the MAX_SENTENCE_SIZE value in SENNA_main.c should be considered and your
system specific binary should be rebuilt. Otherwise this could introduce
misalignment errors.

The input is:

- path to the directory that contains SENNA executables. If the path is incorrect,
  Senna will automatically search for executable file specified in SENNA environment variable
- List of the operations needed to be performed.
- (optionally) the encoding of the input data (default:utf-8)

Note: Unit tests for this module can be found in test/unit/test_senna.py

>>> from nltk.classify import Senna
>>> pipeline = Senna('/usr/share/senna-v3.0', ['pos', 'chk', 'ner'])  # doctest: +SKIP
>>> sent = 'Dusseldorf is an international business center'.split()
>>> [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(sent)]  # doctest: +SKIP
[('Dusseldorf', 'B-NP', 'B-LOC', 'NNP'), ('is', 'B-VP', 'O', 'VBZ'), ('an', 'B-NP', 'O', 'DT'),
('international', 'I-NP', 'O', 'JJ'), ('business', 'I-NP', 'O', 'NN'), ('center', 'I-NP', 'O', 'NN')]
"""
class Senna(TaggerI):
    SUPPORTED_OPERATIONS = ...
    def __init__(self, senna_path, operations, encoding=...) -> None:
        ...
    
    def executable(self, base_path): # -> str:
        """
        The function that determines the system specific binary that should be
        used in the pipeline. In case, the system is not known the default senna binary will
        be used.
        """
        ...
    
    def tag(self, tokens): # -> list[Any]:
        """
        Applies the specified operation(s) on a list of tokens.
        """
        ...
    
    def tag_sents(self, sentences): # -> list[list[Any]]:
        """
        Applies the tag method over a list of sentences. This method will return a
        list of dictionaries. Every dictionary will contain a word with its
        calculated annotations/tags.
        """
        ...
    


