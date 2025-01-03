"""
This type stub file was generated by pyright.
"""

import re
from nltk.corpus.reader import *
from nltk.corpus.util import LazyCorpusLoader
from nltk.tokenize import RegexpTokenizer

"""
NLTK corpus readers.  The modules in this package provide functions
that can be used to read corpus files in a variety of formats.  These
functions can be used to read both the corpus files that are
distributed in the NLTK corpus package, and corpus files that are part
of external corpora.

Available Corpora
=================

Please see https://www.nltk.org/nltk_data/ for a complete list.
Install corpora using nltk.download().

Corpus Reader Functions
=======================
Each corpus module defines one or more "corpus reader functions",
which can be used to read documents from that corpus.  These functions
take an argument, ``item``, which is used to indicate which document
should be read from the corpus:

- If ``item`` is one of the unique identifiers listed in the corpus
  module's ``items`` variable, then the corresponding document will
  be loaded from the NLTK corpus package.
- If ``item`` is a filename, then that file will be read.

Additionally, corpus reader functions can be given lists of item
names; in which case, they will return a concatenation of the
corresponding documents.

Corpus reader functions are named based on the type of information
they return.  Some common examples, and their return types, are:

- words(): list of str
- sents(): list of (list of str)
- paras(): list of (list of (list of str))
- tagged_words(): list of (str,str) tuple
- tagged_sents(): list of (list of (str,str))
- tagged_paras(): list of (list of (list of (str,str)))
- chunked_sents(): list of (Tree w/ (str,str) leaves)
- parsed_sents(): list of (Tree with str leaves)
- parsed_paras(): list of (list of (Tree with str leaves))
- xml(): A single xml ElementTree
- raw(): unprocessed corpus contents

For example, to read a list of the words in the Brown Corpus, use
``nltk.corpus.brown.words()``:

    >>> from nltk.corpus import brown
    >>> print(", ".join(brown.words())) # doctest: +ELLIPSIS
    The, Fulton, County, Grand, Jury, said, ...

"""
abc: PlaintextCorpusReader = ...
alpino: AlpinoCorpusReader = ...
bcp47: BCP47CorpusReader = ...
brown: CategorizedTaggedCorpusReader = ...
cess_cat: BracketParseCorpusReader = ...
cess_esp: BracketParseCorpusReader = ...
cmudict: CMUDictCorpusReader = ...
comtrans: AlignedCorpusReader = ...
comparative_sentences: ComparativeSentencesCorpusReader = ...
conll2000: ConllChunkCorpusReader = ...
conll2002: ConllChunkCorpusReader = ...
conll2007: DependencyCorpusReader = ...
crubadan: CrubadanCorpusReader = ...
dependency_treebank: DependencyCorpusReader = ...
extended_omw: CorpusReader = ...
floresta: BracketParseCorpusReader = ...
framenet15: FramenetCorpusReader = ...
framenet: FramenetCorpusReader = ...
gazetteers: WordListCorpusReader = ...
genesis: PlaintextCorpusReader = ...
gutenberg: PlaintextCorpusReader = ...
ieer: IEERCorpusReader = ...
inaugural: PlaintextCorpusReader = ...
indian: IndianCorpusReader = ...
jeita: ChasenCorpusReader = ...
knbc: KNBCorpusReader = ...
lin_thesaurus: LinThesaurusCorpusReader = ...
mac_morpho: MacMorphoCorpusReader = ...
machado: PortugueseCategorizedPlaintextCorpusReader = ...
masc_tagged: CategorizedTaggedCorpusReader = ...
movie_reviews: CategorizedPlaintextCorpusReader = ...
multext_east: MTECorpusReader = ...
names: WordListCorpusReader = ...
nps_chat: NPSChatCorpusReader = ...
opinion_lexicon: OpinionLexiconCorpusReader = ...
ppattach: PPAttachmentCorpusReader = ...
product_reviews_1: ReviewsCorpusReader = ...
product_reviews_2: ReviewsCorpusReader = ...
pros_cons: ProsConsCorpusReader = ...
ptb: CategorizedBracketParseCorpusReader = ...
qc: StringCategoryCorpusReader = ...
reuters: CategorizedPlaintextCorpusReader = ...
rte: RTECorpusReader = ...
senseval: SensevalCorpusReader = ...
sentence_polarity: CategorizedSentencesCorpusReader = ...
sentiwordnet: SentiWordNetCorpusReader = ...
shakespeare: XMLCorpusReader = ...
sinica_treebank: SinicaTreebankCorpusReader = ...
state_union: PlaintextCorpusReader = ...
stopwords: WordListCorpusReader = ...
subjectivity: CategorizedSentencesCorpusReader = ...
swadesh: SwadeshCorpusReader = ...
swadesh110: PanlexSwadeshCorpusReader = ...
swadesh207: PanlexSwadeshCorpusReader = ...
switchboard: SwitchboardCorpusReader = ...
timit: TimitCorpusReader = ...
timit_tagged: TimitTaggedCorpusReader = ...
toolbox: ToolboxCorpusReader = ...
treebank: BracketParseCorpusReader = ...
treebank_chunk: ChunkedCorpusReader = ...
treebank_raw: PlaintextCorpusReader = ...
twitter_samples: TwitterCorpusReader = ...
udhr: UdhrCorpusReader = ...
udhr2: PlaintextCorpusReader = ...
universal_treebanks: ConllCorpusReader = ...
verbnet: VerbnetCorpusReader = ...
webtext: PlaintextCorpusReader = ...
wordnet: WordNetCorpusReader = ...
wordnet31: WordNetCorpusReader = ...
wordnet2021: WordNetCorpusReader = ...
wordnet2022: WordNetCorpusReader = ...
wordnet_ic: WordNetICCorpusReader = ...
words: WordListCorpusReader = ...
propbank: PropbankCorpusReader = ...
nombank: NombankCorpusReader = ...
propbank_ptb: PropbankCorpusReader = ...
nombank_ptb: NombankCorpusReader = ...
semcor: SemcorCorpusReader = ...
nonbreaking_prefixes: NonbreakingPrefixesCorpusReader = ...
perluniprops: UnicharsCorpusReader = ...
def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
