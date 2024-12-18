"""
This type stub file was generated by pyright.
"""

from functools import total_ordering
from nltk.corpus.reader import CorpusReader

"""
An NLTK interface for WordNet

WordNet is a lexical database of English.
Using synsets, helps find conceptual relationships between words
such as hypernyms, hyponyms, synonyms, antonyms etc.

For details about WordNet see:
https://wordnet.princeton.edu/

This module also allows you to find lemmas in languages
other than English from the Open Multilingual Wordnet
https://omwn.org/

"""
_INF = ...
POS_LIST = ...
VERB_FRAME_STRINGS = ...
SENSENUM_RE = ...
class WordNetError(Exception):
    """An exception class for wordnet-related errors."""
    ...


@total_ordering
class _WordNetObject:
    """A common base class for lemmas and synsets."""
    def hypernyms(self):
        ...
    
    def instance_hypernyms(self):
        ...
    
    def hyponyms(self):
        ...
    
    def instance_hyponyms(self):
        ...
    
    def member_holonyms(self):
        ...
    
    def substance_holonyms(self):
        ...
    
    def part_holonyms(self):
        ...
    
    def member_meronyms(self):
        ...
    
    def substance_meronyms(self):
        ...
    
    def part_meronyms(self):
        ...
    
    def topic_domains(self):
        ...
    
    def in_topic_domains(self):
        ...
    
    def region_domains(self):
        ...
    
    def in_region_domains(self):
        ...
    
    def usage_domains(self):
        ...
    
    def in_usage_domains(self):
        ...
    
    def attributes(self):
        ...
    
    def entailments(self):
        ...
    
    def causes(self):
        ...
    
    def also_sees(self):
        ...
    
    def verb_groups(self):
        ...
    
    def similar_tos(self):
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __ne__(self, other) -> bool:
        ...
    
    def __lt__(self, other) -> bool:
        ...
    


class Lemma(_WordNetObject):
    """
    The lexical entry for a single morphological form of a
    sense-disambiguated word.

    Create a Lemma from a "<word>.<pos>.<number>.<lemma>" string where:
    <word> is the morphological stem identifying the synset
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.
    <lemma> is the morphological form of interest

    Note that <word> and <lemma> can be different, e.g. the Synset
    'salt.n.03' has the Lemmas 'salt.n.03.salt', 'salt.n.03.saltiness' and
    'salt.n.03.salinity'.

    Lemma attributes, accessible via methods with the same name:

    - name: The canonical name of this lemma.
    - synset: The synset that this lemma belongs to.
    - syntactic_marker: For adjectives, the WordNet string identifying the
      syntactic position relative modified noun. See:
      https://wordnet.princeton.edu/documentation/wninput5wn
      For all other parts of speech, this attribute is None.
    - count: The frequency of this lemma in wordnet.

    Lemma methods:

    Lemmas have the following methods for retrieving related Lemmas. They
    correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Lemmas:

    - antonyms
    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - topic_domains, region_domains, usage_domains
    - attributes
    - derivationally_related_forms
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos
    - pertainyms
    """
    __slots__ = ...
    def __init__(self, wordnet_corpus_reader, synset, name, lexname_index, lex_id, syntactic_marker) -> None:
        ...
    
    def name(self): # -> Any:
        ...
    
    def syntactic_marker(self): # -> Any:
        ...
    
    def synset(self): # -> Any:
        ...
    
    def frame_strings(self): # -> list[Any]:
        ...
    
    def frame_ids(self): # -> list[Any]:
        ...
    
    def lang(self): # -> str:
        ...
    
    def key(self): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def count(self):
        """Return the frequency count for this Lemma"""
        ...
    
    def antonyms(self): # -> list[Any]:
        ...
    
    def derivationally_related_forms(self): # -> list[Any]:
        ...
    
    def pertainyms(self): # -> list[Any]:
        ...
    


class Synset(_WordNetObject):
    """Create a Synset from a "<lemma>.<pos>.<number>" string where:
    <lemma> is the word's morphological stem
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.

    Synset attributes, accessible via methods with the same name:

    - name: The canonical name of this synset, formed using the first lemma
      of this synset. Note that this may be different from the name
      passed to the constructor if that string used a different lemma to
      identify the synset.
    - pos: The synset's part of speech, matching one of the module level
      attributes ADJ, ADJ_SAT, ADV, NOUN or VERB.
    - lemmas: A list of the Lemma objects for this synset.
    - definition: The definition for this synset.
    - examples: A list of example strings for this synset.
    - offset: The offset in the WordNet dict file of this synset.
    - lexname: The name of the lexicographer file containing this synset.

    Synset methods:

    Synsets have the following methods for retrieving related Synsets.
    They correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Synsets.

    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - attributes
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos

    Additionally, Synsets support the following methods specific to the
    hypernym relation:

    - root_hypernyms
    - common_hypernyms
    - lowest_common_hypernyms

    Note that Synsets do not support the following relations because
    these are defined by WordNet as lexical relations:

    - antonyms
    - derivationally_related_forms
    - pertainyms
    """
    __slots__ = ...
    def __init__(self, wordnet_corpus_reader) -> None:
        ...
    
    def pos(self): # -> None:
        ...
    
    def offset(self): # -> None:
        ...
    
    def name(self): # -> None:
        ...
    
    def frame_ids(self): # -> list[Any]:
        ...
    
    def definition(self, lang=...): # -> None:
        """Return definition in specified language"""
        ...
    
    def examples(self, lang=...): # -> None:
        """Return examples in specified language"""
        ...
    
    def lexname(self): # -> None:
        ...
    
    def lemma_names(self, lang=...): # -> list[Any]:
        """Return all the lemma_names associated with the synset"""
        ...
    
    def lemmas(self, lang=...): # -> list[Any] | None:
        """Return all the lemma objects associated with the synset"""
        ...
    
    def root_hypernyms(self): # -> list[Any]:
        """Get the topmost hypernyms of this synset in WordNet."""
        ...
    
    def max_depth(self): # -> int:
        """
        :return: The length of the longest hypernym path from this
            synset to the root.
        """
        ...
    
    def min_depth(self): # -> int:
        """
        :return: The length of the shortest hypernym path from this
            synset to the root.
        """
        ...
    
    def closure(self, rel, depth=...): # -> Generator[Any, Any, None]:
        """
        Return the transitive closure of source under the rel
        relationship, breadth-first, discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:s.topic_domains()
        >>> print(list(computer.closure(topic)))
        [Synset('computer_science.n.01')]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth 2


        Include redundant paths (but only once), avoiding duplicate searches
        (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:sorted(s.hypernyms())
        >>> print(list(dog.closure(hyp)))
        [Synset('canine.n.02'), Synset('domestic_animal.n.01'), Synset('carnivore.n.01'),\
 Synset('animal.n.01'), Synset('placental.n.01'), Synset('organism.n.01'),\
 Synset('mammal.n.01'), Synset('living_thing.n.01'), Synset('vertebrate.n.01'),\
 Synset('whole.n.02'), Synset('chordate.n.01'), Synset('object.n.01'),\
 Synset('physical_entity.n.01'), Synset('entity.n.01')]

        UserWarning: Discarded redundant search for Synset('animal.n.01') at depth 7
        """
        ...
    
    def tree(self, rel, depth=..., cut_mark=...): # -> list[Any]:
        """
        Return the full relation tree, including self,
        discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> from pprint import pprint
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:sorted(s.topic_domains())
        >>> pprint(computer.tree(topic))
        [Synset('computer.n.01'), [Synset('computer_science.n.01')]]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth -3


        But keep duplicate branches (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:sorted(s.hypernyms())
        >>> pprint(dog.tree(hyp))
        [Synset('dog.n.01'),
         [Synset('canine.n.02'),
          [Synset('carnivore.n.01'),
           [Synset('placental.n.01'),
            [Synset('mammal.n.01'),
             [Synset('vertebrate.n.01'),
              [Synset('chordate.n.01'),
               [Synset('animal.n.01'),
                [Synset('organism.n.01'),
                 [Synset('living_thing.n.01'),
                  [Synset('whole.n.02'),
                   [Synset('object.n.01'),
                    [Synset('physical_entity.n.01'),
                     [Synset('entity.n.01')]]]]]]]]]]]]],
         [Synset('domestic_animal.n.01'),
          [Synset('animal.n.01'),
           [Synset('organism.n.01'),
            [Synset('living_thing.n.01'),
             [Synset('whole.n.02'),
              [Synset('object.n.01'),
               [Synset('physical_entity.n.01'), [Synset('entity.n.01')]]]]]]]]]
        """
        ...
    
    def hypernym_paths(self): # -> list[list[Self]]:
        """
        Get the path(s) from this synset to the root, where each path is a
        list of the synset nodes traversed on the way to the root.

        :return: A list of lists, where each list gives the node sequence
           connecting the initial ``Synset`` node and a root node.
        """
        ...
    
    def common_hypernyms(self, other): # -> list[Self | Any]:
        """
        Find all synsets that are hypernyms of this synset and the
        other synset.

        :type other: Synset
        :param other: other input synset.
        :return: The synsets that are hypernyms of both synsets.
        """
        ...
    
    def lowest_common_hypernyms(self, other, simulate_root=..., use_min_depth=...): # -> list[Self | Any] | list[Any]:
        """
        Get a list of lowest synset(s) that both synsets have as a hypernym.
        When `use_min_depth == False` this means that the synset which appears
        as a hypernym of both `self` and `other` with the lowest maximum depth
        is returned or if there are multiple such synsets at the same depth
        they are all returned

        However, if `use_min_depth == True` then the synset(s) which has/have
        the lowest minimum depth and appear(s) in both paths is/are returned.

        By setting the use_min_depth flag to True, the behavior of NLTK2 can be
        preserved. This was changed in NLTK3 to give more accurate results in a
        small set of cases, generally with synsets concerning people. (eg:
        'chef.n.01', 'fireman.n.01', etc.)

        This method is an implementation of Ted Pedersen's "Lowest Common
        Subsumer" method from the Perl Wordnet module. It can return either
        "self" or "other" if they are a hypernym of the other.

        :type other: Synset
        :param other: other input synset
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (False by default)
            creates a fake root that connects all the taxonomies. Set it
            to True to enable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will need to be added
            for nouns as well.
        :type use_min_depth: bool
        :param use_min_depth: This setting mimics older (v2) behavior of NLTK
            wordnet If True, will use the min_depth function to calculate the
            lowest common hypernyms. This is known to give strange results for
            some synset pairs (eg: 'chef.n.01', 'fireman.n.01') but is retained
            for backwards compatibility
        :return: The synsets that are the lowest common hypernyms of both
            synsets
        """
        ...
    
    def hypernym_distances(self, distance=..., simulate_root=...): # -> set[tuple[Self, int]]:
        """
        Get the path(s) from this synset to the root, counting the distance
        of each node from the initial node on the way. A set of
        (synset, distance) tuples is returned.

        :type distance: int
        :param distance: the distance (number of edges) from this hypernym to
            the original hypernym ``Synset`` on which this method was called.
        :return: A set of ``(Synset, int)`` tuples where each ``Synset`` is
           a hypernym of the first ``Synset``.
        """
        ...
    
    def shortest_path_distance(self, other, simulate_root=...): # -> float | Literal[0] | None:
        """
        Returns the distance of the shortest path linking the two synsets (if
        one exists). For each synset, all the ancestor nodes and their
        distances are recorded and compared. The ancestor node common to both
        synsets that can be reached with the minimum number of traversals is
        used. If no ancestor nodes are common, None is returned. If a node is
        compared with itself 0 is returned.

        :type other: Synset
        :param other: The Synset to which the shortest path will be found.
        :return: The number of edges in the shortest path connecting the two
            nodes, or None if no path exists.
        """
        ...
    
    def path_similarity(self, other, verbose=..., simulate_root=...): # -> float | None:
        """
        Path Distance Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses in the is-a (hypernym/hypnoym)
        taxonomy. The score is in the range 0 to 1, except in those cases where
        a path cannot be found (will only be true for verbs as there are many
        distinct verb taxonomies), in which case None is returned. A score of
        1 represents identity i.e. comparing a sense with itself will return 1.

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally between 0 and 1. None is returned if no connecting path
            could be found. 1 is returned if a ``Synset`` is compared with
            itself.
        """
        ...
    
    def lch_similarity(self, other, verbose=..., simulate_root=...): # -> float | None:
        """
        Leacock Chodorow Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses (as above) and the maximum depth
        of the taxonomy in which the senses occur. The relationship is given as
        -log(p/2d) where p is the shortest path length and d is the taxonomy
        depth.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally greater than 0. None is returned if no connecting path
            could be found. If a ``Synset`` is compared with itself, the
            maximum score is returned, which varies depending on the taxonomy
            depth.
        """
        ...
    
    def wup_similarity(self, other, verbose=..., simulate_root=...): # -> None:
        """
        Wu-Palmer Similarity:
        Return a score denoting how similar two word senses are, based on the
        depth of the two senses in the taxonomy and that of their Least Common
        Subsumer (most specific ancestor node). Previously, the scores computed
        by this implementation did _not_ always agree with those given by
        Pedersen's Perl implementation of WordNet Similarity. However, with
        the addition of the simulate_root flag (see below), the score for
        verbs now almost always agree but not always for nouns.

        The LCS does not necessarily feature in the shortest path connecting
        the two senses, as it is by definition the common ancestor deepest in
        the taxonomy, not closest to the two senses. Typically, however, it
        will so feature. Where multiple candidates for the LCS exist, that
        whose shortest path to the root node is the longest will be selected.
        Where the LCS has multiple paths to the root, the longer path is used
        for the purposes of the calculation.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A float score denoting the similarity of the two ``Synset``
            objects, normally greater than zero. If no connecting path between
            the two senses can be found, None is returned.

        """
        ...
    
    def res_similarity(self, other, ic, verbose=...): # -> float | Literal[0]:
        """
        Resnik Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects. Synsets whose LCS is the root node of the taxonomy will
            have a score of 0 (e.g. N['dog'][0] and N['table'][0]).
        """
        ...
    
    def jcn_similarity(self, other, ic, verbose=...): # -> float | Literal[0]:
        """
        Jiang-Conrath Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs)).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type  ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects.
        """
        ...
    
    def lin_similarity(self, other, ic, verbose=...): # -> float:
        """
        Lin Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 2 * IC(lcs) / (IC(s1) + IC(s2)).

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects, in the range 0 to 1.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


class WordNetCorpusReader(CorpusReader):
    """
    A corpus reader used to access wordnet or its variants.
    """
    _ENCODING = ...
    _FILEMAP = ...
    _pos_numbers = ...
    _pos_names = ...
    _FILES = ...
    def __init__(self, root, omw_reader) -> None:
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """
        ...
    
    def index_sense(self, version=...): # -> dict[Any, Any]:
        """Read sense key to synset id mapping from index.sense file in corpus directory"""
        ...
    
    def map_to_many(self, version=...): # -> dict[Any, Any]:
        ...
    
    def map_to_one(self, version=...): # -> dict[Any, Any]:
        ...
    
    def map_wn(self, version=...): # -> dict[Any, Any] | None:
        """Mapping from Wordnet 'version' to currently loaded Wordnet version"""
        ...
    
    def split_synsets(self, version=...):
        ...
    
    def merged_synsets(self, version=...):
        ...
    
    def of2ss(self, of): # -> None:
        """take an id and return the synsets"""
        ...
    
    def ss2of(self, ss): # -> str | None:
        """return the ID of the synset"""
        ...
    
    def add_provs(self, reader): # -> None:
        """Add languages from Multilingual Wordnet to the provenance dictionary"""
        ...
    
    def add_omw(self): # -> None:
        ...
    
    def add_exomw(self): # -> None:
        """
        Add languages from Extended OMW

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> wn.add_exomw()
        >>> print(wn.synset('intrinsically.r.01').lemmas(lang="eng_wikt"))
        [Lemma('intrinsically.r.01.per_se'), Lemma('intrinsically.r.01.as_such')]
        """
        ...
    
    def langs(self): # -> list[Any]:
        """return a list of languages supported by Multilingual Wordnet"""
        ...
    
    def get_version(self): # -> str | Any | None:
        ...
    
    def lemma(self, name, lang=...):
        """Return lemma object that matches the name"""
        ...
    
    def lemma_from_key(self, key):
        ...
    
    def synset(self, name): # -> None:
        ...
    
    def synset_from_pos_and_offset(self, pos, offset): # -> None:
        """
        - pos: The synset's part of speech, matching one of the module level
          attributes ADJ, ADJ_SAT, ADV, NOUN or VERB ('a', 's', 'r', 'n', or 'v').
        - offset: The byte offset of this synset in the WordNet dict file
          for this pos.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_pos_and_offset('n', 1740))
        Synset('entity.n.01')
        """
        ...
    
    def synset_from_sense_key(self, sense_key):
        """
        Retrieves synset based on a given sense_key. Sense keys can be
        obtained from lemma.key()

        From https://wordnet.princeton.edu/documentation/senseidx5wn:
        A sense_key is represented as::

            lemma % lex_sense (e.g. 'dog%1:18:01::')

        where lex_sense is encoded as::

            ss_type:lex_filenum:lex_id:head_word:head_id

        :lemma:       ASCII text of word/collocation, in lower case
        :ss_type:     synset type for the sense (1 digit int)
                      The synset type is encoded as follows::

                          1    NOUN
                          2    VERB
                          3    ADJECTIVE
                          4    ADVERB
                          5    ADJECTIVE SATELLITE
        :lex_filenum: name of lexicographer file containing the synset for the sense (2 digit int)
        :lex_id:      when paired with lemma, uniquely identifies a sense in the lexicographer file (2 digit int)
        :head_word:   lemma of the first word in satellite's head synset
                      Only used if sense is in an adjective satellite synset
        :head_id:     uniquely identifies sense in a lexicographer file when paired with head_word
                      Only used if head_word is present (2 digit int)

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_sense_key("drive%1:04:03::"))
        Synset('drive.n.06')

        >>> print(wn.synset_from_sense_key("driving%1:04:03::"))
        Synset('drive.n.06')
        """
        ...
    
    def synsets(self, lemma, pos=..., lang=..., check_exceptions=...): # -> list[Any | None] | list[Any]:
        """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        If lang is specified, all the synsets associated with the lemma name
        of that language will be returned.
        """
        ...
    
    def lemmas(self, lemma, pos=..., lang=...): # -> list[Any]:
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""
        ...
    
    def all_lemma_names(self, pos=..., lang=...): # -> Iterator[Any] | Generator[Any, None, None]:
        """Return all lemma names for all synsets for the given
        part of speech tag and language or languages. If pos is
        not specified, all synsets for all parts of speech will
        be used."""
        ...
    
    def all_omw_synsets(self, pos=..., lang=...): # -> Generator[Any, Any, None]:
        ...
    
    def all_synsets(self, pos=..., lang=...): # -> Generator[Any, Any, None]:
        """Iterate over all synsets with a given part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        ...
    
    def all_eng_synsets(self, pos=...): # -> Generator[Any, Any, None]:
        ...
    
    def words(self, lang=...): # -> Iterator[Any] | Generator[Any, None, None]:
        """return lemmas of the given language as list of words"""
        ...
    
    def synonyms(self, word, lang=...): # -> list[list[Any]]:
        """return nested list with the synonyms of the different senses of word in the given language"""
        ...
    
    def doc(self, file=..., lang=...): # -> bytes | str:
        """Return the contents of readme, license or citation file
        use lang=lang to get the file for an individual language"""
        ...
    
    def license(self, lang=...): # -> bytes | str:
        """Return the contents of LICENSE (for omw)
        use lang=lang to get the license for an individual language"""
        ...
    
    def readme(self, lang=...): # -> bytes | str:
        """Return the contents of README (for omw)
        use lang=lang to get the readme for an individual language"""
        ...
    
    def citation(self, lang=...): # -> bytes | str:
        """Return the contents of citation.bib file (for omw)
        use lang=lang to get the citation for an individual language"""
        ...
    
    def lemma_count(self, lemma): # -> int:
        """Return the frequency count for this Lemma"""
        ...
    
    def path_similarity(self, synset1, synset2, verbose=..., simulate_root=...):
        ...
    
    def lch_similarity(self, synset1, synset2, verbose=..., simulate_root=...):
        ...
    
    def wup_similarity(self, synset1, synset2, verbose=..., simulate_root=...):
        ...
    
    def res_similarity(self, synset1, synset2, ic, verbose=...):
        ...
    
    def jcn_similarity(self, synset1, synset2, ic, verbose=...):
        ...
    
    def lin_similarity(self, synset1, synset2, ic, verbose=...):
        ...
    
    def morphy(self, form, pos=..., check_exceptions=...): # -> None:
        """
        Find a possible base form for the given form, with the given
        part of speech, by checking WordNet's list of exceptional
        forms, or by substituting suffixes for this part of speech.
        If pos=None, try every part of speech until finding lemmas.
        Return the first form found in WordNet, or eventually None.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        >>> print(wn.morphy('abaci'))
        abacus
        >>> wn.morphy('hardrock', wn.ADV)
        >>> print(wn.morphy('book', wn.NOUN))
        book
        >>> wn.morphy('book', wn.ADJ)
        """
        ...
    
    MORPHOLOGICAL_SUBSTITUTIONS = ...
    def ic(self, corpus, weight_senses_equally=..., smoothing=...): # -> dict[Any, Any]:
        """
        Creates an information content lookup dictionary from a corpus.

        :type corpus: CorpusReader
        :param corpus: The corpus from which we create an information
            content dictionary.
        :type weight_senses_equally: bool
        :param weight_senses_equally: If this is True, gives all
            possible senses equal weight rather than dividing by the
            number of possible senses.  (If a word has 3 synses, each
            sense gets 0.3333 per appearance when this is False, 1.0 when
            it is true.)
        :param smoothing: How much do we smooth synset counts (default is 1.0)
        :type smoothing: float
        :return: An information content dictionary
        """
        ...
    
    def custom_lemmas(self, tab_file, lang): # -> None:
        """
        Reads a custom tab file containing mappings of lemmas in the given
        language to Princeton WordNet 3.0 synset offsets, allowing NLTK's
        WordNet functions to then be used with that language.

        See the "Tab files" section at https://omwn.org/omw1.html for
        documentation on the Multilingual WordNet tab file format.

        :param tab_file: Tab file as a file or file-like object
        :type: lang str
        :param: lang ISO 639-3 code of the language of the tab file
        """
        ...
    
    def disable_custom_lemmas(self, lang): # -> None:
        """prevent synsets from being mistakenly added"""
        ...
    
    def digraph(self, inputs, rel=..., pos=..., maxdepth=..., shapes=..., attr=..., verbose=...): # -> str:
        """
        Produce a graphical representation from 'inputs' (a list of
        start nodes, which can be a mix of Synsets, Lemmas and/or words),
        and a synset relation, for drawing with the 'dot' graph visualisation
        program from the Graphviz package.

        Return a string in the DOT graph file language, which can then be
        converted to an image by nltk.parse.dependencygraph.dot2img(dot_string).

        Optional Parameters:
        :rel: Wordnet synset relation
        :pos: for words, restricts Part of Speech to 'n', 'v', 'a' or 'r'
        :maxdepth: limit the longest path
        :shapes: dictionary of strings that trigger a specified shape
        :attr: dictionary with global graph attributes
        :verbose: warn about cycles

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.digraph([wn.synset('dog.n.01')]))
        digraph G {
        "Synset('animal.n.01')" -> "Synset('organism.n.01')";
        "Synset('canine.n.02')" -> "Synset('carnivore.n.01')";
        "Synset('carnivore.n.01')" -> "Synset('placental.n.01')";
        "Synset('chordate.n.01')" -> "Synset('animal.n.01')";
        "Synset('dog.n.01')" -> "Synset('canine.n.02')";
        "Synset('dog.n.01')" -> "Synset('domestic_animal.n.01')";
        "Synset('domestic_animal.n.01')" -> "Synset('animal.n.01')";
        "Synset('living_thing.n.01')" -> "Synset('whole.n.02')";
        "Synset('mammal.n.01')" -> "Synset('vertebrate.n.01')";
        "Synset('object.n.01')" -> "Synset('physical_entity.n.01')";
        "Synset('organism.n.01')" -> "Synset('living_thing.n.01')";
        "Synset('physical_entity.n.01')" -> "Synset('entity.n.01')";
        "Synset('placental.n.01')" -> "Synset('mammal.n.01')";
        "Synset('vertebrate.n.01')" -> "Synset('chordate.n.01')";
        "Synset('whole.n.02')" -> "Synset('object.n.01')";
        }
        <BLANKLINE>
        """
        ...
    


class WordNetICCorpusReader(CorpusReader):
    """
    A corpus reader for the WordNet information content corpus.
    """
    def __init__(self, root, fileids) -> None:
        ...
    
    def ic(self, icfile): # -> dict[Any, Any]:
        """
        Load an information content file from the wordnet_ic corpus
        and return a dictionary.  This dictionary has just two keys,
        NOUN and VERB, whose values are dictionaries that map from
        synsets to information content values.

        :type icfile: str
        :param icfile: The name of the wordnet_ic file (e.g. "ic-brown.dat")
        :return: An information content dictionary
        """
        ...
    


def path_similarity(synset1, synset2, verbose=..., simulate_root=...):
    ...

def lch_similarity(synset1, synset2, verbose=..., simulate_root=...):
    ...

def wup_similarity(synset1, synset2, verbose=..., simulate_root=...):
    ...

def res_similarity(synset1, synset2, ic, verbose=...):
    ...

def jcn_similarity(synset1, synset2, ic, verbose=...):
    ...

def lin_similarity(synset1, synset2, ic, verbose=...):
    ...

def information_content(synset, ic): # -> float:
    ...

