"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader.api import *

"""
Read tokens, phonemes and audio data from the NLTK TIMIT Corpus.

This corpus contains selected portion of the TIMIT corpus.

 - 16 speakers from 8 dialect regions
 - 1 male and 1 female from each dialect region
 - total 130 sentences (10 sentences per speaker.  Note that some
   sentences are shared among other speakers, especially sa1 and sa2
   are spoken by all speakers.)
 - total 160 recording of sentences (10 recordings per speaker)
 - audio format: NIST Sphere, single channel, 16kHz sampling,
   16 bit sample, PCM encoding


Module contents
===============

The timit corpus reader provides 4 functions and 4 data items.

 - utterances

   List of utterances in the corpus.  There are total 160 utterances,
   each of which corresponds to a unique utterance of a speaker.
   Here's an example of an utterance identifier in the list::

       dr1-fvmh0/sx206
         - _----  _---
         | |  |   | |
         | |  |   | |
         | |  |   | `--- sentence number
         | |  |   `----- sentence type (a:all, i:shared, x:exclusive)
         | |  `--------- speaker ID
         | `------------ sex (m:male, f:female)
         `-------------- dialect region (1..8)

 - speakers

   List of speaker IDs.  An example of speaker ID::

       dr1-fvmh0

   Note that if you split an item ID with colon and take the first element of
   the result, you will get a speaker ID.

       >>> itemid = 'dr1-fvmh0/sx206'
       >>> spkrid , sentid = itemid.split('/')
       >>> spkrid
       'dr1-fvmh0'

   The second element of the result is a sentence ID.

 - dictionary()

   Phonetic dictionary of words contained in this corpus.  This is a Python
   dictionary from words to phoneme lists.

 - spkrinfo()

   Speaker information table.  It's a Python dictionary from speaker IDs to
   records of 10 fields.  Speaker IDs the same as the ones in timie.speakers.
   Each record is a dictionary from field names to values, and the fields are
   as follows::

     id         speaker ID as defined in the original TIMIT speaker info table
     sex        speaker gender (M:male, F:female)
     dr         speaker dialect region (1:new england, 2:northern,
                3:north midland, 4:south midland, 5:southern, 6:new york city,
                7:western, 8:army brat (moved around))
     use        corpus type (TRN:training, TST:test)
                in this sample corpus only TRN is available
     recdate    recording date
     birthdate  speaker birth date
     ht         speaker height
     race       speaker race (WHT:white, BLK:black, AMR:american indian,
                SPN:spanish-american, ORN:oriental,???:unknown)
     edu        speaker education level (HS:high school, AS:associate degree,
                BS:bachelor's degree (BS or BA), MS:master's degree (MS or MA),
                PHD:doctorate degree (PhD,JD,MD), ??:unknown)
     comments   comments by the recorder

The 4 functions are as follows.

 - tokenized(sentences=items, offset=False)

   Given a list of items, returns an iterator of a list of word lists,
   each of which corresponds to an item (sentence).  If offset is set to True,
   each element of the word list is a tuple of word(string), start offset and
   end offset, where offset is represented as a number of 16kHz samples.

 - phonetic(sentences=items, offset=False)

   Given a list of items, returns an iterator of a list of phoneme lists,
   each of which corresponds to an item (sentence).  If offset is set to True,
   each element of the phoneme list is a tuple of word(string), start offset
   and end offset, where offset is represented as a number of 16kHz samples.

 - audiodata(item, start=0, end=None)

   Given an item, returns a chunk of audio samples formatted into a string.
   When the function is called, if start and end are omitted, the entire
   samples of the recording will be returned.  If only end is omitted,
   samples from the start offset to the end of the recording will be returned.

 - play(data)

   Play the given audio samples. The audio samples can be obtained from the
   timit.audiodata function.

"""
class TimitCorpusReader(CorpusReader):
    """
    Reader for the TIMIT corpus (or any other corpus with the same
    file layout and use of file formats).  The corpus root directory
    should contain the following files:

      - timitdic.txt: dictionary of standard transcriptions
      - spkrinfo.txt: table of speaker information

    In addition, the root directory should contain one subdirectory
    for each speaker, containing three files for each utterance:

      - <utterance-id>.txt: text content of utterances
      - <utterance-id>.wrd: tokenized text content of utterances
      - <utterance-id>.phn: phonetic transcription of utterances
      - <utterance-id>.wav: utterance sound file
    """
    _FILE_RE = ...
    _UTTERANCE_RE = ...
    def __init__(self, root, encoding=...) -> None:
        """
        Construct a new TIMIT corpus reader in the given directory.
        :param root: The root directory for this corpus.
        """
        ...
    
    def fileids(self, filetype=...): # -> list[str | Any] | list[Any] | list[str]:
        """
        Return a list of file identifiers for the files that make up
        this corpus.

        :param filetype: If specified, then ``filetype`` indicates that
            only the files that have the given type should be
            returned.  Accepted values are: ``txt``, ``wrd``, ``phn``,
            ``wav``, or ``metadata``,
        """
        ...
    
    def utteranceids(self, dialect=..., sex=..., spkrid=..., sent_type=..., sentid=...): # -> list[str | Any]:
        """
        :return: A list of the utterance identifiers for all
            utterances in this corpus, or for the given speaker, dialect
            region, gender, sentence type, or sentence number, if
            specified.
        """
        ...
    
    def transcription_dict(self): # -> dict[Any, Any]:
        """
        :return: A dictionary giving the 'standard' transcription for
            each word.
        """
        ...
    
    def spkrid(self, utterance):
        ...
    
    def sentid(self, utterance):
        ...
    
    def utterance(self, spkrid, sentid): # -> str:
        ...
    
    def spkrutteranceids(self, speaker): # -> list[str | Any]:
        """
        :return: A list of all utterances associated with a given
            speaker.
        """
        ...
    
    def spkrinfo(self, speaker):
        """
        :return: A dictionary mapping .. something.
        """
        ...
    
    def phones(self, utterances=...): # -> list[Any]:
        ...
    
    def phone_times(self, utterances=...): # -> list[Any]:
        """
        offset is represented as a number of 16kHz samples!
        """
        ...
    
    def words(self, utterances=...): # -> list[Any]:
        ...
    
    def word_times(self, utterances=...): # -> list[Any]:
        ...
    
    def sents(self, utterances=...): # -> list[Any]:
        ...
    
    def sent_times(self, utterances=...): # -> list[tuple[bytes | str | Any, int, int]]:
        ...
    
    def phone_trees(self, utterances=...): # -> list[Any]:
        ...
    
    def wav(self, utterance, start=..., end=...): # -> bytes:
        ...
    
    def audiodata(self, utterance, start=..., end=...): # -> bytes | str:
        ...
    
    def play(self, utterance, start=..., end=...): # -> None:
        """
        Play the given audio sample.

        :param utterance: The utterance id of the sample to play
        """
        ...
    


class SpeakerInfo:
    def __init__(self, id, sex, dr, use, recdate, birthdate, ht, race, edu, comments=...) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def read_timit_block(stream): # -> list[Any]:
    """
    Block reader for timit tagged sentences, which are preceded by a sentence
    number that will be ignored.
    """
    ...

