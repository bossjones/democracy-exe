"""
This type stub file was generated by pyright.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import ModelOutput

"""Tokenization class for Wav2Vec2Phoneme."""
logger = ...
if TYPE_CHECKING:
    ...
VOCAB_FILES_NAMES = ...
ListOfDict = List[Dict[str, Union[int, str]]]
@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2PhonemeCTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
    """
    text: Union[List[str], str]
    char_offsets: Union[List[ListOfDict], ListOfDict] = ...


class Wav2Vec2PhonemeCTCTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2PhonemeCTC tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_phonemize (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
            tokenizer, `do_phonemize` should be set to `False`.
        phonemizer_lang (`str`, *optional*, defaults to `"en-us"`):
            The language of the phoneme set to which the tokenizer should phonetize the input text to.
        phonemizer_backend (`str`, *optional*. defaults to `"espeak"`):
            The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
            See the [phonemizer package](https://github.com/bootphon/phonemizer#readme). for more information.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., unk_token=..., pad_token=..., phone_delimiter_token=..., word_delimiter_token=..., do_phonemize=..., phonemizer_lang=..., phonemizer_backend=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def init_backend(self, phonemizer_lang: str): # -> None:
        """
        Initializes the backend.

        Args:
            phonemizer_lang (`str`): The language to be used.
        """
        ...
    
    def prepare_for_tokenization(self, text: str, is_split_into_words: bool = ..., phonemizer_lang: Optional[str] = ..., do_phonemize: Optional[bool] = ...) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            phonemizer_lang (`str`, *optional*):
                The language of the phoneme set to which the tokenizer should phonetize the input text to.
            do_phonemize (`bool`, *optional*):
                Whether the tokenizer should phonetize the input text or not. Only if a sequence of phonemes is passed
                to the tokenizer, `do_phonemize` should be set to `False`.


        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        ...
    
    def phonemize(self, text: str, phonemizer_lang: Optional[str] = ...) -> str:
        ...
    
    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        ...
    
    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        ...
    
    @word_delimiter_token.setter
    def word_delimiter_token(self, value): # -> None:
        ...
    
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value): # -> None:
        ...
    
    @property
    def phone_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        ...
    
    @property
    def phone_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the phone_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        ...
    
    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value): # -> None:
        ...
    
    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value): # -> None:
        ...
    
    def convert_tokens_to_string(self, tokens: List[str], group_tokens: bool = ..., spaces_between_special_tokens: bool = ..., filter_word_delimiter_token: bool = ..., output_char_offsets: bool = ...) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        ...
    
    def decode(self, token_ids: Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor], skip_special_tokens: bool = ..., clean_up_tokenization_spaces: bool = ..., output_char_offsets: bool = ..., **kwargs) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better
                understand how to make use of `output_word_offsets`.
                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works the same way with
                phonemes.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The decoded
            sentence. Will be a [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]
            when `output_char_offsets == True`.
        """
        ...
    
    def batch_decode(self, sequences: Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor], skip_special_tokens: bool = ..., clean_up_tokenization_spaces: bool = ..., output_char_offsets: bool = ..., **kwargs) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better
                understand how to make use of `output_word_offsets`.
                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works analogous with phonemes
                and batched output.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The
            decoded sentence. Will be a
            [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`] when
            `output_char_offsets == True`.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


