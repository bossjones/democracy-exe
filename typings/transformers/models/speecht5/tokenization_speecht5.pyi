"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization class for SpeechT5."""
logger = ...
VOCAB_FILES_NAMES = ...
class SpeechT5Tokenizer(PreTrainedTokenizer):
    """
    Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether to convert numeric quantities in the text to their spelt-out english counterparts.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., unk_token=..., pad_token=..., normalize=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...

    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs): # -> tuple[str | Any, dict[str, Any]]:
        ...

    @property
    def vocab_size(self):
        ...

    @property
    def normalizer(self): # -> EnglishNumberNormalizer:
        ...

    @normalizer.setter
    def normalizer(self, value): # -> None:
        ...

    def get_vocab(self): # -> dict[str, int]:
        ...

    def __getstate__(self): # -> dict[str, Any]:
        ...

    def __setstate__(self, d): # -> None:
        ...

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        ...

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
