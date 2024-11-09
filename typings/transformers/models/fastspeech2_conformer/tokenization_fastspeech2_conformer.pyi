"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for FastSpeech2Conformer."""
logger = ...
VOCAB_FILES_NAMES = ...
class FastSpeech2ConformerTokenizer(PreTrainedTokenizer):
    """
    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        should_strip_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the spaces from the list of tokens.
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., pad_token=..., unk_token=..., should_strip_spaces=..., **kwargs) -> None:
        ...

    @property
    def vocab_size(self): # -> int:
        ...

    def get_vocab(self): # -> dict[Any, Any]:
        "Returns vocab as a dict"
        ...

    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs): # -> tuple[str, dict[str, Any]]:
        ...

    def decode(self, token_ids, **kwargs):
        ...

    def convert_tokens_to_string(self, tokens, **kwargs):
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        ...

    def __getstate__(self): # -> dict[str, Any]:
        ...

    def __setstate__(self, d): # -> None:
        ...
