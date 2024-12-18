"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for Ernie-M."""
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
RESOURCE_FILES_NAMES = ...
class ErnieMTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Ernie-M tokenizer. It uses the `sentencepiece` tools to cut the words to sub-words.

    Args:
        sentencepiece_model_file (`str`):
            The file path of sentencepiece model.
        vocab_file (`str`, *optional*):
            The file path of the vocabulary.
        do_lower_case (`str`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            A special token representing the `unknown (out-of-vocabulary)` token. An unknown token is set to be
            `unk_token` inorder to be converted to an ID.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            A special token separating two different sentences in the same input.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            A special token used to make arrays of tokens the same size for batching purposes.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            A special token used for sequence classification. It is the last token of the sequence when built with
            special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            A special token representing a masked token. This is the token used in the masked language modeling task
            which the model tries to predict the original unmasked ones.
    """
    model_input_names: List[str] = ...
    vocab_files_names = ...
    resource_files_names = ...
    def __init__(self, sentencepiece_model_ckpt, vocab_file=..., do_lower_case=..., encoding=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., **kwargs) -> None:
        ...
    
    def get_offset_mapping(self, text): # -> list[Any] | None:
        ...
    
    @property
    def vocab_size(self): # -> int:
        ...
    
    def get_vocab(self): # -> dict[Any, int]:
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, d): # -> None:
        ...
    
    def clean_text(self, text): # -> LiteralString:
        """Performs invalid character removal and whitespace cleanup on text."""
        ...
    
    def convert_tokens_to_string(self, tokens): # -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        ...
    
    def convert_ids_to_string(self, ids): # -> str:
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...): # -> list[int | None]:
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ErnieM sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of input_id with the appropriate special tokens.
        """
        ...
    
    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=...):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. An Ernie-M
        offset_mapping has the following format:

        - single sequence: `(0,0) X (0,0)`
        - pair of sequences: `(0,0) A (0,0) (0,0) B (0,0)`

        Args:
            offset_mapping_ids_0 (`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (`List[tuple]`, *optional*):
                Optional second list of wordpiece offsets for offset mapping pairs.
        Returns:
            `List[tuple]`: List of wordpiece offsets with the appropriate offsets of special tokens.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=..., already_has_special_tokens=...): # -> list[int]:
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `encode` method.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`str`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`:
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids) Should be overridden in a subclass if the model has a special way of
        building: those.

        Args:
            token_ids_0 (`List[int]`):
                The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*):
                The second tokenized sequence.
        Returns:
            `List[int]`: The token type ids.
        """
        ...
    
    def is_ch_char(self, char): # -> bool:
        """
        is_ch_char
        """
        ...
    
    def is_alpha(self, char): # -> bool:
        """
        is_alpha
        """
        ...
    
    def is_punct(self, char): # -> bool:
        """
        is_punct
        """
        ...
    
    def is_whitespace(self, char): # -> bool:
        """
        is whitespace
        """
        ...
    
    def load_vocab(self, filepath): # -> dict[Any, Any]:
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    

