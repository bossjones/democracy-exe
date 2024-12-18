"""
This type stub file was generated by pyright.
"""

from typing import List, Optional
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast tokenization class for BlenderbotSmall."""
logger = ...
VOCAB_FILES_NAMES = ...
class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" BlenderbotSmall tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    vocab_files_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., merges_file=..., unk_token=..., bos_token=..., eos_token=..., add_prefix_space=..., trim_offsets=..., **kwargs) -> None:
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...): # -> list[int | None]:
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
        does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        ...
    


