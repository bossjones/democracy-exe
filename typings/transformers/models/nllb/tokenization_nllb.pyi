"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer

logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
FAIRSEQ_LANGUAGE_CODES = ...
class NllbTokenizer(PreTrainedTokenizer):
    """
    Construct an NLLB tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import NllbTokenizer

    >>> tokenizer = NllbTokenizer.from_pretrained(
    ...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*):
            The language to use as target language for translation.
        sp_model_kwargs (`Dict[str, str]`):
            Additional keyword arguments to pass to the model initialization.
    """
    vocab_files_names = ...
    model_input_names = ...
    prefix_tokens: List[int] = ...
    suffix_tokens: List[int] = ...
    def __init__(self, vocab_file, bos_token=..., eos_token=..., sep_token=..., cls_token=..., unk_token=..., pad_token=..., mask_token=..., tokenizer_file=..., src_lang=..., tgt_lang=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., additional_special_tokens=..., legacy_behaviour=..., **kwargs) -> None:
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, d): # -> None:
        ...
    
    @property
    def vocab_size(self): # -> int:
        ...
    
    @property
    def src_lang(self) -> str:
        ...
    
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """
        ...
    
    def get_vocab(self): # -> dict[str, int]:
        ...
    
    def convert_tokens_to_string(self, tokens): # -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str = ..., tgt_texts: Optional[List[str]] = ..., tgt_lang: str = ..., **kwargs) -> BatchEncoding:
        ...
    
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        - In legacy mode: No prefix and suffix=[eos, src_lang_code].
        - In default mode: Prefix=[src_lang_code], suffix = [eos]
        """
        ...
    
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]
        """
        ...
    


