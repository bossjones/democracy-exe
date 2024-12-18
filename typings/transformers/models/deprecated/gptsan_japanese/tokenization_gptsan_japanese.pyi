"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for GPTSANJapanese."""
logger = ...
VOCAB_FILES_NAMES = ...
def load_vocab_and_emoji(vocab_file, emoji_file): # -> tuple[OrderedDict[Any, Any], OrderedDict[Any, Any], OrderedDict[Any, Any], Any]:
    """Loads a vocabulary file and emoji file into a dictionary."""
    ...

class GPTSanJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling
    - Return token_type_ids for Prefix-LM model
    The bagofword token represents a repetition of the previous token and is converted to 3 consecutive tokens when
    decoding In addition, the original Japanese special Sub-Word-Encoding has been released in this repository
    (https://github.com/tanreinama/Japanese-BPEEncoder_V2). The token_type_ids is a mask indicating the prefix input
    position of the Prefix-LM model. To specify a prefix position, specify a prefix input for prefix_text, or specify a
    sentence of the prefix part and the part after it as a text pair of batch input.

    Example:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17750
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [35993, 35998, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Example for Prefix-LM:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["input_ids"]
    [35993, 34347, 31459, 30647, 31448, 25, 30659, 35729, 35676, 35998, 32417, 30647, 17750, 35589, 17750, 35590, 321, 1281]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer("実は慶応(慶應)大学出身", prefix_text="吾輩は猫である🐯。")["token_type_ids"]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ```

    Example for batch encode:

    ```python
    >>> from transformers import GPTSanJapaneseTokenizer

    >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["input_ids"]
    [[35993, 35998, 8640, 25948, 35993, 35998, 30647, 35675, 35999, 35999], [35993, 35998, 10382, 9868, 35993, 35998, 30646, 9459, 30646, 35675]]

    >>> # Mask for Prefix-LM inputs
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["token_type_ids"]
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    >>> # Mask for padding
    >>> tokenizer([["武田信玄", "は、"], ["織田信長", "の配下の、"]], padding=True)["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|nottoken|>"`):
            The token used for unknown charactor
        pad_token (`str`, *optional*, defaults to `"<|separator|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"<|segmenter|>"`):
            A special token to separate token to prefix part and general input part.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, emoji_file, unk_token=..., pad_token=..., bos_token=..., eos_token=..., sep_token=..., do_clean_text=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self): # -> int:
        ...
    
    def get_vocab(self): # -> dict[Any, Any]:
        ...
    
    def convert_tokens_to_string(self, tokens): # -> LiteralString:
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        The tokenizer returns token_type_ids as separators between the Prefix part and the rest.
        token_type_ids is 1 for the Prefix part and 0 for the rest of the token.

        Example:
        ```python
        >>> from transformers import GPTSanJapaneseTokenizer

        >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("ｱｲｳｴ")
        >>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

        >>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
        >>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
        >>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

        >>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
        >>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
        ```"""
        ...
    
    def prepare_for_tokenization(self, text, prefix_text=..., add_sep_token=..., **kwargs): # -> tuple[Any, dict[str, Any]]:
        ...
    


class SubWordJapaneseTokenizer:
    """
    This tokenizer is based on GPTNeoXJapaneseTokenizer and has the following modifications
    - Decoding byte0~byte255 tokens correctly
    - Added bagofword token handling

    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, vocab, ids_to_tokens, emoji) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def clean_text(self, content): # -> str:
        ...
    
    def tokenize(self, text, clean=...): # -> list[Any]:
        ...
    
    def convert_id_to_token(self, index):
        ...
    

