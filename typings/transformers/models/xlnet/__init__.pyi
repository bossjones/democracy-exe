"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING

from ...utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available

_import_structure = ...
if not is_sentencepiece_available():
    ...
if not is_tokenizers_available():
    ...
if not is_torch_available():
    ...
if not is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
