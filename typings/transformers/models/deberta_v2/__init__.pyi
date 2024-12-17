"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_deberta_v2 import DebertaV2Config, DebertaV2OnnxConfig
from .tokenization_deberta_v2 import DebertaV2Tokenizer

_import_structure = ...
if not is_tokenizers_available():
    ...
if not is_tf_available():
    ...
if not is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
