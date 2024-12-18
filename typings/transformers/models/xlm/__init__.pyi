"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available
from .configuration_xlm import XLMConfig, XLMOnnxConfig
from .tokenization_xlm import XLMTokenizer

_import_structure = ...
if not is_torch_available():
    ...
if not is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
