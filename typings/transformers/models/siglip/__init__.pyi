"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available, is_vision_available
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from .processing_siglip import SiglipProcessor

_import_structure = ...
if not is_sentencepiece_available():
    ...
if not is_vision_available():
    ...
if not is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
