"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_tf_available, is_torch_available
from .configuration_wav2vec2 import Wav2Vec2Config
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .processing_wav2vec2 import Wav2Vec2Processor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer

_import_structure = ...
if not is_torch_available():
    ...
if not is_tf_available():
    ...
if not is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
