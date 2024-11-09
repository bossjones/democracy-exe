"""
This type stub file was generated by pyright.
"""

from PIL import Image

from ..models.auto import AutoProcessor
from ..models.vision_encoder_decoder import VisionEncoderDecoderModel
from ..utils import is_vision_available
from .base import PipelineTool

if is_vision_available():
    ...
class DocumentQuestionAnsweringTool(PipelineTool):
    default_checkpoint = ...
    description = ...
    name = ...
    pre_processor_class = AutoProcessor
    model_class = VisionEncoderDecoderModel
    inputs = ...
    outputs = ...
    def __init__(self, *args, **kwargs) -> None:
        ...

    def encode(self, document: Image, question: str): # -> dict[str, Any]:
        ...

    def forward(self, inputs):
        ...

    def decode(self, outputs):
        ...
