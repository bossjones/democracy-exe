"""
This type stub file was generated by pyright.
"""

from typing import List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, is_vision_available
from .base import Pipeline, build_pipeline_init_args
from PIL import Image

if is_vision_available():
    ...
if is_tf_available():
    ...
if is_torch_available():
    ...
logger = ...
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
class ImageToTextPipeline(Pipeline):
    """
    Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def __call__(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]] = ..., **kwargs): # -> list[Any] | PipelineIterator | Generator[Any, Any, None] | Tensor | Any | None:
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

            max_new_tokens (`int`, *optional*):
                The amount of maximum tokens to generate. By default it will use `generate` default.

            generate_kwargs (`Dict`, *optional*):
                Pass it to send all of these arguments directly to `generate` allowing full control of this function.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        ...
    
    def preprocess(self, image, prompt=..., timeout=...): # -> transformers.feature_extraction_utils.BatchFeature | transformers.image_processing_base.BatchFeature:
        ...
    
    def postprocess(self, model_outputs): # -> list[Any]:
        ...
    


