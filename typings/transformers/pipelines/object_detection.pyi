"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available
from .base import Pipeline, build_pipeline_init_args

if is_vision_available():
    ...
if is_torch_available():
    ...
logger = ...
Prediction = Dict[str, Any]
Predictions = List[Prediction]
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

    >>> # x, y  are expressed relative to the top left hand corner.
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability necessary to make a prediction.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **box** (`List[Dict[str, int]]`) -- The bounding box of detected object in image's original size.
        """
        ...
    
    def preprocess(self, image, timeout=...): # -> BatchEncoding | transformers.feature_extraction_utils.BatchFeature | transformers.image_processing_base.BatchFeature:
        ...
    
    def postprocess(self, model_outputs, threshold=...): # -> list[dict[str, Any | str | Dict[str, int]]] | list[dict[str, Any]]:
        ...
    


