"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for UDOP.
"""
logger = ...
class UdopTextKwargs(TextKwargs, total=False):
    word_labels: Optional[Union[List[int], List[List[int]]]]
    boxes: Union[List[List[int]], List[List[List[int]]]]
    ...


class UdopProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: UdopTextKwargs
    _defaults = ...


class UdopProcessor(ProcessorMixin):
    r"""
    Constructs a UDOP processor which combines a LayoutLMv3 image processor and a UDOP tokenizer into a single processor.

    [`UdopProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3ImageProcessor`] to resize, rescale and normalize document images, and optionally applies OCR
    to get words and normalized bounding boxes. These are then provided to [`UdopTokenizer`] or [`UdopTokenizerFast`],
    which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    Additionally, it also supports passing `text_target` and `text_pair_target` to the tokenizer, which can be used to
    prepare labels for language modeling tasks.

    Args:
        image_processor (`LayoutLMv3ImageProcessor`):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`):
            An instance of [`UdopTokenizer`] or [`UdopTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    optional_call_args = ...
    def __init__(self, image_processor, tokenizer) -> None:
        ...
    
    def __call__(self, images: Optional[ImageInput] = ..., text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = ..., *args, audio=..., videos=..., **kwargs: Unpack[UdopProcessorKwargs]) -> BatchFeature:
        """
        This method first forwards the `images` argument to [`~UdopImageProcessor.__call__`]. In case
        [`UdopImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~UdopTokenizer.__call__`] and returns the output,
        together with the prepared `pixel_values`. In case [`UdopImageProcessor`] was initialized with `apply_ocr` set
        to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the
        additional arguments to [`~UdopTokenizer.__call__`] and returns the output, together with the prepared
        `pixel_values`.

        Alternatively, one can pass `text_target` and `text_pair_target` to prepare the targets of UDOP.

        Please refer to the docstring of the above two methods for more information.
        """
        ...
    
    def get_overflowing_images(self, images, overflow_to_sample_mapping): # -> list[Any]:
        ...
    
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        ...
    
    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        ...
    
    @property
    def model_input_names(self): # -> list[str]:
        ...
    


