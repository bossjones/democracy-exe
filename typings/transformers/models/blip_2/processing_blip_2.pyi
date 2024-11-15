"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

"""
Processor class for BLIP-2.
"""
class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None:
        ...

    def __call__(self, images: ImageInput = ..., text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., stride: int = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_token_type_ids: bool = ..., return_length: bool = ..., verbose: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        ...

    @property
    def model_input_names(self): # -> list[Any]:
        ...
