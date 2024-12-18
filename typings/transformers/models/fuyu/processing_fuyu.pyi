"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Union
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available
from .image_processing_fuyu import FuyuBatchFeature

"""
Image/Text processor class for GIT
"""
if is_torch_available():
    ...
logger = ...
if is_torch_available():
    ...
TEXT_REPR_BBOX_OPEN = ...
TEXT_REPR_BBOX_CLOSE = ...
TEXT_REPR_POINT_OPEN = ...
TEXT_REPR_POINT_CLOSE = ...
TOKEN_BBOX_OPEN_STRING = ...
TOKEN_BBOX_CLOSE_STRING = ...
TOKEN_POINT_OPEN_STRING = ...
TOKEN_POINT_CLOSE_STRING = ...
BEGINNING_OF_ANSWER_STRING = ...
class FuyuProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...


def full_unpacked_stream_to_tensor(all_bi_tokens_to_place: List[int], full_unpacked_stream: List[torch.Tensor], fill_value: int, batch_size: int, new_seq_len: int, offset: int) -> torch.Tensor:
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """
    ...

def construct_full_unpacked_stream(num_real_text_tokens: Union[List[List[int]], torch.Tensor], input_stream: torch.Tensor, image_tokens: List[List[torch.Tensor]], batch_size: int, num_sub_sequences: int) -> List[torch.Tensor]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""
    ...

def original_to_transformed_h_coords(original_coords, scale_h):
    ...

def original_to_transformed_w_coords(original_coords, scale_w):
    ...

def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> List[int]:
    ...

def scale_bbox_to_transformed_image(top: float, left: float, bottom: float, right: float, scale_factor: float) -> List[int]:
    ...

class FuyuProcessor(ProcessorMixin):
    r"""
    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ...
    valid_kwargs = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer, **kwargs) -> None:
        ...
    
    def get_sample_encoding(self, prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, image_placeholder_id, image_newline_id, tensor_batch_images): # -> dict[str, Tensor]:
        ...
    
    def __call__(self, images: ImageInput = ..., text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = ..., audio=..., videos=..., **kwargs: Unpack[FuyuProcessorKwargs]) -> FuyuBatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        ...
    
    def post_process_box_coordinates(self, outputs, target_sizes=...): # -> list[Any]:
        """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """
        ...
    
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        ...
    


