"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_fuyu import FuyuConfig

"""PyTorch Fuyu model."""
logger = ...
_CONFIG_FOR_DOC = ...
FUYU_START_DOCSTRING = ...
@add_start_docstrings("The bare Fuyu Model outputting raw hidden-states without any specific head on top.", FUYU_START_DOCSTRING)
class FuyuPreTrainedModel(PreTrainedModel):
    config_class = FuyuConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...


FUYU_INPUTS_DOCSTRING = ...
@add_start_docstrings("Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.", FUYU_START_DOCSTRING)
class FuyuForCausalLM(FuyuPreTrainedModel):
    def __init__(self, config: FuyuConfig) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_decoder(self, decoder): # -> None:
        ...
    
    def get_decoder(self):
        ...
    
    def tie_weights(self):
        ...
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = ..., pad_to_multiple_of=...) -> nn.Embedding:
        ...
    
    def gather_continuous_embeddings(self, word_embeddings: torch.Tensor, continuous_embeddings: List[torch.Tensor], image_patch_input_indices: torch.Tensor) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
        ...
    
    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., image_patches: torch.Tensor = ..., image_patches_indices: torch.Tensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import FuyuProcessor, FuyuForCausalLM
        >>> from PIL import Image
        >>> import requests

        >>> processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        >>> model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Generate a coco-style caption.\n"

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=7)
        >>> generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
        >>> print(generation_text[0])
        A blue bus parked on the side of a road.
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., inputs_embeds=..., image_patches=..., image_patches_indices=..., **kwargs): # -> dict[str, Any]:
        ...
    


