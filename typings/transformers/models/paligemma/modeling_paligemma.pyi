"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from ...cache_utils import Cache
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_flash_attn_2_available, replace_return_docstrings
from .configuration_paligemma import PaliGemmaConfig

"""PyTorch PaliGemmamodel."""
if is_flash_attn_2_available():
    ...
logger = ...
_CONFIG_FOR_DOC = ...
@dataclass
class PaliGemmaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for PaliGemmacausal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig) -> None:
        ...
    
    def forward(self, image_features): # -> Any:
        ...
    


PALIGEMMA_START_DOCSTRING = ...
@add_start_docstrings("The bare LLaMA Model outputting raw hidden-states without any specific head on top.", PALIGEMMA_START_DOCSTRING)
class PaliGemmaPreTrainedModel(PreTrainedModel):
    config_class = PaliGemmaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    _supports_cache_class = ...


PALIGEMMA_INPUTS_DOCSTRING = ...
@add_start_docstrings("""The PALIGEMMA model which consists of a vision backbone and a language model.""", PALIGEMMA_START_DOCSTRING)
class PaliGemmaForConditionalGeneration(PaliGemmaPreTrainedModel):
    def __init__(self, config: PaliGemmaConfig) -> None:
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
    
    @add_start_docstrings_to_model_forward(PALIGEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PaliGemmaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., pixel_values: torch.FloatTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = ..., token_type_ids: Optional[torch.LongTensor] = ..., cache_position: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, PaliGemmaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/PaliGemma-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/PaliGemma-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., inputs_embeds=..., cache_position=..., position_ids=..., pixel_values=..., attention_mask=..., token_type_ids=..., use_cache=..., **kwargs): # -> dict[str, Any]:
        ...
    


