"""
This type stub file was generated by pyright.
"""

import os
from typing import Union
from ...configuration_utils import PretrainedConfig

"""Idefics3 model configuration"""
logger = ...
class Idefics3VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics3VisionModel`]. It is used to instantiate a
    Idefics3 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) used in the Idefics3 model
    [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        intializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing all weight matrices in the model.

    Example:

    ```python
    >>> from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
    >>> from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig

    >>> # Initializing a Idefics3VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics3VisionConfig()

    >>> # Initializing a Idefics3VisionTransformer (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics3VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, hidden_size=..., intermediate_size=..., num_hidden_layers=..., num_attention_heads=..., num_channels=..., image_size=..., patch_size=..., hidden_act=..., layer_norm_eps=..., attention_dropout=..., initializer_range=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        ...
    


class Idefics3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics3Model`]. It is used to instantiate a
    Idefics3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics3
    [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism. Only
            relevant if `config.is_decoder=True`.
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*, defaults to `IdeficsVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `LlamaConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        pad_token_id (`int`, *optional*, defaults to 128002):
            The id of the padding token.

    Example:
    ```python
    >>> from transformers import Idefics3Model, Idefics3Config
    >>> # Initializing configuration
    >>> configuration = Idefics3Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    is_composition = ...
    def __init__(self, use_cache=..., image_token_id=..., tie_word_embeddings=..., vision_config=..., text_config=..., scale_factor=..., pad_token_id=..., **kwargs) -> None:
        ...
    

