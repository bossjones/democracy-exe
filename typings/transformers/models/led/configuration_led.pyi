"""
This type stub file was generated by pyright.
"""

from typing import List, Union

from ...configuration_utils import PretrainedConfig

""" LED model configuration"""
logger = ...
class LEDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LEDModel`]. It is used to instantiate an LED
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LED
    [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the LED model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LEDModel`] or [`TFLEDModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that the encoder might ever be used with.
        max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that the decoder might ever be used with.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models)

    Example:

    ```python
    >>> from transformers import LEDModel, LEDConfig

    >>> # Initializing a LED allenai/led-base-16384 style configuration
    >>> configuration = LEDConfig()

    >>> # Initializing a model from the allenai/led-base-16384 style configuration
    >>> model = LEDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    attribute_map = ...
    def __init__(self, vocab_size=..., max_encoder_position_embeddings=..., max_decoder_position_embeddings=..., encoder_layers=..., encoder_ffn_dim=..., encoder_attention_heads=..., decoder_layers=..., decoder_ffn_dim=..., decoder_attention_heads=..., encoder_layerdrop=..., decoder_layerdrop=..., use_cache=..., is_encoder_decoder=..., activation_function=..., d_model=..., dropout=..., attention_dropout=..., activation_dropout=..., init_std=..., decoder_start_token_id=..., classifier_dropout=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., attention_window: Union[List[int], int] = ..., **kwargs) -> None:
        ...
