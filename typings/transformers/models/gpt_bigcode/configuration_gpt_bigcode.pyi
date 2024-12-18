"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""GPTBigCode configuration"""
logger = ...
class GPTBigCodeConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPTBigCodeModel`]. It is used to instantiate a
    GPTBigCode model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTBigCode
    [gpt_bigcode](https://huggingface.co/gpt_bigcode) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTBigCodeModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new",
            "gelu_pytorch_tanh"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to call the fused softmax in float32.
        scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to scale the attention softmax in float32.
        attention_type (`bool`, *optional*, defaults to `True`):
            Whether to use Multi-Query Attion (`True`) or Multi-Head Attention (`False`).
    Example:

    ```python
    >>> from transformers import GPTBigCodeConfig, GPTBigCodeModel

    >>> # Initializing a GPTBigCode configuration
    >>> configuration = GPTBigCodeConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPTBigCodeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(self, vocab_size=..., n_positions=..., n_embd=..., n_layer=..., n_head=..., n_inner=..., activation_function=..., resid_pdrop=..., embd_pdrop=..., attn_pdrop=..., layer_norm_epsilon=..., initializer_range=..., scale_attn_weights=..., use_cache=..., bos_token_id=..., eos_token_id=..., attention_softmax_in_fp32=..., scale_attention_softmax_in_fp32=..., multi_query=..., **kwargs) -> None:
        ...
    


