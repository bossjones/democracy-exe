"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""XLNet configuration"""
logger = ...
class XLNetConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`XLNetModel`] or a [`TFXLNetModel`]. It is used to
    instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [xlnet/xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`XLNetModel`] or [`TFXLNetModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_inner (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        ff_activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and
            `"gelu_new"` are supported.
        untie_r (`bool`, *optional*, defaults to `True`):
            Whether or not to untie relative position biases
        attn_type (`str`, *optional*, defaults to `"bi"`):
            The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        mem_len (`int` or `None`, *optional*):
            The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
            forward pass won't be re-computed. See the
            [quickstart](https://huggingface.co/transformers/quickstart.html#using-the-past) for more information.
        reuse_len (`int`, *optional*):
            The number of tokens in the current batch to be cached and reused in the future.
        bi_data (`bool`, *optional*, defaults to `False`):
            Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False`
            during finetuning.
        clamp_len (`int`, *optional*, defaults to -1):
            Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.
        same_length (`bool`, *optional*, defaults to `False`):
            Whether or not to use the same attention length for each token.
        summary_type (`str`, *optional*, defaults to "last"):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`boo`, *optional*, defaults to `True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_last_dropout (`float`, *optional*, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        use_mems_eval (`bool`, *optional*, defaults to `True`):
            Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
        use_mems_train (`bool`, *optional*, defaults to `False`):
            Whether or not the model should make use of the recurrent memory mechanism in train mode.

            <Tip>

            For pretraining, it is recommended to set `use_mems_train` to `True`. For fine-tuning, it is recommended to
            set `use_mems_train` to `False` as discussed
            [here](https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587). If `use_mems_train` is set to
            `True`, one has to make sure that the train batches are correctly pre-processed, *e.g.* `batch_1 = [[This
            line is], [This is the]]` and `batch_2 = [[ the first line], [ second line]]` and that all batches are of
            equal size.

            </Tip>

    Examples:

    ```python
    >>> from transformers import XLNetConfig, XLNetModel

    >>> # Initializing a XLNet configuration
    >>> configuration = XLNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(self, vocab_size=..., d_model=..., n_layer=..., n_head=..., d_inner=..., ff_activation=..., untie_r=..., attn_type=..., initializer_range=..., layer_norm_eps=..., dropout=..., mem_len=..., reuse_len=..., use_mems_eval=..., use_mems_train=..., bi_data=..., clamp_len=..., same_length=..., summary_type=..., summary_use_proj=..., summary_activation=..., summary_last_dropout=..., start_n_top=..., end_n_top=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        """Constructs XLNetConfig."""
        ...
    
    @property
    def max_position_embeddings(self): # -> Literal[-1]:
        ...
    
    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        ...
    


