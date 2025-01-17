"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

class LlavaNextVideoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaNextVideoForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)
    model.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32001):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_grid_pinpoints (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        video_token_index (`int`, *optional*, defaults to 32000):
            The video token index to encode the image prompt.
        spatial_pool_mode (`str`, *optional*, defaults to `"average"`):
            Pooling mode to use for videos. Can be "average", "max" or "conv".
        spatial_pool_stride (`int`, *optional*, defaults to 2):
            Stride used in the pooling layer for videos.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        video_seq_length (`int`, *optional*, defaults to 288):
            Sequence length of one video embedding.

    Example:

    ```python
    >>> from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> configuration = LlavaNextVideoConfig(vision_config, text_config)

    >>> model = LlavaNextVideoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    is_composition = ...
    def __init__(self, vision_config=..., text_config=..., ignore_index=..., image_token_index=..., projector_hidden_act=..., vision_feature_select_strategy=..., vision_feature_layer=..., image_grid_pinpoints=..., tie_word_embeddings=..., video_token_index=..., spatial_pool_mode=..., spatial_pool_stride=..., image_seq_length=..., video_seq_length=..., **kwargs) -> None:
        ...
    


