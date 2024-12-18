"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""VideoLlava model configuration"""
logger = ...
class VideoLlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VideoLlavaForConditionalGeneration`]. It is used to instantiate an
    VideoLlava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the like LanguageBind/Video-LLaVA-7B-hf.

    e.g. [LanguageBind/Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`VideoLlavaVisionConfig`, *optional*):
            Custom vision config or dict. Defaults to `CLIPVisionConfig` if not indicated.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
            Defaults to `LlamaConfig` if not indicated.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 32001):
            The video token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
            Can be either "full" to select all features or "default" to select features without `CLS`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_seq_length (`int`, *optional*, defaults to 256):
            Sequence length of one image embedding.
        video_seq_length (`int`, *optional*, defaults to 2056):
            Sequence length of one video embedding.

    Example:

    ```python
    >>> from transformers import VideoLlavaForConditionalGeneration, VideoLlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VideoLlava video_llava-1.5-7b style configuration
    >>> configuration = VideoLlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the video_llava-1.5-7b style configuration
    >>> model = VideoLlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    is_composition = ...
    def __init__(self, vision_config=..., text_config=..., ignore_index=..., image_token_index=..., video_token_index=..., projector_hidden_act=..., vision_feature_select_strategy=..., vision_feature_layer=..., image_seq_length=..., video_seq_length=..., **kwargs) -> None:
        ...
    

