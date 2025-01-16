"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""PaliGemmamodel configuration"""
logger = ...
class PaliGemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PaliGemmaForConditionalGeneration`]. It is used to instantiate an
    PaliGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`PaliGemmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PaliGemmamodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PaliGemmaForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    is_composition = ...
    def __init__(self, vision_config=..., text_config=..., ignore_index=..., image_token_index=..., vocab_size=..., projection_dim=..., hidden_size=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self): # -> int:
        ...
    
    @vocab_size.setter
    def vocab_size(self, value): # -> None:
        ...
    
    def to_dict(self): # -> Dict[str, Any]:
        ...
    


