"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""Dac model configuration"""
logger = ...
class DacConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`DacModel`]. It is used to instantiate a
    Dac model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [descript/dac_16khz](https://huggingface.co/descript/dac_16khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_hidden_size (`int`, *optional*, defaults to 64):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 4, 8, 8]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 1536):
            Intermediate representation dimension for the decoder.
        n_codebooks (`int`, *optional*, defaults to 9):
            Number of codebooks in the VQVAE.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of discrete codes in each codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Dimension of the codebook vectors. If not defined, uses `encoder_hidden_size`.
        quantizer_dropout (`bool`, *optional*, defaults to 0):
            Whether to apply dropout to the quantizer.
        commitment_loss_weight (float, *optional*, defaults to 0.25):
            Weight of the commitment loss term in the VQVAE loss function.
        codebook_loss_weight (float, *optional*, defaults to 1.0):
            Weight of the codebook loss term in the VQVAE loss function.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    Example:

    ```python
    >>> from transformers import DacModel, DacConfig

    >>> # Initializing a "descript/dac_16khz" style configuration
    >>> configuration = DacConfig()

    >>> # Initializing a model (with random weights) from the "descript/dac_16khz" style configuration
    >>> model = DacModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, encoder_hidden_size=..., downsampling_ratios=..., decoder_hidden_size=..., n_codebooks=..., codebook_size=..., codebook_dim=..., quantizer_dropout=..., commitment_loss_weight=..., codebook_loss_weight=..., sampling_rate=..., **kwargs) -> None:
        ...
    
    @property
    def frame_rate(self) -> int:
        ...
    


