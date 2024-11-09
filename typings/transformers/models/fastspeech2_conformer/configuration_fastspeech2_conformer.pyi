"""
This type stub file was generated by pyright.
"""

from typing import Dict

from ...configuration_utils import PretrainedConfig

""" FastSpeech2Conformer model configuration"""
logger = ...
class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to
    instantiate a FastSpeech2Conformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            The dimensionality of the hidden layers.
        vocab_size (`int`, *optional*, defaults to 78):
            The size of the vocabulary.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel filters used in the filter bank.
        encoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the encoder.
        encoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the encoder.
        encoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the encoder.
        decoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the decoder.
        decoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the decoder.
        decoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the decoder.
        speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
            The number of layers in the post-net of the speech decoder.
        speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
            The number of units in the post-net layers of the speech decoder.
        speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
            The kernel size in the post-net of the speech decoder.
        positionwise_conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolution kernel used in the position-wise layer.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before encoder layers.
        decoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before decoder layers.
        encoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after encoder layers.
        decoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after decoder layers.
        reduction_factor (`int`, *optional*, defaults to 1):
            The factor by which the speech frame rate is reduced.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            The speed of the speech produced.
        use_macaron_style_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use macaron style in the conformer.
        use_cnn_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use convolutional neural networks in the conformer.
        encoder_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size used in the encoder.
        decoder_kernel_size (`int`, *optional*, defaults to 31):
            The kernel size used in the decoder.
        duration_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the duration predictor.
        duration_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the duration predictor.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the duration predictor.
        energy_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the energy predictor.
        energy_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the energy predictor.
        energy_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the energy predictor.
        energy_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the energy predictor.
        energy_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the energy embed layer.
        energy_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the energy embed layer.
        stop_gradient_from_energy_predictor (`bool`, *optional*, defaults to `False`):
            Specifies whether to stop gradients from the energy predictor.
        pitch_predictor_layers (`int`, *optional*, defaults to 5):
            The number of layers in the pitch predictor.
        pitch_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the pitch predictor.
        pitch_predictor_kernel_size (`int`, *optional*, defaults to 5):
            The kernel size used in the pitch predictor.
        pitch_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the pitch predictor.
        pitch_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the pitch embed layer.
        pitch_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the pitch embed layer.
        stop_gradient_from_pitch_predictor (`bool`, *optional*, defaults to `True`):
            Specifies whether to stop gradients from the pitch predictor.
        encoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the encoder.
        encoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the encoder.
        encoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the encoder.
        decoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the decoder.
        decoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the decoder.
        decoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the decoder.
        duration_predictor_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the duration predictor.
        speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the speech decoder postnet.
        max_source_positions (`int`, *optional*, defaults to 5000):
            if `"relative"` position embeddings are used, defines the maximum source input positions.
        use_masking (`bool`, *optional*, defaults to `True`):
            Specifies whether to use masking in the model.
        use_weighted_masking (`bool`, *optional*, defaults to `False`):
            Specifies whether to use weighted masking in the model.
        num_speakers (`int`, *optional*):
            Number of speakers. If set to > 1, assume that the speaker ids will be provided as the input and use
            speaker id embedding layer.
        num_languages (`int`, *optional*):
            Number of languages. If set to > 1, assume that the language ids will be provided as the input and use the
            languge id embedding layer.
        speaker_embed_dim (`int`, *optional*):
            Speaker embedding dimension. If set to > 0, assume that speaker_embedding will be provided as the input.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Specifies whether the model is an encoder-decoder.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    attribute_map = ...
    def __init__(self, hidden_size=..., vocab_size=..., num_mel_bins=..., encoder_num_attention_heads=..., encoder_layers=..., encoder_linear_units=..., decoder_layers=..., decoder_num_attention_heads=..., decoder_linear_units=..., speech_decoder_postnet_layers=..., speech_decoder_postnet_units=..., speech_decoder_postnet_kernel=..., positionwise_conv_kernel_size=..., encoder_normalize_before=..., decoder_normalize_before=..., encoder_concat_after=..., decoder_concat_after=..., reduction_factor=..., speaking_speed=..., use_macaron_style_in_conformer=..., use_cnn_in_conformer=..., encoder_kernel_size=..., decoder_kernel_size=..., duration_predictor_layers=..., duration_predictor_channels=..., duration_predictor_kernel_size=..., energy_predictor_layers=..., energy_predictor_channels=..., energy_predictor_kernel_size=..., energy_predictor_dropout=..., energy_embed_kernel_size=..., energy_embed_dropout=..., stop_gradient_from_energy_predictor=..., pitch_predictor_layers=..., pitch_predictor_channels=..., pitch_predictor_kernel_size=..., pitch_predictor_dropout=..., pitch_embed_kernel_size=..., pitch_embed_dropout=..., stop_gradient_from_pitch_predictor=..., encoder_dropout_rate=..., encoder_positional_dropout_rate=..., encoder_attention_dropout_rate=..., decoder_dropout_rate=..., decoder_positional_dropout_rate=..., decoder_attention_dropout_rate=..., duration_predictor_dropout_rate=..., speech_decoder_postnet_dropout=..., max_source_positions=..., use_masking=..., use_weighted_masking=..., num_speakers=..., num_languages=..., speaker_embed_dim=..., is_encoder_decoder=..., **kwargs) -> None:
        ...



class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, model_in_dim=..., upsample_initial_channel=..., upsample_rates=..., upsample_kernel_sizes=..., resblock_kernel_sizes=..., resblock_dilation_sizes=..., initializer_range=..., leaky_relu_slope=..., normalize_before=..., **kwargs) -> None:
        ...



class FastSpeech2ConformerWithHifiGanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerWithHifiGan`]. It is used to
    instantiate a `FastSpeech2ConformerWithHifiGanModel` model according to the specified sub-models configurations,
    defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2ConformerModel [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer) and
    FastSpeech2ConformerHifiGan
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_config (`typing.Dict`, *optional*):
            Configuration of the text-to-speech model.
        vocoder_config (`typing.Dict`, *optional*):
            Configuration of the vocoder model.
    model_config ([`FastSpeech2ConformerConfig`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = ...
    is_composition = ...
    def __init__(self, model_config: Dict = ..., vocoder_config: Dict = ..., **kwargs) -> None:
        ...
