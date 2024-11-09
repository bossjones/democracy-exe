"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType

"""Feature extractor class for CLAP."""
logger = ...
class ClapFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLAP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the *Short Time
    Fourier Transform* (STFT) which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 64):
            The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters
            (`n_mels`).
        sampling_rate (`int`, *optional*, defaults to 48000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves
            to warn users if the audio fed to the feature extractor does not have the same sampling rate.
        hop_length (`int`,*optional*, defaults to 480):
            Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split
            in smaller `frames` with a step of `hop_length` between each frame.
        max_length_s (`int`, *optional*, defaults to 10):
            The maximum input length of the model in seconds. This is used to pad the audio.
        fft_window_size (`int`, *optional*, defaults to 1024):
            Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency
            resolution of the spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the attention masks coresponding to the input.
        frequency_min (`float`, *optional*, defaults to 0):
            The lowest frequency of interest. The STFT will not be computed for values below this.
        frequency_max (`float`, *optional*, defaults to 14000):
            The highest frequency of interest. The STFT will not be computed for values above this.
        top_db (`float`, *optional*):
            The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the
            `audio_utils.power_to_db` function
        truncation (`str`, *optional*, defaults to `"fusion"`):
            Truncation pattern for long audio inputs. Two patterns are available:
                - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a
                  downsampled version of the entire mel spectrogram.
            If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy
            of the original mel obtained from the padded audio.
                - `rand_trunc` will select a random crop of the mel spectrogram.
        padding (`str`, *optional*, defaults to `"repeatpad"`):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                - `repeat`: the audio is repeated and then cut to fit the `max_length`
                - `pad`: the audio is padded.
    """
    model_input_names = ...
    def __init__(self, feature_size=..., sampling_rate=..., hop_length=..., max_length_s=..., fft_window_size=..., padding_value=..., return_attention_mask=..., frequency_min: float = ..., frequency_max: float = ..., top_db: int = ..., truncation: str = ..., padding: str = ..., **kwargs) -> None:
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        ...

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], truncation: str = ..., padding: Optional[str] = ..., max_length: Optional[int] = ..., sampling_rate: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`str`, *optional*):
                Truncation pattern for long audio inputs. Two patterns are available:
                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and
                      a downsampled version of the entire mel spectrogram.
                If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a
                copy of the original mel obtained from the padded audio.
                    - `rand_trunc` will select a random crop of the mel spectrogram.
            padding (`str`, *optional*):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                    - `repeat`: the audio is repeated and then cut to fit the `max_length`
                    - `pad`: the audio is padded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
        """
        ...
