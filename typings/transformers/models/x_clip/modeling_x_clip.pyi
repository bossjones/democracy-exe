"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig

"""PyTorch X-CLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def x_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class XCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity.
        logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
        video_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`].
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_video: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    video_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    mit_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class XCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: XCLIPVisionConfig) -> None:
        ...
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...
    


class XCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: XCLIPTextConfig) -> None:
        ...
    
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...
    


class XCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class XCLIPMLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class XCLIPEncoderLayer(nn.Module):
    def __init__(self, config: XCLIPConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class XCLIPDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class XCLIPVisionEncoderLayer(nn.Module):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """
    def __init__(self, config: XCLIPConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class XCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XCLIPConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


X_CLIP_START_DOCSTRING = ...
X_CLIP_TEXT_INPUTS_DOCSTRING = ...
X_CLIP_VISION_INPUTS_DOCSTRING = ...
X_CLIP_INPUTS_DOCSTRING = ...
class XCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPEncoderLayer`].

    Args:
        config: XCLIPConfig
    """
    def __init__(self, config: XCLIPConfig) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    


class XCLIPTextTransformer(nn.Module):
    def __init__(self, config: XCLIPTextConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...
    


class XCLIPTextModel(XCLIPPreTrainedModel):
    config_class = XCLIPTextConfig
    def __init__(self, config: XCLIPTextConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, XCLIPTextModel

        >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        ...
    


class XCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPVisionEncoderLayer`].

    Args:
        config: XCLIPConfig
    """
    def __init__(self, config: XCLIPConfig) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    


class XCLIPVisionTransformer(nn.Module):
    """
    This corresponds to the `CrossFrameCommunicationTransformer` class in the original implementation.
    """
    def __init__(self, config: XCLIPVisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...
    


class XCLIPVisionModel(XCLIPPreTrainedModel):
    config_class = XCLIPVisionConfig
    main_input_name = ...
    def __init__(self, config: XCLIPVisionConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, XCLIPVisionModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 16 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

        >>> batch_size, num_frames, num_channels, height, width = pixel_values.shape
        >>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        >>> outputs = model(pixel_values)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        ...
    


class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """
    def __init__(self, config: XCLIPVisionConfig) -> None:
        ...
    
    def forward(self, hidden_states, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...
    


class XCLIPCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, queries, keys, values): # -> Any:
        """Input shape: Batch x Time x Channel"""
        ...
    


class PromptGeneratorLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, x, visual):
        ...
    


class XCLIPPromptGenerator(nn.Module):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, text, visual): # -> Any:
        ...
    


@add_start_docstrings(X_CLIP_START_DOCSTRING)
class XCLIPModel(XCLIPPreTrainedModel):
    config_class = XCLIPConfig
    def __init__(self, config: XCLIPConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    def get_video_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            video_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The video embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPVisionModel`] and
            [`XCLIPMultiframeIntegrationTransformer`].

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, AutoModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = processor(videos=list(video), return_tensors="pt")

        >>> video_features = model.get_video_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(X_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XCLIPOutput, config_class=XCLIPConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, XCLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, AutoModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = processor(
        ...     text=["playing sports", "eating spaghetti", "go shopping"],
        ...     videos=list(video),
        ...     return_tensors="pt",
        ...     padding=True,
        ... )

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
        >>> probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
        >>> print(probs)
        tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])
        ```"""
        ...
    


