"""
This type stub file was generated by pyright.
"""

from ..utils import is_soundfile_availble, is_torch_available, is_vision_available
from PIL.Image import Image as ImageType
from torch import Tensor

logger = ...
if is_vision_available():
    ...
else:
    ImageType = ...
if is_torch_available():
    ...
else:
    Tensor = ...
if is_soundfile_availble():
    ...
class AgentType:
    """
    Abstract class to be reimplemented to define types that can be returned by agents.

    These objects serve three purposes:

    - They behave as they were the type they're meant to be, e.g., a string for text, a PIL.Image for images
    - They can be stringified: str(object) in order to return a string defining the object
    - They should be displayed correctly in ipython notebooks/colab/jupyter
    """
    def __init__(self, value) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def to_raw(self): # -> Any:
        ...
    
    def to_string(self) -> str:
        ...
    


class AgentText(AgentType, str):
    """
    Text type returned by the agent. Behaves as a string.
    """
    def to_raw(self): # -> Any:
        ...
    
    def to_string(self): # -> str:
        ...
    


class AgentImage(AgentType, ImageType):
    """
    Image type returned by the agent. Behaves as a PIL.Image.
    """
    def __init__(self, value) -> None:
        ...
    
    def to_raw(self): # -> Image | object | ImageFile | None:
        """
        Returns the "raw" version of that object. In the case of an AgentImage, it is a PIL.Image.
        """
        ...
    
    def to_string(self): # -> str | Path | None:
        """
        Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
        version of the image.
        """
        ...
    
    def save(self, output_bytes, format, **params): # -> None:
        """
        Saves the image to a file.
        Args:
            output_bytes (bytes): The output bytes to save the image to.
            format (str): The format to use for the output image. The format is the same as in PIL.Image.save.
            **params: Additional parameters to pass to PIL.Image.save.
        """
        ...
    


class AgentAudio(AgentType, str):
    """
    Audio type returned by the agent.
    """
    def __init__(self, value, samplerate=...) -> None:
        ...
    
    def to_raw(self): # -> Tensor | None:
        """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
        ...
    
    def to_string(self): # -> str | Path | None:
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        ...
    


AGENT_TYPE_MAPPING = ...
INSTANCE_TYPE_MAPPING = ...
if is_torch_available():
    ...
def handle_agent_inputs(*args, **kwargs): # -> tuple[list[Any], dict[str, Any]]:
    ...

def handle_agent_outputs(output, output_type=...):
    ...
