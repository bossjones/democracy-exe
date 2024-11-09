"""
This type stub file was generated by pyright.
"""

from collections import OrderedDict

from torch import Tensor, nn

logger = ...
class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """
    def __init__(self) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: Tensor) -> Tensor:
        ...



class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self, use_gelu_python: bool = ...) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        ...



class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        ...



class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """
    def __init__(self, min: float, max: float) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...



class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """
    def __init__(self) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    def __init__(self) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        ...



class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """
    def forward(self, input: Tensor) -> Tensor:
        ...



class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """
    def forward(self, input, mu=..., sigma=...): # -> Tensor:
        ...



class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """
    def forward(self, input): # -> Tensor:
        ...



class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        ...



ACT2CLS = ...
ACT2FN = ...
def get_activation(activation_string):
    ...

gelu_python = ...
gelu_new = ...
gelu = ...
gelu_fast = ...
quick_gelu = ...
silu = ...
mish = ...
linear_act = ...
