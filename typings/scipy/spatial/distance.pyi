"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Literal, Protocol, SupportsFloat, SupportsIndex, overload
from numpy.typing import ArrayLike, NDArray

_FloatValue = None | str | bytes | SupportsFloat | SupportsIndex
class _MetricCallback1(Protocol):
    def __call__(self, __XA: NDArray[Any], __XB: NDArray[Any]) -> _FloatValue:
        ...
    


class _MetricCallback2(Protocol):
    def __call__(self, __XA: NDArray[Any], __XB: NDArray[Any], **kwargs: Any) -> _FloatValue:
        ...
    


_MetricCallback = _MetricCallback1 | _MetricCallback2
_MetricKind = Literal['braycurtis', 'canberra', 'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch', 'cityblock', 'cblock', 'cb', 'c', 'correlation', 'co', 'cosine', 'cos', 'dice', 'euclidean', 'euclid', 'eu', 'e', 'hamming', 'hamm', 'ha', 'h', 'minkowski', 'mi', 'm', 'pnorm', 'jaccard', 'jacc', 'ja', 'j', 'jensenshannon', 'js', 'kulczynski1', 'mahalanobis', 'mahal', 'mah', 'rogerstanimoto', 'russellrao', 'seuclidean', 'se', 's', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'sqe', 'sqeuclid', 'yule',]
def braycurtis(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def canberra(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

@overload
def cdist(XA: ArrayLike, XB: ArrayLike, metric: _MetricKind = ..., *, out: None | NDArray[np.floating[Any]] = ..., p: float = ..., w: ArrayLike | None = ..., V: ArrayLike | None = ..., VI: ArrayLike | None = ...) -> NDArray[np.floating[Any]]:
    ...

@overload
def cdist(XA: ArrayLike, XB: ArrayLike, metric: _MetricCallback, *, out: None | NDArray[np.floating[Any]] = ..., **kwargs: Any) -> NDArray[np.floating[Any]]:
    ...

def chebyshev(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> Any:
    ...

def cityblock(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> Any:
    ...

def correlation(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ..., centered: bool = ...) -> np.float64:
    ...

def cosine(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def dice(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

def directed_hausdorff(u: ArrayLike, v: ArrayLike, seed: int | None = ...) -> tuple[float, int, int]:
    ...

def euclidean(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

def hamming(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def is_valid_dm(D: ArrayLike, tol: float = ..., throw: bool = ..., name: str | None = ..., warning: bool = ...) -> bool:
    ...

def is_valid_y(y: ArrayLike, warning: bool = ..., throw: bool = ..., name: str | None = ...) -> bool:
    ...

def jaccard(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def jensenshannon(p: ArrayLike, q: ArrayLike, base: float | None = ...) -> np.float64:
    ...

def kulczynski1(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def mahalanobis(u: ArrayLike, v: ArrayLike, VI: ArrayLike) -> np.float64:
    ...

def minkowski(u: ArrayLike, v: ArrayLike, p: float = ..., w: ArrayLike | None = ...) -> float:
    ...

def num_obs_dm(d: ArrayLike) -> int:
    ...

def num_obs_y(Y: ArrayLike) -> int:
    ...

@overload
def pdist(X: ArrayLike, metric: _MetricKind = ..., *, out: None | NDArray[np.floating[Any]] = ..., p: float = ..., w: ArrayLike | None = ..., V: ArrayLike | None = ..., VI: ArrayLike | None = ...) -> NDArray[np.floating[Any]]:
    ...

@overload
def pdist(X: ArrayLike, metric: _MetricCallback, *, out: None | NDArray[np.floating[Any]] = ..., **kwargs: Any) -> NDArray[np.floating[Any]]:
    ...

def seuclidean(u: ArrayLike, v: ArrayLike, V: ArrayLike) -> float:
    ...

def sokalmichener(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

def sokalsneath(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def sqeuclidean(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> np.float64:
    ...

def squareform(X: ArrayLike, force: Literal["no", "tomatrix", "tovector"] = ..., checks: bool = ...) -> NDArray[Any]:
    ...

def rogerstanimoto(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

def russellrao(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

def yule(u: ArrayLike, v: ArrayLike, w: ArrayLike | None = ...) -> float:
    ...

