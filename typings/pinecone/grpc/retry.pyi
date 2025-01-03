"""
This type stub file was generated by pyright.
"""

import abc
import grpc
from typing import NamedTuple, Optional, Tuple

_logger = ...
class SleepPolicy(abc.ABC):
    @abc.abstractmethod
    def sleep(self, try_i: int): # -> None:
        """
        How long to sleep in milliseconds.
        :param try_i: the number of retry (starting from zero)
        """
        ...
    


class ExponentialBackoff(SleepPolicy):
    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int, multiplier: int) -> None:
        ...
    
    def sleep(self, try_i: int): # -> None:
        ...
    


class RetryOnRpcErrorClientInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor, grpc.StreamUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    """gRPC retry.

    Referece: https://github.com/grpc/grpc/issues/19514#issuecomment-531700657
    """
    def __init__(self, retry_config: RetryConfig) -> None:
        ...
    
    def intercept_unary_unary(self, continuation, client_call_details, request): # -> None:
        ...
    
    def intercept_unary_stream(self, continuation, client_call_details, request): # -> None:
        ...
    
    def intercept_stream_unary(self, continuation, client_call_details, request_iterator): # -> None:
        ...
    
    def intercept_stream_stream(self, continuation, client_call_details, request_iterator): # -> None:
        ...
    


class RetryConfig(NamedTuple):
    """Config settings related to retry"""
    max_attempts: int = ...
    sleep_policy: SleepPolicy = ...
    retryable_status: Optional[Tuple[grpc.StatusCode, ...]] = ...


