"""
This type stub file was generated by pyright.
"""

from typing import Optional
from pinecone.core.openapi.shared.configuration import Configuration as OpenApiConfiguration

TCP_KEEPINTVL = ...
TCP_KEEPIDLE = ...
TCP_KEEPCNT = ...
class OpenApiConfigFactory:
    @classmethod
    def build(cls, api_key: str, host: Optional[str] = ..., **kwargs): # -> Configuration:
        ...
    
    @classmethod
    def copy(cls, openapi_config: OpenApiConfiguration, api_key: str, host: str) -> OpenApiConfiguration:
        """
        Copy a user-supplied openapi configuration and update it with the user's api key and host.
        If they have not specified other socket configuration, we will use the default values.
        We expect these objects are being passed mainly a vehicle for proxy configuration, so
        we don't modify those settings.
        """
        ...
    


