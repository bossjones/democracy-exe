"""
This type stub file was generated by pyright.
"""

from typing import Dict, Tuple, Union
from pinecone.core.grpc.protos.vector_service_pb2 import Vector as GRPCVector
from pinecone import Vector as NonGRPCVector

class VectorFactoryGRPC:
    @staticmethod
    def build(item: Union[GRPCVector, NonGRPCVector, Tuple, Dict]) -> GRPCVector:
        ...
    


