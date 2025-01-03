"""
This type stub file was generated by pyright.
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import google.protobuf.struct_pb2
import typing

"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Generated file - DO NOT EDIT
This file is generated from the pinecone-io/apis repo.
Any changes made directly here WILL be overwritten.
"""
DESCRIPTOR: google.protobuf.descriptor.FileDescriptor
@typing.final
class SparseValues(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INDICES_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    @property
    def indices(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        ...
    
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        ...
    
    def __init__(self, *, indices: collections.abc.Iterable[builtins.int] | None = ..., values: collections.abc.Iterable[builtins.float] | None = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["indices", b"indices", "values", b"values"]) -> None:
        ...
    


global___SparseValues = SparseValues
@typing.final
class Vector(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    SPARSE_VALUES_FIELD_NUMBER: builtins.int
    METADATA_FIELD_NUMBER: builtins.int
    id: builtins.str
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """This is the vector data included in the request."""
        ...
    
    @property
    def sparse_values(self) -> global___SparseValues:
        ...
    
    @property
    def metadata(self) -> google.protobuf.struct_pb2.Struct:
        """This is the metadata included in the request."""
        ...
    
    def __init__(self, *, id: builtins.str = ..., values: collections.abc.Iterable[builtins.float] | None = ..., sparse_values: global___SparseValues | None = ..., metadata: google.protobuf.struct_pb2.Struct | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["metadata", b"metadata", "sparse_values", b"sparse_values"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["id", b"id", "metadata", b"metadata", "sparse_values", b"sparse_values", "values", b"values",]) -> None:
        ...
    


global___Vector = Vector
@typing.final
class ScoredVector(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    SCORE_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    SPARSE_VALUES_FIELD_NUMBER: builtins.int
    METADATA_FIELD_NUMBER: builtins.int
    id: builtins.str
    score: builtins.float
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """This is the vector data, if it is requested."""
        ...
    
    @property
    def sparse_values(self) -> global___SparseValues:
        """This is the sparse data, if it is requested."""
        ...
    
    @property
    def metadata(self) -> google.protobuf.struct_pb2.Struct:
        """This is the metadata, if it is requested."""
        ...
    
    def __init__(self, *, id: builtins.str = ..., score: builtins.float = ..., values: collections.abc.Iterable[builtins.float] | None = ..., sparse_values: global___SparseValues | None = ..., metadata: google.protobuf.struct_pb2.Struct | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["metadata", b"metadata", "sparse_values", b"sparse_values"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["id", b"id", "metadata", b"metadata", "score", b"score", "sparse_values", b"sparse_values", "values", b"values",]) -> None:
        ...
    


global___ScoredVector = ScoredVector
@typing.final
class RequestUnion(google.protobuf.message.Message):
    """This is a container to hold mutating vector requests. This is not actually used
    in any public APIs.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    UPSERT_FIELD_NUMBER: builtins.int
    DELETE_FIELD_NUMBER: builtins.int
    UPDATE_FIELD_NUMBER: builtins.int
    @property
    def upsert(self) -> global___UpsertRequest:
        ...
    
    @property
    def delete(self) -> global___DeleteRequest:
        ...
    
    @property
    def update(self) -> global___UpdateRequest:
        ...
    
    def __init__(self, *, upsert: global___UpsertRequest | None = ..., delete: global___DeleteRequest | None = ..., update: global___UpdateRequest | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["RequestUnionInner", b"RequestUnionInner", "delete", b"delete", "update", b"update", "upsert", b"upsert",]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["RequestUnionInner", b"RequestUnionInner", "delete", b"delete", "update", b"update", "upsert", b"upsert",]) -> None:
        ...
    
    def WhichOneof(self, oneof_group: typing.Literal["RequestUnionInner", b"RequestUnionInner"]) -> typing.Literal["upsert", "delete", "update"] | None:
        ...
    


global___RequestUnion = RequestUnion
@typing.final
class UpsertRequest(google.protobuf.message.Message):
    """The request for the `upsert` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VECTORS_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def vectors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Vector]:
        """An array containing the vectors to upsert. Recommended batch limit is 100 vectors."""
        ...
    
    def __init__(self, *, vectors: collections.abc.Iterable[global___Vector] | None = ..., namespace: builtins.str = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["namespace", b"namespace", "vectors", b"vectors"]) -> None:
        ...
    


global___UpsertRequest = UpsertRequest
@typing.final
class UpsertResponse(google.protobuf.message.Message):
    """The response for the `upsert` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    UPSERTED_COUNT_FIELD_NUMBER: builtins.int
    upserted_count: builtins.int
    def __init__(self, *, upserted_count: builtins.int = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["upserted_count", b"upserted_count"]) -> None:
        ...
    


global___UpsertResponse = UpsertResponse
@typing.final
class DeleteRequest(google.protobuf.message.Message):
    """The request for the `Delete` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    IDS_FIELD_NUMBER: builtins.int
    DELETE_ALL_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    FILTER_FIELD_NUMBER: builtins.int
    delete_all: builtins.bool
    namespace: builtins.str
    @property
    def ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Vectors to delete."""
        ...
    
    @property
    def filter(self) -> google.protobuf.struct_pb2.Struct:
        """If specified, the metadata filter here will be used to select the vectors to delete. This is mutually exclusive
        with specifying ids to delete in the ids param or using delete_all=True.
        See https://www.pinecone.io/docs/metadata-filtering/.
        """
        ...
    
    def __init__(self, *, ids: collections.abc.Iterable[builtins.str] | None = ..., delete_all: builtins.bool = ..., namespace: builtins.str = ..., filter: google.protobuf.struct_pb2.Struct | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["filter", b"filter"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["delete_all", b"delete_all", "filter", b"filter", "ids", b"ids", "namespace", b"namespace",]) -> None:
        ...
    


global___DeleteRequest = DeleteRequest
@typing.final
class DeleteResponse(google.protobuf.message.Message):
    """The response for the `Delete` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    def __init__(self) -> None:
        ...
    


global___DeleteResponse = DeleteResponse
@typing.final
class FetchRequest(google.protobuf.message.Message):
    """The request for the `fetch` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    IDS_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """The vector IDs to fetch. Does not accept values containing spaces."""
        ...
    
    def __init__(self, *, ids: collections.abc.Iterable[builtins.str] | None = ..., namespace: builtins.str = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["ids", b"ids", "namespace", b"namespace"]) -> None:
        ...
    


global___FetchRequest = FetchRequest
@typing.final
class FetchResponse(google.protobuf.message.Message):
    """The response for the `fetch` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    @typing.final
    class VectorsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___Vector:
            ...
        
        def __init__(self, *, key: builtins.str = ..., value: global___Vector | None = ...) -> None:
            ...
        
        def HasField(self, field_name: typing.Literal["value", b"value"]) -> builtins.bool:
            ...
        
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None:
            ...
        
    
    
    VECTORS_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    USAGE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def vectors(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___Vector]:
        """The fetched vectors, in the form of a map between the fetched ids and the fetched vectors"""
        ...
    
    @property
    def usage(self) -> global___Usage:
        """The usage for this operation."""
        ...
    
    def __init__(self, *, vectors: collections.abc.Mapping[builtins.str, global___Vector] | None = ..., namespace: builtins.str = ..., usage: global___Usage | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["_usage", b"_usage", "usage", b"usage"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["_usage", b"_usage", "namespace", b"namespace", "usage", b"usage", "vectors", b"vectors",]) -> None:
        ...
    
    def WhichOneof(self, oneof_group: typing.Literal["_usage", b"_usage"]) -> typing.Literal["usage"] | None:
        ...
    


global___FetchResponse = FetchResponse
@typing.final
class ListRequest(google.protobuf.message.Message):
    """The request for the `list` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    PREFIX_FIELD_NUMBER: builtins.int
    LIMIT_FIELD_NUMBER: builtins.int
    PAGINATION_TOKEN_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    prefix: builtins.str
    limit: builtins.int
    pagination_token: builtins.str
    namespace: builtins.str
    def __init__(self, *, prefix: builtins.str | None = ..., limit: builtins.int | None = ..., pagination_token: builtins.str | None = ..., namespace: builtins.str = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["_limit", b"_limit", "_pagination_token", b"_pagination_token", "_prefix", b"_prefix", "limit", b"limit", "pagination_token", b"pagination_token", "prefix", b"prefix",]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["_limit", b"_limit", "_pagination_token", b"_pagination_token", "_prefix", b"_prefix", "limit", b"limit", "namespace", b"namespace", "pagination_token", b"pagination_token", "prefix", b"prefix",]) -> None:
        ...
    
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_limit", b"_limit"]) -> typing.Literal["limit"] | None:
        ...
    
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_pagination_token", b"_pagination_token"]) -> typing.Literal["pagination_token"] | None:
        ...
    
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_prefix", b"_prefix"]) -> typing.Literal["prefix"] | None:
        ...
    


global___ListRequest = ListRequest
@typing.final
class Pagination(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NEXT_FIELD_NUMBER: builtins.int
    next: builtins.str
    def __init__(self, *, next: builtins.str = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["next", b"next"]) -> None:
        ...
    


global___Pagination = Pagination
@typing.final
class ListItem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    id: builtins.str
    def __init__(self, *, id: builtins.str = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["id", b"id"]) -> None:
        ...
    


global___ListItem = ListItem
@typing.final
class ListResponse(google.protobuf.message.Message):
    """The response for the `List` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VECTORS_FIELD_NUMBER: builtins.int
    PAGINATION_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    USAGE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def vectors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ListItem]:
        """A list of ids"""
        ...
    
    @property
    def pagination(self) -> global___Pagination:
        """Pagination token to continue past this listing"""
        ...
    
    @property
    def usage(self) -> global___Usage:
        """The usage for this operation."""
        ...
    
    def __init__(self, *, vectors: collections.abc.Iterable[global___ListItem] | None = ..., pagination: global___Pagination | None = ..., namespace: builtins.str = ..., usage: global___Usage | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["_pagination", b"_pagination", "_usage", b"_usage", "pagination", b"pagination", "usage", b"usage",]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["_pagination", b"_pagination", "_usage", b"_usage", "namespace", b"namespace", "pagination", b"pagination", "usage", b"usage", "vectors", b"vectors",]) -> None:
        ...
    
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_pagination", b"_pagination"]) -> typing.Literal["pagination"] | None:
        ...
    
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_usage", b"_usage"]) -> typing.Literal["usage"] | None:
        ...
    


global___ListResponse = ListResponse
@typing.final
class QueryVector(google.protobuf.message.Message):
    """A single query vector within a `QueryRequest`."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VALUES_FIELD_NUMBER: builtins.int
    SPARSE_VALUES_FIELD_NUMBER: builtins.int
    TOP_K_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    FILTER_FIELD_NUMBER: builtins.int
    top_k: builtins.int
    namespace: builtins.str
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """The query vector values. This should be the same length as the dimension of the index being queried."""
        ...
    
    @property
    def sparse_values(self) -> global___SparseValues:
        """The query sparse values."""
        ...
    
    @property
    def filter(self) -> google.protobuf.struct_pb2.Struct:
        """An override for the metadata filter to apply. This replaces the request-level filter."""
        ...
    
    def __init__(self, *, values: collections.abc.Iterable[builtins.float] | None = ..., sparse_values: global___SparseValues | None = ..., top_k: builtins.int = ..., namespace: builtins.str = ..., filter: google.protobuf.struct_pb2.Struct | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["filter", b"filter", "sparse_values", b"sparse_values"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["filter", b"filter", "namespace", b"namespace", "sparse_values", b"sparse_values", "top_k", b"top_k", "values", b"values",]) -> None:
        ...
    


global___QueryVector = QueryVector
@typing.final
class QueryRequest(google.protobuf.message.Message):
    """The request for the `query` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAMESPACE_FIELD_NUMBER: builtins.int
    TOP_K_FIELD_NUMBER: builtins.int
    FILTER_FIELD_NUMBER: builtins.int
    INCLUDE_VALUES_FIELD_NUMBER: builtins.int
    INCLUDE_METADATA_FIELD_NUMBER: builtins.int
    QUERIES_FIELD_NUMBER: builtins.int
    VECTOR_FIELD_NUMBER: builtins.int
    SPARSE_VECTOR_FIELD_NUMBER: builtins.int
    ID_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    top_k: builtins.int
    include_values: builtins.bool
    include_metadata: builtins.bool
    id: builtins.str
    @property
    def filter(self) -> google.protobuf.struct_pb2.Struct:
        """The filter to apply. You can use vector metadata to limit your search. See https://www.pinecone.io/docs/metadata-filtering/."""
        ...
    
    @property
    def queries(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___QueryVector]:
        """DEPRECATED. The query vectors. Each `query()` request can contain only one of the parameters `queries`, `vector`, or  `id`."""
        ...
    
    @property
    def vector(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """The query vector. This should be the same length as the dimension of the index being queried. Each `query()` request can contain only one of the parameters `id` or `vector`."""
        ...
    
    @property
    def sparse_vector(self) -> global___SparseValues:
        """The query sparse values."""
        ...
    
    def __init__(self, *, namespace: builtins.str = ..., top_k: builtins.int = ..., filter: google.protobuf.struct_pb2.Struct | None = ..., include_values: builtins.bool = ..., include_metadata: builtins.bool = ..., queries: collections.abc.Iterable[global___QueryVector] | None = ..., vector: collections.abc.Iterable[builtins.float] | None = ..., sparse_vector: global___SparseValues | None = ..., id: builtins.str = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["filter", b"filter", "sparse_vector", b"sparse_vector"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["filter", b"filter", "id", b"id", "include_metadata", b"include_metadata", "include_values", b"include_values", "namespace", b"namespace", "queries", b"queries", "sparse_vector", b"sparse_vector", "top_k", b"top_k", "vector", b"vector",]) -> None:
        ...
    


global___QueryRequest = QueryRequest
@typing.final
class SingleQueryResults(google.protobuf.message.Message):
    """The query results for a single `QueryVector`"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MATCHES_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def matches(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ScoredVector]:
        """The matches for the vectors."""
        ...
    
    def __init__(self, *, matches: collections.abc.Iterable[global___ScoredVector] | None = ..., namespace: builtins.str = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["matches", b"matches", "namespace", b"namespace"]) -> None:
        ...
    


global___SingleQueryResults = SingleQueryResults
@typing.final
class QueryResponse(google.protobuf.message.Message):
    """The response for the `query` operation. These are the matches found for a particular query vector. The matches are ordered from most similar to least similar."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    RESULTS_FIELD_NUMBER: builtins.int
    MATCHES_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    USAGE_FIELD_NUMBER: builtins.int
    namespace: builtins.str
    @property
    def results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SingleQueryResults]:
        """DEPRECATED. The results of each query. The order is the same as `QueryRequest.queries`."""
        ...
    
    @property
    def matches(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ScoredVector]:
        """The matches for the vectors."""
        ...
    
    @property
    def usage(self) -> global___Usage:
        """The usage for this operation."""
        ...
    
    def __init__(self, *, results: collections.abc.Iterable[global___SingleQueryResults] | None = ..., matches: collections.abc.Iterable[global___ScoredVector] | None = ..., namespace: builtins.str = ..., usage: global___Usage | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["_usage", b"_usage", "usage", b"usage"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["_usage", b"_usage", "matches", b"matches", "namespace", b"namespace", "results", b"results", "usage", b"usage",]) -> None:
        ...
    
    def WhichOneof(self, oneof_group: typing.Literal["_usage", b"_usage"]) -> typing.Literal["usage"] | None:
        ...
    


global___QueryResponse = QueryResponse
@typing.final
class Usage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    READ_UNITS_FIELD_NUMBER: builtins.int
    read_units: builtins.int
    def __init__(self, *, read_units: builtins.int | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["_read_units", b"_read_units", "read_units", b"read_units"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["_read_units", b"_read_units", "read_units", b"read_units"]) -> None:
        ...
    
    def WhichOneof(self, oneof_group: typing.Literal["_read_units", b"_read_units"]) -> typing.Literal["read_units"] | None:
        ...
    


global___Usage = Usage
@typing.final
class UpdateRequest(google.protobuf.message.Message):
    """The request for the `update` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    SPARSE_VALUES_FIELD_NUMBER: builtins.int
    SET_METADATA_FIELD_NUMBER: builtins.int
    NAMESPACE_FIELD_NUMBER: builtins.int
    id: builtins.str
    namespace: builtins.str
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """Vector data."""
        ...
    
    @property
    def sparse_values(self) -> global___SparseValues:
        ...
    
    @property
    def set_metadata(self) -> google.protobuf.struct_pb2.Struct:
        """Metadata to *set* for the vector."""
        ...
    
    def __init__(self, *, id: builtins.str = ..., values: collections.abc.Iterable[builtins.float] | None = ..., sparse_values: global___SparseValues | None = ..., set_metadata: google.protobuf.struct_pb2.Struct | None = ..., namespace: builtins.str = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["set_metadata", b"set_metadata", "sparse_values", b"sparse_values"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["id", b"id", "namespace", b"namespace", "set_metadata", b"set_metadata", "sparse_values", b"sparse_values", "values", b"values",]) -> None:
        ...
    


global___UpdateRequest = UpdateRequest
@typing.final
class UpdateResponse(google.protobuf.message.Message):
    """The response for the `update` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    def __init__(self) -> None:
        ...
    


global___UpdateResponse = UpdateResponse
@typing.final
class DescribeIndexStatsRequest(google.protobuf.message.Message):
    """The request for the `describe_index_stats` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    FILTER_FIELD_NUMBER: builtins.int
    @property
    def filter(self) -> google.protobuf.struct_pb2.Struct:
        """If this parameter is present, the operation only returns statistics
        for vectors that satisfy the filter.
        See https://www.pinecone.io/docs/metadata-filtering/.
        """
        ...
    
    def __init__(self, *, filter: google.protobuf.struct_pb2.Struct | None = ...) -> None:
        ...
    
    def HasField(self, field_name: typing.Literal["filter", b"filter"]) -> builtins.bool:
        ...
    
    def ClearField(self, field_name: typing.Literal["filter", b"filter"]) -> None:
        ...
    


global___DescribeIndexStatsRequest = DescribeIndexStatsRequest
@typing.final
class NamespaceSummary(google.protobuf.message.Message):
    """A summary of the contents of a namespace."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VECTOR_COUNT_FIELD_NUMBER: builtins.int
    vector_count: builtins.int
    def __init__(self, *, vector_count: builtins.int = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["vector_count", b"vector_count"]) -> None:
        ...
    


global___NamespaceSummary = NamespaceSummary
@typing.final
class DescribeIndexStatsResponse(google.protobuf.message.Message):
    """The response for the `describe_index_stats` operation."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    @typing.final
    class NamespacesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___NamespaceSummary:
            ...
        
        def __init__(self, *, key: builtins.str = ..., value: global___NamespaceSummary | None = ...) -> None:
            ...
        
        def HasField(self, field_name: typing.Literal["value", b"value"]) -> builtins.bool:
            ...
        
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None:
            ...
        
    
    
    NAMESPACES_FIELD_NUMBER: builtins.int
    DIMENSION_FIELD_NUMBER: builtins.int
    INDEX_FULLNESS_FIELD_NUMBER: builtins.int
    TOTAL_VECTOR_COUNT_FIELD_NUMBER: builtins.int
    dimension: builtins.int
    index_fullness: builtins.float
    total_vector_count: builtins.int
    @property
    def namespaces(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___NamespaceSummary]:
        """A mapping for each namespace in the index from the namespace name to a
        summary of its contents. If a metadata filter expression is present, the
        summary will reflect only vectors matching that expression.
        """
        ...
    
    def __init__(self, *, namespaces: collections.abc.Mapping[builtins.str, global___NamespaceSummary] | None = ..., dimension: builtins.int = ..., index_fullness: builtins.float = ..., total_vector_count: builtins.int = ...) -> None:
        ...
    
    def ClearField(self, field_name: typing.Literal["dimension", b"dimension", "index_fullness", b"index_fullness", "namespaces", b"namespaces", "total_vector_count", b"total_vector_count",]) -> None:
        ...
    


global___DescribeIndexStatsResponse = DescribeIndexStatsResponse
