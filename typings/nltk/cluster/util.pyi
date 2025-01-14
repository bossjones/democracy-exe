"""
This type stub file was generated by pyright.
"""

from abc import abstractmethod
from nltk.cluster.api import ClusterI

class VectorSpaceClusterer(ClusterI):
    """
    Abstract clusterer which takes tokens and maps them into a vector space.
    Optionally performs singular value decomposition to reduce the
    dimensionality.
    """
    def __init__(self, normalise=..., svd_dimensions=...) -> None:
        """
        :param normalise:       should vectors be normalised to length 1
        :type normalise:        boolean
        :param svd_dimensions:  number of dimensions to use in reducing vector
                                dimensionsionality with SVD
        :type svd_dimensions:   int
        """
        ...
    
    def cluster(self, vectors, assign_clusters=..., trace=...): # -> list[None] | None:
        ...
    
    @abstractmethod
    def cluster_vectorspace(self, vectors, trace): # -> None:
        """
        Finds the clusters using the given set of vectors.
        """
        ...
    
    def classify(self, vector): # -> None:
        ...
    
    @abstractmethod
    def classify_vectorspace(self, vector): # -> None:
        """
        Returns the index of the appropriate cluster for the vector.
        """
        ...
    
    def likelihood(self, vector, label): # -> float:
        ...
    
    def likelihood_vectorspace(self, vector, cluster): # -> float:
        """
        Returns the likelihood of the vector belonging to the cluster.
        """
        ...
    
    def vector(self, vector): # -> Any:
        """
        Returns the vector after normalisation and dimensionality reduction
        """
        ...
    


def euclidean_distance(u, v): # -> float:
    """
    Returns the euclidean distance between vectors u and v. This is equivalent
    to the length of the vector (u - v).
    """
    ...

def cosine_distance(u, v): # -> Any:
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to ``1 - (u.v / |u||v|)``.
    """
    ...

class _DendrogramNode:
    """Tree node of a dendrogram."""
    def __init__(self, value, *children) -> None:
        ...
    
    def leaves(self, values=...): # -> list[Any] | list[Self]:
        ...
    
    def groups(self, n): # -> list[Any]:
        ...
    
    def __lt__(self, comparator) -> bool:
        ...
    


class Dendrogram:
    """
    Represents a dendrogram, a tree with a specified branching order.  This
    must be initialised with the leaf items, then iteratively call merge for
    each branch. This class constructs a tree representing the order of calls
    to the merge function.
    """
    def __init__(self, items=...) -> None:
        """
        :param  items: the items at the leaves of the dendrogram
        :type   items: sequence of (any)
        """
        ...
    
    def merge(self, *indices): # -> None:
        """
        Merges nodes at given indices in the dendrogram. The nodes will be
        combined which then replaces the first node specified. All other nodes
        involved in the merge will be removed.

        :param  indices: indices of the items to merge (at least two)
        :type   indices: seq of int
        """
        ...
    
    def groups(self, n): # -> list[Any]:
        """
        Finds the n-groups of items (leaves) reachable from a cut at depth n.
        :param  n: number of groups
        :type   n: int
        """
        ...
    
    def show(self, leaf_labels=...): # -> None:
        """
        Print the dendrogram in ASCII art to standard out.

        :param leaf_labels: an optional list of strings to use for labeling the
                            leaves
        :type leaf_labels: list
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


