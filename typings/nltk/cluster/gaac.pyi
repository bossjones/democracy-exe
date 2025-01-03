"""
This type stub file was generated by pyright.
"""

from nltk.cluster.util import VectorSpaceClusterer

class GAAClusterer(VectorSpaceClusterer):
    """
    The Group Average Agglomerative starts with each of the N vectors as singleton
    clusters. It then iteratively merges pairs of clusters which have the
    closest centroids.  This continues until there is only one cluster. The
    order of merges gives rise to a dendrogram: a tree with the earlier merges
    lower than later merges. The membership of a given number of clusters c, 1
    <= c <= N, can be found by cutting the dendrogram at depth c.

    This clusterer uses the cosine similarity metric only, which allows for
    efficient speed-up in the clustering process.
    """
    def __init__(self, num_clusters=..., normalise=..., svd_dimensions=...) -> None:
        ...
    
    def cluster(self, vectors, assign_clusters=..., trace=...): # -> list[None] | None:
        ...
    
    def cluster_vectorspace(self, vectors, trace=...): # -> None:
        ...
    
    def update_clusters(self, num_clusters): # -> None:
        ...
    
    def classify_vectorspace(self, vector): # -> int:
        ...
    
    def dendrogram(self): # -> Dendrogram | None:
        """
        :return: The dendrogram representing the current clustering
        :rtype:  Dendrogram
        """
        ...
    
    def num_clusters(self): # -> int:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def demo(): # -> None:
    """
    Non-interactive demonstration of the clusterers with simple 2-D data.
    """
    ...

if __name__ == "__main__":
    ...
