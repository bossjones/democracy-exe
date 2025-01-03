"""
This type stub file was generated by pyright.
"""

from nltk.cluster.util import VectorSpaceClusterer

class KMeansClusterer(VectorSpaceClusterer):
    """
    The K-means clusterer starts with k arbitrary chosen means then allocates
    each vector to the cluster with the closest mean. It then recalculates the
    means of each cluster as the centroid of the vectors in the cluster. This
    process repeats until the cluster memberships stabilise. This is a
    hill-climbing algorithm which may converge to a local maximum. Hence the
    clustering is often repeated with random initial means and the most
    commonly occurring output means are chosen.
    """
    def __init__(self, num_means, distance, repeats=..., conv_test=..., initial_means=..., normalise=..., svd_dimensions=..., rng=..., avoid_empty_clusters=...) -> None:
        """
        :param  num_means:  the number of means to use (may use fewer)
        :type   num_means:  int
        :param  distance:   measure of distance between two vectors
        :type   distance:   function taking two vectors and returning a float
        :param  repeats:    number of randomised clustering trials to use
        :type   repeats:    int
        :param  conv_test:  maximum variation in mean differences before
                            deemed convergent
        :type   conv_test:  number
        :param  initial_means: set of k initial means
        :type   initial_means: sequence of vectors
        :param  normalise:  should vectors be normalised to length 1
        :type   normalise:  boolean
        :param svd_dimensions: number of dimensions to use in reducing vector
                               dimensionsionality with SVD
        :type svd_dimensions: int
        :param  rng:        random number generator (or None)
        :type   rng:        Random
        :param avoid_empty_clusters: include current centroid in computation
                                     of next one; avoids undefined behavior
                                     when clusters become empty
        :type avoid_empty_clusters: boolean
        """
        ...
    
    def cluster_vectorspace(self, vectors, trace=...): # -> None:
        ...
    
    def classify_vectorspace(self, vector): # -> int | None:
        ...
    
    def num_clusters(self): # -> int | Any:
        ...
    
    def means(self): # -> list[Any] | None:
        """
        The means used for clustering.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


def demo(): # -> None:
    ...

if __name__ == "__main__":
    ...
