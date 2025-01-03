"""
This type stub file was generated by pyright.
"""

from functools import cache

"""Base classes for low memory simplicial complex structures."""
class Complex:
    """
    Base class for a simplicial complex described as a cache of vertices
    together with their connections.

    Important methods:
        Domain triangulation:
                Complex.triangulate, Complex.split_generation
        Triangulating arbitrary points (must be traingulable,
            may exist outside domain):
                Complex.triangulate(sample_set)
        Converting another simplicial complex structure data type to the
            structure used in Complex (ex. OBJ wavefront)
                Complex.convert(datatype, data)

    Important objects:
        HC.V: The cache of vertices and their connection
        HC.H: Storage structure of all vertex groups

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the complex R^dim
    domain : list of tuples, optional
        The bounds [x_l, x_u]^dim of the hyperrectangle space
        ex. The default domain is the hyperrectangle [0, 1]^dim
        Note: The domain must be convex, non-convex spaces can be cut
              away from this domain using the non-linear
              g_cons functions to define any arbitrary domain
              (these domains may also be disconnected from each other)
    sfield :
        A scalar function defined in the associated domain f: R^dim --> R
    sfield_args : tuple
        Additional arguments to be passed to `sfield`
    vfield :
        A scalar function defined in the associated domain
                       f: R^dim --> R^m
                   (for example a gradient function of the scalar field)
    vfield_args : tuple
        Additional arguments to be passed to vfield
    symmetry : None or list
            Specify if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by up to
            O(n!) times in the fully symmetric case.

            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
             x_6 are symmetric to x_4, this can be specified to the solver as:

            symmetry = [0,  # Variable 1
                        0,  # symmetric to variable 1
                        0,  # symmetric to variable 1
                        3,  # Variable 4
                        3,  # symmetric to variable 4
                        3,  # symmetric to variable 4
                        ]

    constraints : dict or sequence of dict, optional
        Constraints definition.
        Function(s) ``R**n`` in the form::

            g(x) <= 0 applied as g : R^n -> R^m
            h(x) == 0 applied as h : R^n -> R^p

        Each constraint is defined in a dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be
        non-negative.constraints : dict or sequence of dict, optional
        Constraints definition.
        Function(s) ``R**n`` in the form::

            g(x) <= 0 applied as g : R^n -> R^m
            h(x) == 0 applied as h : R^n -> R^p

        Each constraint is defined in a dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (unused).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.

    workers : int  optional
        Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
         functions in parallel.
    """
    def __init__(self, dim, domain=..., sfield=..., sfield_args=..., symmetry=..., constraints=..., workers=...) -> None:
        ...
    
    def __call__(self): # -> list[Any]:
        ...
    
    def cyclic_product(self, bounds, origin, supremum, centroid=...):
        """Generate initial triangulation using cyclic product"""
        ...
    
    def triangulate(self, n=..., symmetry=..., centroid=..., printout=...): # -> None:
        """
        Triangulate the initial domain, if n is not None then a limited number
        of points will be generated

        Parameters
        ----------
        n : int, Number of points to be sampled.
        symmetry :

            Ex. Dictionary/hashtable
            f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            symmetry = symmetry[0]: 0,  # Variable 1
                       symmetry[1]: 0,  # symmetric to variable 1
                       symmetry[2]: 0,  # symmetric to variable 1
                       symmetry[3]: 3,  # Variable 4
                       symmetry[4]: 3,  # symmetric to variable 4
                       symmetry[5]: 3,  # symmetric to variable 4
                        }
        centroid : bool, if True add a central point to the hypercube
        printout : bool, if True print out results

        NOTES:
        ------
        Rather than using the combinatorial algorithm to connect vertices we
        make the following observation:

        The bound pairs are similar a C2 cyclic group and the structure is
        formed using the cartesian product:

        H = C2 x C2 x C2 ... x C2 (dim times)

        So construct any normal subgroup N and consider H/N first, we connect
        all vertices within N (ex. N is C2 (the first dimension), then we move
        to a left coset aN (an operation moving around the defined H/N group by
        for example moving from the lower bound in C2 (dimension 2) to the
        higher bound in C2. During this operation connection all the vertices.
        Now repeat the N connections. Note that these elements can be connected
        in parallel.
        """
        ...
    
    def refine(self, n=...): # -> None:
        ...
    
    def refine_all(self, centroids=...): # -> None:
        """Refine the entire domain of the current complex."""
        ...
    
    def refine_local_space(self, origin, supremum, bounds, centroid=...):
        ...
    
    def refine_star(self, v): # -> None:
        """Refine the star domain of a vertex `v`."""
        ...
    
    @cache
    def split_edge(self, v1, v2):
        ...
    
    def vpool(self, origin, supremum): # -> set[Any]:
        ...
    
    def vf_to_vv(self, vertices, simplices): # -> None:
        """
        Convert a vertex-face mesh to a vertex-vertex mesh used by this class

        Parameters
        ----------
        vertices : list
            Vertices
        simplices : list
            Simplices
        """
        ...
    
    def connect_vertex_non_symm(self, v_x, near=...): # -> bool | None:
        """
        Adds a vertex at coords v_x to the complex that is not symmetric to the
        initial triangulation and sub-triangulation.

        If near is specified (for example; a star domain or collections of
        cells known to contain v) then only those simplices containd in near
        will be searched, this greatly speeds up the process.

        If near is not specified this method will search the entire simplicial
        complex structure.

        Parameters
        ----------
        v_x : tuple
            Coordinates of non-symmetric vertex
        near : set or list
            List of vertices, these are points near v to check for
        """
        ...
    
    def in_simplex(self, S, v_x, A_j0=...): # -> bool:
        """Check if a vector v_x is in simplex `S`.

        Parameters
        ----------
        S : array_like
            Array containing simplex entries of vertices as rows
        v_x :
            A candidate vertex
        A_j0 : array, optional,
            Allows for A_j0 to be pre-calculated

        Returns
        -------
        res : boolean
            True if `v_x` is in `S`
        """
        ...
    
    def deg_simplex(self, S, proj=...): # -> bool:
        """Test a simplex S for degeneracy (linear dependence in R^dim).

        Parameters
        ----------
        S : np.array
            Simplex with rows as vertex vectors
        proj : array, optional,
            If the projection S[1:] - S[0] is already
            computed it can be added as an optional argument.
        """
        ...
    


