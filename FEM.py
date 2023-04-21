from numpy.linalg import det
from numpy import cross, sort, zeros, array, zeros_like
from scipy.sparse import csr_array, lil_array

from numpy.ctypeslib import ndpointer as ndp
from ctypes import *

cFEM = CDLL("./cFEM.so").FEM
cFEM.argtypes=(
        [ndp(c_int)]*8
       +[c_int]*4
       +[POINTER(c_int)]
       +[ndp(c_double)]*2 )

def FEM(mesh, N_excluded=0, construct="lil"):
    tri = mesh.delaunay
    indptr, indices = tri.vertex_neighbor_vertices
    convex_hull = array(sorted(set(tri.convex_hull.ravel())),
            dtype=c_int)
    N_points = mesh.points.shape[0]
    N_interior = N_points - N_excluded - convex_hull.size
    N_indices_int = indices.size + N_interior
    data        = zeros(N_indices_int,c_double)
    indices_int = zeros(N_indices_int,c_int   )
    indptr_int  = zeros(N_interior+1 ,c_int   )
    to_interior = zeros(N_points     ,c_int   ) - 1
    to_original = zeros(N_interior   ,c_int   )

    N_indices = c_int(0)

    cFEM(indices, indptr, indices_int, indptr_int, convex_hull,
        to_interior, to_original, tri.simplices, N_points,
        convex_hull.size, N_excluded, tri.simplices.shape[0],
        N_indices, tri.points, data)

    nabla = csr_array((data[:N_indices.value],
                      indices_int[:N_indices.value],
                      indptr_int))
    nabla += nabla.T
    return nabla
