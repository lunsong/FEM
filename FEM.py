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
       +[ndp(c_double)]*4 )

cdiag = CDLL("./cFEM.so").diag
cdiag.argtypes=(
        [ndp(c_int)]*4
       +[c_int]
       +[ndp(c_double)]*3)

class FEM:
    def __init__(self, mesh, N_excluded=0, construct="lil"):
        tri = mesh.delaunay
        neumann = mesh.neumann
        indptr, indices = tri.vertex_neighbor_vertices
        convex_hull = array(sorted(set(tri.convex_hull.ravel())),
                dtype=c_int)
        N_points = mesh.points.shape[0]
        N_interior = N_points - N_excluded - convex_hull.size
        N_indices_int = indices.size + N_interior
        N_simplices = tri.simplices.shape[0]
        N_neumann   = neumann.shape[0]
        nabla_data  = zeros(N_indices_int,c_double)
        inner_data  = zeros(N_indices_int,c_double)
        neumann_data= zeros(N_indices_int,c_double)
        indices_int = zeros(N_indices_int,c_int   )
        indptr_int  = zeros(N_interior+1 ,c_int   )
        to_interior = zeros(N_points     ,c_int   ) - 1
        to_original = zeros(N_interior   ,c_int   )
        V6s         = zeros(N_simplices  ,c_double)

        N_indices = c_int(0)

        cFEM(indices, indptr, indices_int, indptr_int, convex_hull,
          to_interior, to_original, tri.simplices, neumann, N_points,
          convex_hull.size, N_excluded, N_simplices, N_neumann,
          N_indices, tri.points, nabla_data, inner_data, neumann_data,V6s)

        nabla = csr_array((nabla_data[:N_indices.value],
                          indices_int[:N_indices.value],
                          indptr_int))
        inner = csr_array((inner_data[:N_indices.value],
                          indices_int[:N_indices.value],
                          indptr_int))
        neumann = csr_array((neumann_data[:N_indices.value],
                              indices_int[:N_indices.value],
                              indptr_int))
        diag  = csr_array((zeros_like(nabla.data),
                          indices_int[:N_indices.value],
                          indptr_int))
        nabla += nabla.T
        inner += inner.T
        neumann += neumann.T

        self.tri   = tri
        self.nabla = nabla
        self.inner = inner
        self.neumann = neumann 
        self.V6s   = V6s
        self._diag = diag
        self.to_original = to_original
        self.to_interior = to_interior
        self.fixed = mesh.fixed
        self.points = mesh.points

    def diag(self, f):
        if callable(f):
            f = f(self.points[self.to_original])
        f = f.astype(c_double)
        cdiag(self.nabla.indices, self.nabla.indptr, self.tri.simplices,
              self.to_interior, self.tri.simplices.shape[0],
              self.V6s, f, self._diag.data)
        self._diag += self._diag.T
        return self._diag
