from numpy import zeros
from scipy.sparse import csr_array

from numpy.ctypeslib import ndpointer as ndp
from ctypes import *

cFEM = CDLL("./cFEM.so").FEM
cFEM.argtypes=(
        [ndp(c_int)]*6
       +[c_int]*2
       +[ndp(c_double)]*5 )

cdiag = CDLL("./cFEM.so").diag
cdiag.argtypes=(
        [ndp(c_int)]*3
       +[c_int]
       +[ndp(c_double)]*3)

class FEM:
    def __init__(self, mesh):

        tri = mesh.tri
        points = mesh.points
        simplices = mesh.get_simplices()
        boundary = mesh.get_boundary()
        indptr, indices = tri.vertex_neighbor_vertices

        N_points = points.shape[0]
        N_indices_fem = indices.size + N_points
        N_simplices = simplices.shape[0]
        nabla_data  = zeros(N_indices_fem,c_double)
        inner_data  = zeros(N_indices_fem,c_double)
        diag_data   = zeros(N_indices_fem,c_double)
        bound_data  = zeros(N_indices_fem,c_double)
        indices_fem = zeros(N_indices_fem,c_int   )
        indptr_fem  = zeros(N_points  +1 ,c_int   )
        V6s         = zeros(N_simplices  ,c_double)

        cFEM(indices, indptr, indices_fem, indptr_fem, simplices,
             boundary, N_points, N_simplices, points, nabla_data,
             inner_data, bound_data,V6s)

        nabla = csr_array((nabla_data, indices_fem, indptr_fem))
        inner = csr_array((inner_data, indices_fem, indptr_fem))
        diag  = csr_array(( diag_data, indices_fem, indptr_fem))
        bound = csr_array((bound_data, indices_fem, indptr_fem))

        nabla += nabla.T
        inner += inner.T
        bound += bound.T

        self.tri   = tri
        self.V6s   = V6s
        self.nabla = nabla
        self.inner = inner
        self.bound = bound 
        self._diag = diag
        self.fixed = mesh.fixed
        self.points = mesh.points
        self.indptr = indptr_fem
        self.indices = indices_fem
        self.simplices = simplices

    def diag(self, f):
        if callable(f):
            f = f(self.points)
        self._diag.data[:] = 0
        cdiag(self.indices, self.indptr, self.simplices,
              self.simplices.shape[0], self.V6s, f, self._diag.data)
        self._diag += self._diag.T
        return self._diag
