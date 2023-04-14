from numpy.linalg import det
from numpy import cross, sort, zeros, array, zeros_like
from scipy.sparse import csr_array, lil_array

class FEM:
    def __init__(self, mesh, construct="lil"):
        tri = mesh.delaunay
        to_interior = -1 + zeros(len(tri.points),int)
        to_original = []
        convex_hull = sorted(set(tri.convex_hull.ravel()))
        for i in range(len(tri.points)):
            if len(convex_hull)>0 and i == convex_hull[0]:
                convex_hull.pop(0)
                continue
            to_interior[i] = len(to_original)
            to_original.append(i)
        for i in range(len(tri.points)):
            if len(convex_hull)>0 and i == convex_hull[0]:
                convex_hull.pop(0)
                continue
            to_interior[i] = len(to_original)
            to_original.append(i)
        if construct=="lil":
            A = lil_array((len(to_original),)*2)
            B = lil_array((len(to_original),)*2)
        elif construct=="csr":
            indptr, indices = tri.vertex_neighbor_vertices
            indices_int = []
            indptr_int = [0]
            for i_int, i in enumerate(to_original):
                indices_int.append(i_int)
                for j in indices[indptr[i]:indptr[i+1]]:
                    if to_interior[j]!=-1:
                        indices_int.append(to_interior[j])
                indptr_int.append(len(indices_int))
            indptr,indices = map(lambda x:array(x,dtype='int32'),
                    (indptr_int, indices_int))
            N = indices.shape[0]
            A,B = (csr_array((zeros(N),indices,indptr)) for _ in range(2))
        else:
            raise ValueError("unknown construct type")
        V6s = []
        tot = len(tri.simplices)
        for step, s in enumerate(tri.simplices):
            print(f"{step:4}/{tot}", end="\r")
            p = tri.points[s]
            V6 = abs(det(p[1:]-p[0].reshape(1,3)))
            if V6<1e-6: breakpoint()
            V6s.append(V6)
            for i in range(4):
                if to_interior[s[i]] == -1: continue
                i_int = to_interior[s[i]]
                for j in range(i+1):
                    if to_interior[s[j]] == -1: continue
                    j_int = to_interior[s[j]]
                    l,k = tuple(set(range(4)) - set((i,j)))[:2]
                    r0 = cross(p[i]-p[l], p[k]-p[l])
                    r1 = cross(p[j]-p[l], p[k]-p[l])
                    # The following coefficient result from
                    # A_real = A + A.T at the end the function
                    sign = .5 if i==j else -1
                    A[i_int,j_int] = sign * (r0@r1) / (6*V6)
                    # B_ii is really V6 / 60 but as said before, the
                    # conjugate is added
                    B[i_int,j_int] = V6 / 120
        A += A.T
        B += B.T
        self.A = A.tocsr()
        self.B = B.tocsr()
        self.C = csr_array((zeros_like(self.A.data),self.A.indices,
                            self.A.indptr))
        self.tri = tri
        self.to_interior = to_interior
        self.to_original = to_original
        self.V6s = V6s

    def Hemholtz(self,f):
        simps = self.tri.simplices
        C,V6s,to_int = self.C,self.V6s,self.to_interior
        tot = len(simps)
        C.data[:] = 0
        for step, s in enumerate(simps):
            print(f"{step:4}/{tot}", end="\r")
            for i in range(4):
                i_int = to_int[s[i]]
                if i_int == -1: continue
                for j in range(i+1):
                    j_int = to_int[s[j]]
                    if j_int == -1: continue
                    if i==j:
                        for k in range(4):
                            if i==k:
                                e = 1 / 120 / 2
                            else:
                                e = 1 / 360 / 2
                            C[i_int,j_int] += f[k_int] * e * V6s[step]
                    else:
                        for k in range(4):
                            if k in (i,j):
                                e = 1/360
                            else:
                                e = 1/720
                            C[i_int,j_int] += f[k_int] * e * V6s[step]
        C += C.T
                        

                    



    



