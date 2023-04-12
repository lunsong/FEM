import numpy as np
from numpy.ctypeslib import ndpointer as ndp
from scipy.spatial import Delaunay, delaunay_plot_2d
from matplotlib.pyplot import scatter, show
from ctypes import *

__all__ = ["Mesh"]

mutual = CDLL("./lib.so").mutual
mutual.argtypes = [ndp(c_double),c_int,c_int,ndp(c_int),ndp(c_int),
                   ndp(c_double),ndp(c_double)]
_avr = CDLL("./lib.so").avr_dist_satis
_avr.restype = c_double
_avr.argtypes = [ndp(c_int),ndp(c_int),ndp(c_double),c_int,ndp(c_double),
        c_int]

class Mesh:
    def __init__(self, N, sample_range, quality, domain=None, step=4):
        if domain==None:
            domain = lambda x:True

        def uniform_sampler():
            while True:
                x = [np.random.uniform(*r) for r in sample_range]
                if domain(x):
                    return x

        dim = len(sample_range)
        P = lambda x: quality(x)**(-dim)

        def markov_sampler():
            x = uniform_sampler()
            for _ in range(step):
                y = uniform_sampler()
                if P(y)>P(x) or np.random.uniform() < P(y)/P(x):
                    x = y 
            return x

        self.quality = quality

        points = [markov_sampler() for _ in range(N)]
        tri = Delaunay(points, incremental=True)
        def avr():
            indptr,indices = tri.vertex_neighbor_vertices
            quality = self.quality(tri.points)
            x = _avr(indptr,indices,quality,tri.points.shape[0],
                    tri.points,tri.points.shape[1]) / (indices.size/2)
            print(x)
            return x
        while avr() > 0:
            tri.add_points([markov_sampler() for _ in range(N//5)])

        tri.close()

        self.delaunay = tri
        self.points = tri.points

    def refine(self, k, N_sub=20, N_tot=30, tol=1e-3, fixed=None):
        points = self.points
        dis = np.zeros_like(points)
        print("step\tsubstep\tmax dis")
        for step in range(N_tot):
            self.delaunay = Delaunay(points)
            indptr,indices = self.delaunay.vertex_neighbor_vertices
            for substep in range(N_sub):
                dis[:] = 0
                mutual(points, points.shape[0], points.shape[1],
                       indptr, indices, self.quality(points), dis)
                dis *= k
                if fixed!=None: dis[fixed] = 0
                points += dis
            max_dis = np.max(np.linalg.norm(dis,axis=1))
            print(step,max_dis,sep="\t", end="\r")
            if max_dis < tol: break

if __name__ == "__main__":
    quality = lambda p: (.2*np.linalg.norm(p,axis=-1)+1)*.3
    domain =  lambda p: np.linalg.norm(p)<=3.
    mesh = Mesh(100, ((-3,3),)*2, quality, domain)
    mesh.refine(.1, N_tot=80)


       

