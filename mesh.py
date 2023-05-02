import numpy as np
from numpy.ctypeslib import ndpointer as ndp
from scipy.spatial import Delaunay, delaunay_plot_2d
#from matplotlib.pyplot import scatter, show
from ctypes import *

__all__ = ["Mesh"]

mutual = CDLL("./lib.so").mutual
mutual.argtypes = [ndp(c_double),c_int,c_int,ndp(c_int),ndp(c_int),
                   ndp(c_double),ndp(c_double),c_int]
_avr = CDLL("./lib.so").avr_dist_satis
_avr.restype = c_double
_avr.argtypes = [ndp(c_int),ndp(c_int),ndp(c_double),c_int,ndp(c_double),
        c_int,c_int]

class Mesh:
    def __init__(self, N, sample_range, quality, domain=None, points=None,
            step=4, is_uniform=False):
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

        if is_uniform:
            sampler = uniform_sampler
            quality = np.array([quality])
        else:
            sampler = markov_sampler

        self.quality = quality
        self.is_uniform = is_uniform

        if points is None: points = []
        self.fixed = range(len(points))
        if type(points)==list:
            points += [sampler() for _ in range(N)]
        elif type(points)==np.ndarray:
            points = np.concatenate((
                points,[sampler() for _ in range(N)]))
        else: raise ValueError("Invalid points type")
        tri = Delaunay(points, incremental=True)
        print("N\tavr")
        def avr():
            indptr,indices = tri.vertex_neighbor_vertices
            if self.is_uniform:
                quality = self.quality
            else:
                quality = self.quality(tri.points)
            x = _avr(indptr,indices,quality,tri.points.shape[0],
                    tri.points,tri.points.shape[1], is_uniform)
            x /= indices.size / 2
            print(f"{tri.points.shape[0]}\t{x}",end="\r")
            return x
        while avr() > 0:
            tri.add_points([sampler() for _ in range(N//5)])

        print()

        tri.close()

        self.delaunay = tri
        self.points = tri.points

    def refine(self, k, N_sub=20, N_tot=30, tol=1e-2, verbose=False):
        points = self.points
        dis = np.zeros_like(points)
        print("step\tsubstep\tmax dis")
        for step in range(N_tot):
            self.delaunay = Delaunay(points)
            indptr,indices = self.delaunay.vertex_neighbor_vertices
            for substep in range(N_sub):
                if self.is_uniform:
                    quality = self.quality
                else:
                    quality = self.quality(points)
                dis[:] = 0
                mutual(points, points.shape[0], points.shape[1],
                       indptr, indices, quality, dis, self.is_uniform)
                dis *= k
                dis[self.fixed] = 0
                points += dis
                if substep % (N_sub//5) == 0:
                    max_dis = np.max(np.linalg.norm(dis,axis=1))
                    endline = "\n" if verbose else "\r"
                    print(step,substep,max_dis/k,sep="\t", end=endline)
                    if max_dis < tol*k: break
        print()

if __name__ == "__main__":
    quality = lambda p: (.2*np.linalg.norm(p,axis=-1)+1)*.3
    domain =  lambda p: np.linalg.norm(p)<=3.
    mesh = Mesh(100, ((-3,3),)*2, quality, domain)
    mesh.refine(.1, N_tot=80)


       

