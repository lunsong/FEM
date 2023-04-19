import numpy as np
from itertools import product

from mesh import Mesh
from FEM import FEM

def make_exterior_mesh(N,dx):
    N = np.array(N)
    L = (N-1)*dx
    Lm = np.max(L)
    R = Lm*5
    dim = L.shape[0]
    def quality(p):
        p = np.abs(p)-L.reshape(1,dim) / 2
        p = np.where(p<0, 0, p)
        return dx * (6*np.linalg.norm(p,axis=-1)/Lm+1.1)
    domain = lambda p: (np.linalg.norm(p)<R) & (~(np.abs(p)<L/2).all())
    if dim==3:
        def add_face_points3d(a,b,c,d):
            for i,j in product(range(N[a]),range(1,N[b]-1)):
                new_point = [0,0,0]
                new_point[a] = i * dx - L[a]/2
                new_point[b] = j * dx - L[b]/2
                new_point[c] = d*L[c]/2
                points.append(new_point)
        points = []
        add_face_points3d(0,1,2,1)
        add_face_points3d(0,1,2,-1)
        add_face_points3d(1,2,0,1)
        add_face_points3d(1,2,0,-1)
        add_face_points3d(2,0,1,1)
        add_face_points3d(2,0,1,-1)
        for x in product((-1,1),repeat=3):
            points.append(x*L/2)
    elif dim==2:
        points =  [(i*dx-L[0]/2,L[1]/2) for i in range(N[0]-1)]
        points += [(i*dx-L[0]/2,-L[1]/2) for i in range(1,N[0])]
        points += [(-L[0]/2,i*dx-L[1]/2) for i in range(N[0]-1)]
        points += [( L[0]/2,i*dx-L[1]/2) for i in range(1,N[0])]
    mesh = Mesh(np.prod(N),((-R,R),)*dim, quality, domain, points)
    return mesh

def solve_exterior(fem,f):
    x = np.zeros(fem.points.shape[0])
    x[fem.fixed] = f(fem.points[fem.fixed])
    x = (fem.nabla @ x )
    var = slice(fem.fixed.stop,None)
    x = x[var]
    nabla = fem.nabla[var,var]
    return x, nable



def solve(j,k):
    pass

