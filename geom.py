from numpy import (arcsin, abs as nabs, sin, cos, arccos, pi, array,
        arctan2, zeros, cross, sqrt, einsum, tan, ndarray, iterable,
        concatenate, ceil, max as nmax, min as nmin)
from numpy.linalg import norm
from scipy.linalg import expm

from itertools import product
from mesh import Mesh
from FEM import FEM


class Geom:
    eps = 1e-3
    def __init__(self, points, domain):
        self.points = points
        self.domain = domain
    def __add__(self, x):
        assert type(x)==Geom
        domain = lambda ps: nmax((self.domain(ps),x.domain(ps)),axis=0)
        points = concatenate((self.points, x.points))
        points = points[nabs(domain(points))<1e-4 ]
        return Geom(points, domain)
    def __sub__(self, x):
        assert type(x)==Geom
        domain = lambda ps: nmin((self.domain(ps), -x.domain(ps)),axis=0)
        points = concatenate((self.points, x.points))
        points = points[nabs(domain(points))<1e-4 ]
        return Geom(points, domain)
    def move(self, vec):
        self.points += vec
        _domain = self.domain
        self.domain = lambda ps: _domain(ps-vec)
        return self
    def rot(self, axis, deg=None):
        #if deg!=None:
        #    axis *= deg / norm(axis)
        #M = expm(einsum("ijk,j->ik", levi_civita, axis))
        #self.points = einsum("ij,lj->li", M, self.points)
        #return self
        raise NotImplementedError
    def range(self):
        ans = ((nmin(self.points,axis=0),nmax(self.points,axis=0)))
        return array(ans).T

def box(L,dx,center=None):
    N = ceil(array(L)/dx).astype(int)+1
    L = (N-1)*dx
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
    if center is None:
        center = array([[0,0,0]])
    points = array(points+center)
    inside = lambda ps: (L/2-nabs(ps-center)).min(axis=-1)
    return Geom(points, inside)

def coord(thphi):
    th,phi = thphi
    return array([sin(th)*cos(phi), sin(th)*sin(phi), cos(th)])

levi_civita = zeros((3,3,3))
levi_civita[0,1,2] = levi_civita[1,2,0] = levi_civita[2,0,1] = 1
levi_civita[0,2,1] = levi_civita[1,0,2] = levi_civita[2,1,0] = -1

def get_mat(r0,r1,n=1):
    w = cross(r0,r1)
    w *= arccos(r0@r1) / sqrt(w@w) / n
    M = einsum("ijk,j->ik", levi_civita, w)
    return expm(M)

def ball(R,dx):
    dth = arcsin(dx/2/R) * 2
    th = arccos(cos(pi/5)/(1+cos(pi/5)))
    n = int(ceil(th / dth))

    points = list(map(coord, ((0,0),(th,0),(th,pi*2/5),(th,pi*4/5),
        (th,pi*6/5),(th,pi*8/5),(pi,0),(pi-th,pi*1/5),(pi-th,pi*3/5),
        (pi-th,pi*5/5),(pi-th,pi*7/5),(pi-th,pi*9/5))))
    edges = sum((((0,i), (6,i+6), (i,i+6), (i,(i-2)%5+7),
                  (i,i%5+1), (i+6,i%5+7)) for i in range(1,6)), start=())
    faces = sum((((0,i,i%5+1), (6,i+6,i%5+7), (i,i%5+1,i+6),
                  (i,i+6,(i-2)%5+7)) for i in range(1,6)), start=())
    for a,b in edges:
        a,b = points[a],points[b]
        M = get_mat(a,b,n)
        c = array(a)
        for i in range(n-1):
            c = M @ c
            points.append(c)
    for a,b,c in faces:
        a,b,c = map(points.__getitem__,(a,b,c))
        M0 = get_mat(a,b,n)
        M1 = get_mat(a,c,n)
        d,e = map(array,(a,a))
        for i in range(n-1):
            d,e = M0@d, M1@e
            M2 = get_mat(d,e,i+1)
            f = array(d)
            for j in range(i):
                f = M2 @ f
                points.append(f)
    points = array(points) * R
    def foo(x):
        if not iterable(x): return array([x])
        return x
    domain = lambda ps: foo(R - norm(ps, axis=-1))
    return Geom(points, domain)
