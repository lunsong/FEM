from numpy import (sin, cos, arccos, pi, array, arctan2, zeros,
        cross, sqrt, einsum, tan, ndarray, iterable)
from numpy.linalg import norm
from scipy.linalg import expm

from mesh import Mesh
from FEM import FEM

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

def icosahedral_mesh(n):
    th = arccos(cos(pi/5)/(1+cos(pi/5)))

    points = list(map(coord, ((0,0),(th,0),(th,pi*2/5),(th,pi*4/5),
        (th,pi*6/5),(th,pi*8/5),(pi,0),(pi-th,pi*1/5),(pi-th,pi*3/5),
        (pi-th,pi*5/5),(pi-th,pi*7/5),(pi-th,pi*9/5))))
    edges = sum((((0,i), (6,i+6), (i,i+6), (i,(i-2)%5+7),
                  (i,i%5+1), (i+6,i%5+7)) for i in range(1,6)), start=())
    faces = sum((((0,i,i%5+1), (6,i+6,i%5+7), (i,i%5+1,i+6),
                  (i,i+6,(i-2)%5+7)) for i in range(1,6)), start=())
    print(len(edges),len(faces))
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
    return array(points)
        
def make_mesh(R,n):
    mesh = icosahedral_mesh(n)*R
    dx = 2*tan(arccos(cos(pi/5)/(1+cos(pi/5))) / 2 / n)
    domain = lambda p: norm(p)<R-dx
    N = int(4*pi*R**3 / 3 / (sqrt(2)/12 * dx**3) / 5)
    print(f"N={N}")
    mesh = Mesh(N, ((-R,R),)*3,dx,domain,mesh,is_uniform=True)
    return mesh



