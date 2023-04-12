from itertools import product

from numpy import (prod, array, square, roll, zeros_like, concatenate,
        pi, where, zeros, indices)
from numpy.linalg import norm
from numpy.random import normal

from scipy.sparse.linalg import LinearOperator, bicgstab
from scipy.fft import fftn, ifftn

class GL_solver:

    def __init__(self, psi, v, kappa=1., dx=1.):
        self.v = v
        self.dx = dx
        self.psi = psi
        self.kdx = kappa*dx
        self.dim = len(psi.shape)
        self.grid = array(psi.shape)

        self.A = zeros(tuple(self.grid)+(3,),float)
        self.A0 = 0.

        self.psi_eq_linear = LinearOperator(
                shape = (prod(self.grid-2)*2,)*2,
                matvec = self._psi_eq_linear,
                rmatvec = lambda x: self._psi_eq_linear(x,True),
                dtype=float)

        self.psi_eq_multiplier = self.kdx**2

        self.interior = tuple(
                slice(1,self.grid[d]-1) for d in range(self.dim))

        grid = self.grid.reshape((3,)+(1,)*self.dim)
        x = ((indices(self.grid*2) + grid ) % (2*grid) - grid) * dx
        x = norm(x, axis=0)
        x[(0,)*self.dim] = dx / 2.38
        self.greenf = dx**self.dim / (4*pi*x)
        self.greenf = fftn(self.greenf).reshape(self.greenf.shape+(1,))

    def solve_A(self, J=None):
        J = self.current() if J is None else J
        fft_axis = tuple(range(self.dim))
        cropped = tuple(map(lambda x: slice(0,x), self.grid))
        J_fft = fftn(J, self.grid*2, fft_axis)
        A = ifftn(J_fft * self.greenf, axes=fft_axis)[cropped]
        self.A = A.real + self.A0

    def solve_psi(self, N0):
        residual = self.psi_eq()
        dpsi, info = bicgstab(self.psi_eq_linear, -residual, maxiter=N0)
        print("bicgstab info:", info)
        N = int(dpsi.size/2)
        dpsi = dpsi[:N] + 1j*dpsi[N:]
        self.psi[self.interior] += dpsi.reshape(self.grid-2)

    def kin(self, psi, conj=False):
        """The (linear) kinetic part of the free energy"""
        def sub(dim, side):
            # apply BC
            vec   = [slice(None) for _ in range(self.dim)] + [dim]
            inner = [slice(None) for _ in range(self.dim)]
            bound = [slice(None) for _ in range(self.dim)]
            if side == -1:
                sign = -1.
                vec[dim] = 0
                bound[dim] = 0
                inner[dim] = 1
            else:
                sign = 1.
                vec[dim] = self.grid[dim] - 2
                bound[dim] = self.grid[dim] - 1
                inner[dim] = self.grid[dim] - 2
            vec,inner,bound = map(tuple, (vec,inner,bound))
            vecs = (self.A[vec]+self.v[vec]) * self.kdx * sign
            if not conj:
                psi[bound] = psi[inner] * (2j-vecs)/(2j+vecs)
            else:
                psi[bound] = psi[inner] * (2j+vecs)/(2j-vecs)

            rolled = tuple(
                    slice(1+side,self.grid[d]-1+side) if d==dim else
                    slice(1,self.grid[d]-1) for d in range(self.dim))

            #print(rolled, self.interior, dim, side)

            psi_j = psi[rolled]
            psi_i = psi[self.interior]

            if side==1:
                A = self.A[self.interior+(dim,)]
            else:
                A = -self.A[rolled+(dim,)]

            if not conj:
                ans =  ( (psi_i-psi_j) / self.kdx**2
                       + A**2 * (psi_i+psi_j) / 4
                       + 1j * A * psi_j / self.kdx )
            else:
                ans =  ( (psi_i-psi_j) / self.kdx**2
                       + A**2 * (psi_i+psi_j) / 4 )
                psi[bound] *= -1
                ans += 1j * A * psi_j / self.kdx

            #breakpoint()

            return ans

        return sum(sub(d,s) for d,s in product((0,1,2),(-1,1)))

    def psi_eq(self,reture_float=True):
        psi = self.psi
        ans = self.kin(psi)
        psi = psi[self.interior]
        ans += ((psi*psi.conj()).real-1) * psi 
        ans *= self.psi_eq_multiplier
        ans = ans.flatten()
        return concatenate((ans.real, ans.imag))
        
    def _psi_eq_linear(self, psi, conj=False):
        """linearized equation"""
        if not psi.dtype==complex:
            N = int(psi.size / 2)
            #breakpoint()
            psi = psi[:N] + 1j*psi[N:]
        psi = psi.reshape(self.grid-2)
        _psi = zeros_like(self.psi)
        _psi[self.interior] = psi
        ans = self.kin(_psi, conj)
        _psi = self.psi[self.interior]
        ans += ((_psi*_psi.conj()).real-1) * psi
        ans += 2 * (_psi.conj()*psi).real * _psi
        ans *= self.psi_eq_multiplier
        ans = ans.flatten()
        return concatenate((ans.real, ans.imag))

    def current(self):

        J = zeros_like(self.A)

        def _set_J(dim):
            off = tuple(0 if d!= dim else 1 for d in range(self.dim))
            idx    = tuple(map(lambda x: slice(1,-x+self.grid[d]-1), off))
            rolled = tuple(map(lambda x: slice(1+x, self.grid[d]-1), off))
            psi = self.psi[idx]
            _psi = self.psi[rolled]
            J[idx+(dim,)] = (_psi*psi.conj()).imag / self.kdx
            J[idx+(dim,)] -= (
                        .25 * self.A[idx+(dim,)] * abs(psi+_psi)**2)

        for d in range(self.dim):
            _set_J(d)

        return J


if __name__ == "__main__":
    size = (30,30,30)
    dx = .1
    kappa = 1.
    psi = (normal(size=size)+1j*normal(size=size))*.01 + 1
    v = zeros(size+(3,),float)
    v[0,:,:,0] = 1.
    v[-1,:,:,0] = 1.
    v *= 0.01
    from matplotlib.pyplot import imshow, show
    solver = GL_solver(psi, v, kappa, dx)
    imshow(abs(solver.psi[:,:,15])); show()
    for step in range(1):
        solver.solve_psi(500)
        for fft_step in range(10):
            solver.solve_A()
    imshow(abs(solver.psi[:,:,15])); show()

