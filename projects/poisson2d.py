import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        x = np.linspace(0, self.px.L, self.px.N+1)
        y = np.linspace(0, self.py.L, self.py.N+1)
        return np.meshgrid(x, y, indexing='ij')

    def laplace(self):
        """Return a vectorized Laplace operator"""
        N_x = self.px.N
        N_y = self.py.N
        D2x = self.px.D2()
        D2y = self.py.D2()
        return (sparse.kron(D2x, sparse.eye(N_y+1)) + sparse.kron(sparse.eye(N_x+1), D2y))

    def assemble(self, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        xij, yij = self.create_mesh()
        F = sp.lambdify((x,y), f)(xij, yij)
        A = self.laplace()
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0      
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        b = F.ravel()
        b[bnds] = 0

        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        dx = self.px.dx
        dy = self.py.dx
        xij, yij = self.create_mesh()
        uj = sp.lambdify((x, y), ue)(xij, yij)
        return np.sqrt(dx*dy*np.sum(uj - u)**2)

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    object = Poisson2D(Lx=1, Ly=1, Nx=30, Ny=30)
    function = 2*(x*(x - 1) + y*(y - 1))
    u_num = object(f = function)
    ue = x*(1 - x)*y*(1 - y)
    assert object.l2_error(u_num, ue) < 1e-12
    ue2 = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    function2 = ue2.diff(x, 2) + ue2.diff(y, 2)
    u_num2 = object(f = function2)
    assert object.l2_error(u_num2, ue2) < 0.0001

if __name__ == '__main__':
    test_poisson2d()