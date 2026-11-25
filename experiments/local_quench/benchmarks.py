"""AQN benchmark functions used in the local-quench study with analytic gradients.

Boxes specify initialization only: all objectives are optimized on R^d.
The block variants below are explicitly NOT unmodified CEC/BBOB functions.
"""
from dataclasses import dataclass
import numpy as np

TAU = 2.0 * np.pi


def hadamard8(x):
    """Symmetric orthonormal 8-coordinate block transform; no dense d*d matrix."""
    a = np.asarray(x, dtype=float).reshape(-1, 8).copy()
    for stride in (1, 2, 4):
        z = a.reshape(-1, 2 * stride)
        left, right = z[:, :stride].copy(), z[:, stride:].copy()
        z[:, :stride], z[:, stride:] = left + right, left - right
    return a.ravel() / np.sqrt(8.0)


@dataclass
class Benchmark:
    name: str
    dim: int

    def __post_init__(self):
        if self.dim < 8 or self.dim % 8:
            raise ValueError('Known benchmark and dimension divisible by 8 required')
        specs = {
            'block_bohachevsky3': (-100.0, 100.0, 1.0),
            'block_three_hump_camel': (-5.0, 5.0, 1.0),
            'ic_rastrigin': (-5.12, 5.12, 10.0),
            'block_bohachevsky2': (-100.0, 100.0, 1.0),
            'ic_cosine_mixture': (-1.0, 1.0, 10.0),
            'corrugated_rosenbrock': (-3.0, 3.0, 1.0),
            'styblinski_tang': (-5.0, 5.0, 1.0),
            'block_griewank': (-100.0, 100.0, 1.0),
        }
        if self.name not in specs:
            raise ValueError(f'Unknown benchmark {self.name!r}')
        self.low, self.high, kappa = specs[self.name]
        self.scales = np.tile(np.geomspace(1.0, kappa, 8), self.dim // 8)
        self.fstar = 0.0
        if self.name == 'ic_cosine_mixture':
            self.fstar = -0.1 * self.dim
        if self.name == 'styblinski_tang':
            roots = np.roots([4.0, 0.0, -32.0, 5.0])
            self.st_min = min(roots, key=lambda r: r**4 - 16*r*r + 5*r).real
            self.fstar = 0.5 * self.dim * (self.st_min**4 - 16*self.st_min**2 + 5*self.st_min)
        self.gw_denom = np.sqrt(np.arange(1.0, 9.0))

    def optimum(self):
        c = 1.0 if self.name == 'corrugated_rosenbrock' else 0.0
        if self.name == 'styblinski_tang':
            c = self.st_min
        return np.full(self.dim, c)

    def value_grad(self, x):
        name = self.name
        if name in ('block_bohachevsky3', 'block_three_hump_camel'):
            pairs=x.reshape(-1,2)
            u,v=pairs[:,0],pairs[:,1]
            g=np.empty_like(pairs)
            if name == 'block_bohachevsky3':
                a=3*np.pi*u+4*np.pi*v
                f=np.sum(u*u+2*v*v+.6*np.sin(a/2)**2)
                sn=np.sin(a)
                g[:,0]=2*u+.9*np.pi*sn
                g[:,1]=4*v+1.2*np.pi*sn
            else:
                u2=u*u
                f=np.sum(2*u2-1.05*u2*u2+u2*u2*u2/6+u*v+v*v)
                g[:,0]=4*u-4.2*u*u2+u*u2*u2+v
                g[:,1]=u+2*v
            return float(f),g.ravel()
        if name == 'ic_rastrigin':
            y = self.scales * hadamard8(x)
            f = np.sum(y*y + 10.0 * (1.0 - np.cos(TAU*y)))
            g = 2*y + 10*TAU*np.sin(TAU*y)
            return float(f), hadamard8(self.scales*g)
        if name == 'block_bohachevsky2':
            # Sum the standard 2D Bohachevsky 2 over independent pairs.
            # This custom scalable extension has at least 3**(d/2) isolated
            # strict local minima and unique global minimum x=0, f=0.
            pairs = x.reshape(-1, 2)
            u, v = pairs[:, 0], pairs[:, 1]
            au, av = 3*np.pi*u, 4*np.pi*v
            cu, cv = np.cos(au), np.cos(av)
            # Exactly 1-cos(au)*cos(av), without cancellation near x=0.
            corrugation = np.sin((au+av)/2)**2+np.sin((au-av)/2)**2
            f = np.sum(u*u+2*v*v+.3*corrugation)
            g = np.empty_like(pairs)
            g[:, 0] = 2*u+.9*np.pi*np.sin(au)*cv
            g[:, 1] = 4*v+1.2*np.pi*cu*np.sin(av)
            return float(f), g.ravel()
        if name == 'ic_cosine_mixture':
            y = self.scales*hadamard8(x)
            f = np.sum(y*y-0.1*np.cos(5*np.pi*y))
            g = 2*y+0.5*np.pi*np.sin(5*np.pi*y)
            return float(f), hadamard8(self.scales*g)
        if name == 'corrugated_rosenbrock':
            q = x[1:]-x[:-1]**2
            f = 100*np.dot(q,q)+np.sum((x[:-1]-1)**2)+2*np.sum(1-np.cos(TAU*(x-1)))
            g = 2*TAU*np.sin(TAU*(x-1))
            g[:-1] += -400*x[:-1]*q+2*(x[:-1]-1)
            g[1:] += 200*q
            return float(f), g
        if name == 'styblinski_tang':
            x2 = x*x
            return float(0.5*np.sum(x2*x2-16*x2+5*x)), 2*x*x2-16*x+2.5
        if name == 'block_griewank':
            z = x.reshape(-1,8)
            a = z/self.gw_denom
            c, sn = np.cos(a), np.sin(a)
            # Products excluding coordinate j are safe even if cos(a_j)=0.
            left = np.ones_like(c)
            right = np.ones_like(c)
            left[:,1:] = np.cumprod(c[:,:-1],axis=1)
            right[:,:-1] = np.cumprod(c[:,1:][:,::-1],axis=1)[:,::-1]
            f = np.sum(z*z/4000.0)+np.sum(1-np.prod(c,axis=1))
            g = z/2000.0 + sn*left*right/self.gw_denom
            return float(f), g.ravel()
        raise AssertionError(name)
