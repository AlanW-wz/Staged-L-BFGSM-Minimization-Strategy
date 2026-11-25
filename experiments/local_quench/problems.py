"""Analytic objectives and paired initialization for the ten-function quench study."""
import numpy as np
from .benchmarks import Benchmark


NAMES = {
    1: "styblinski_tang",
    2: "block_three_hump_camel",
    3: "block_bohachevsky2",
    4: "block_griewank",
    5: "ic_cosine_mixture",
    6: "ic_rastrigin",
    7: "block_bohachevsky3",
    8: "corrugated_rosenbrock",
    9: "block8_corrugated_rosenbrock",
    10: "heterogeneous_coupled_double_well",
}
IDS = tuple(NAMES)
SEEDS = tuple(range(2026110701, 2026110731))
RANGES = {
    1: (-5., 5.), 2: (-5., 5.), 3: (-100., 100.), 4: (-100., 100.),
    5: (-1., 1.), 6: (-5.12, 5.12), 7: (-100., 100.), 8: (-3., 3.),
    9: (-3., 3.), 10: (-2., 2.),
}


class CoupledDoubleWell:
    """Periodic chain, interleaved (u_j,v_j), lambda_j cycling from 1 to 100."""

    name = NAMES[10]
    fstar = 0.
    low, high = RANGES[10]

    def __init__(self, dimension):
        if dimension < 2 or dimension % 2:
            raise ValueError("F10 requires a positive even dimension")
        self.dim = dimension
        self.stiffness = 10. ** (2. * (np.arange(dimension // 2) % 8) / 7.)

    def optimum(self):
        return np.ones(self.dim)

    def value_grad(self, x):
        u, v = x[::2], x[1::2]
        u_sq = u * u
        wells = u_sq - 1.
        residual = v - 0.5 * (u_sq + np.roll(u, -1))
        weighted = self.stiffness * residual
        value = np.dot(wells, wells) + 0.5 * np.dot(weighted, residual)
        gradient = np.empty(self.dim)
        gradient[::2] = 4. * u * wells - u * weighted - 0.5 * np.roll(weighted, 1)
        gradient[1::2] = weighted
        return float(value), gradient


class BlockCorrugatedRosenbrock:
    """A sum of independent open F8 chains. Each chain has eight coordinates."""
    name = NAMES[9]
    fstar = 0.0
    low, high = RANGES[9]

    def __init__(self, dimension):
        if dimension < 8 or dimension % 8:
            raise ValueError("F9 requires a positive dimension divisible by eight")
        self.dim = dimension

    def optimum(self):
        return np.ones(self.dim)

    def value_grad(self, x):
        blocks = x.reshape(-1, 8)
        left, right = blocks[:, :-1], blocks[:, 1:]
        residual = right - left * left
        offset = left - 1.
        angle = (2. * np.pi) * (blocks - 1.)
        f = (100. * np.sum(residual * residual) + np.sum(offset * offset)
             + 2. * np.sum(1. - np.cos(angle)))
        g = (4. * np.pi) * np.sin(angle)
        g[:, :-1] += -400. * left * residual + 2. * offset
        g[:, 1:] += 200. * residual
        return float(f), g.ravel()


def problem(function_id: int, dimension: int):
    if function_id == 9:
        return BlockCorrugatedRosenbrock(dimension)
    if function_id == 10:
        return CoupledDoubleWell(dimension)
    return Benchmark(NAMES[function_id], dimension)


def initial_point(function_id: int, dimension: int, seed: int) -> np.ndarray:
    random = np.random.default_rng(np.random.SeedSequence([seed, function_id, dimension]))
    return random.uniform(*RANGES[function_id], size=dimension)
