"""Analytic objectives, published initialization boxes, and paired study starts.

The two byte-for-byte AQN snapshots in vendor/ originate from the author's
AQN-MH experiment collection:

  experiments_10f/benchmarks.py -> aqn_10f.py
  SHA256 f3063089f8bf2197c121269bdb2360b74c51b311f3414bed0899110deebaa88a
  experiments_v2/benchmarks.py -> aqn_v2.py
  SHA256 fd586d680f8b524632e9ee1139fd647a43695872734f2cc921d330a3feffaf91

F6, F7 and F9 use the v2 snapshot because its function registry includes
these cases. The initialization seed stream is specific to this study.
"""

import numpy as np

if __package__:
    from .vendor import aqn_10f, aqn_v2
else:
    from vendor import aqn_10f, aqn_v2


NAMES = {
    1: "styblinski_tang",
    2: "block_three_hump_camel",
    3: "block_bohachevsky2",
    4: "block_griewank",
    5: "ic_cosine_mixture",
    6: "ic_rastrigin",
    7: "ic_levy",
    8: "corrugated_rosenbrock",
    9: "expanded_griewank_rosenbrock",
    10: "heterogeneous_coupled_double_well",
}
IDS = tuple(NAMES)
SEEDS = tuple(range(2026110701, 2026110731))
RANGES = {
    1: (-5., 5.), 2: (-5., 5.), 3: (-100., 100.), 4: (-100., 100.),
    5: (-1., 1.), 6: (-5.12, 5.12), 7: (-10., 10.), 8: (-3., 3.),
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


def problem(function_id: int, dimension: int):
    """Return the specified analytic objective and its known optimal value."""
    if function_id == 10:
        return CoupledDoubleWell(dimension)
    factory = aqn_v2 if function_id in (6, 7, 9) else aqn_10f
    return factory.Benchmark(NAMES[function_id], dimension)


def initial_point(function_id: int, dimension: int, seed: int) -> np.ndarray:
    """Draw one common U([low,high]^d) point for all methods at this task."""
    random = np.random.default_rng(np.random.SeedSequence([seed, function_id, dimension]))
    low, high = RANGES[function_id]
    return random.uniform(low, high, size=dimension)
