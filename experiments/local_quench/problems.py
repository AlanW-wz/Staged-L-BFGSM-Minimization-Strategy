"""Second-round objectives. Only F7 and F9 differ from original_problems."""
import numpy as np

if __package__:
    from . import original_problems as original
    from .vendor import aqn_10f
else:
    import original_problems as original
    from vendor import aqn_10f

NAMES = {**original.NAMES, 7: "block_bohachevsky3", 9: "block8_corrugated_rosenbrock"}
RANGES = {**original.RANGES, 7: (-100.0, 100.0), 9: (-3.0, 3.0)}
IDS = original.IDS
SEEDS = original.SEEDS


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
    if function_id == 7:
        return aqn_10f.Benchmark(NAMES[7], dimension)
    if function_id == 9:
        return BlockCorrugatedRosenbrock(dimension)
    return original.problem(function_id, dimension)


def initial_point(function_id: int, dimension: int, seed: int) -> np.ndarray:
    random = np.random.default_rng(np.random.SeedSequence([seed, function_id, dimension]))
    return random.uniform(*RANGES[function_id], size=dimension)
