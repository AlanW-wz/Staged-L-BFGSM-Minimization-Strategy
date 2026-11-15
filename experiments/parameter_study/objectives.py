"""Seven parameter-study objectives; original floating-point expressions retained.

The trigonometric chain is a strongly convex oscillatory control. Schwefel is
used unconstrained here, so its usual bounded-domain optimum is not asserted.
"""
from __future__ import annotations
import math
import numpy as np
from .config import PROBLEMS
Vector = np.ndarray


def dixon_price(x: Vector) -> float:
    if x.size == 1:
        return float((x[0] - 1.0) ** 2)
    t = 2.0 * x[1:] ** 2 - x[:-1]
    weights = np.arange(2, x.size + 1, dtype=float)
    return float((x[0] - 1.0) ** 2 + np.sum(weights * t * t))

def grad_dixon_price(x: Vector) -> Vector:
    g = np.zeros_like(x)
    g[0] += 2.0 * (x[0] - 1.0)
    if x.size == 1:
        return g
    t = 2.0 * x[1:] ** 2 - x[:-1]
    weights = np.arange(2, x.size + 1, dtype=float)
    g[:-1] += -2.0 * weights * t
    g[1:] += 8.0 * weights * x[1:] * t
    return g

def levy(x: Vector) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    first = math.sin(math.pi * w[0]) ** 2
    middle_arg = math.pi * w[:-1] + 1.0
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(middle_arg) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return float(first + middle + last)

def grad_levy(x: Vector) -> Vector:
    w = 1.0 + (x - 1.0) / 4.0
    gw = np.zeros_like(x)
    gw[0] += math.pi * math.sin(2.0 * math.pi * w[0])
    if x.size > 1:
        u = w[:-1]
        arg = math.pi * u + 1.0
        sin_arg = np.sin(arg)
        b = 1.0 + 10.0 * sin_arg * sin_arg
        gw[:-1] += 2.0 * (u - 1.0) * b + (u - 1.0) ** 2 * 10.0 * math.pi * np.sin(2.0 * arg)
    u_last = w[-1]
    last_arg = 2.0 * math.pi * u_last
    gw[-1] += 2.0 * (u_last - 1.0) * (1.0 + math.sin(last_arg) ** 2)
    gw[-1] += (u_last - 1.0) ** 2 * 2.0 * math.pi * math.sin(2.0 * last_arg)
    return gw / 4.0

def rastrigin(x: Vector, a: float = 10.0) -> float:
    n = x.size
    return float(a * n + np.sum(x * x - a * np.cos(2.0 * np.pi * x)))

def grad_rastrigin(x: Vector, a: float = 10.0) -> Vector:
    return 2.0 * x + 2.0 * np.pi * a * np.sin(2.0 * np.pi * x)

def schwefel(x: Vector) -> float:
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def grad_schwefel(x: Vector) -> Vector:
    ax = np.abs(x)
    s = np.sqrt(ax)
    out = np.zeros_like(x)
    mask = s > 1e-16
    out[mask] = -np.sin(s[mask]) - 0.5 * s[mask] * np.cos(s[mask])
    return out

def levy_scaled(x: Vector) -> float:
    z = 0.75 * x
    w = 1.0 + (z - 1.0) / 4.0
    first = math.sin(math.pi * w[0]) ** 2
    u = w[:-1]
    middle = np.sum((u - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * u + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[-1]) ** 2)
    return float(first + middle + last)

def grad_levy_scaled(x: Vector) -> Vector:
    z = 0.75 * x
    w = 1.0 + (z - 1.0) / 4.0
    gw = np.zeros_like(x)
    gw[0] += math.pi * math.sin(2.0 * math.pi * w[0])
    u = w[:-1]
    arg = math.pi * u + 1.0
    gw[:-1] += 2.0 * (u - 1.0) * (1.0 + 10.0 * np.sin(arg) ** 2)
    gw[:-1] += (u - 1.0) ** 2 * 10.0 * math.pi * np.sin(2.0 * arg)
    u_last = w[-1]
    last_arg = 2.0 * math.pi * u_last
    gw[-1] += 2.0 * (u_last - 1.0) * (1.0 + math.sin(last_arg) ** 2)
    gw[-1] += (u_last - 1.0) ** 2 * 2.0 * math.pi * math.sin(2.0 * last_arg)
    return 0.75 * gw / 4.0

def levy_shifted(x: Vector) -> float:
    return levy_scaled(x - 0.5)

def grad_levy_shifted(x: Vector) -> Vector:
    return grad_levy_scaled(x - 0.5)

def trig_chain(x: Vector) -> float:
    d = x[1:] - x[:-1]
    return float(0.5 * np.sum(x * x) + 0.05 * np.sum(1.0 - np.cos(2.0 * d)))

def grad_trig_chain(x: Vector) -> Vector:
    g = x.copy()
    d = x[1:] - x[:-1]
    q = 0.10 * np.sin(2.0 * d)
    g[:-1] -= q
    g[1:] += q
    return g

def problem(name):
    """Return objective and analytical gradient for a configured named function."""
    if name not in PROBLEMS:
        raise ValueError(f"Unknown parameter-study function: {name}")
    return globals()[name], globals()['grad_' + name]


def initial_point(name, dimension):
    if name not in PROBLEMS or dimension < 1:
        raise ValueError("Use a configured function and a positive dimension")
    return np.random.default_rng(PROBLEMS[name]['base_seed'] + dimension).uniform(-3.0, 3.0, dimension).astype(float)
