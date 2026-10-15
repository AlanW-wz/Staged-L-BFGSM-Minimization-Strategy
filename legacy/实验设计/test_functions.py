# -*- coding: utf-8 -*-
"""Benchmark test functions for the 实验设计 suite."""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]
ProblemSpec = Tuple[str, Objective, Gradient, str]


def ackley(x: Vector, a: float = 20.0, b: float = 0.2, c: float = 2.0 * math.pi) -> float:
    n = x.size
    s1 = float(np.sum(x * x)) / n
    s2 = float(np.sum(np.cos(c * x))) / n
    term1 = -a * math.exp(-b * math.sqrt(max(s1, 0.0)))
    term2 = -math.exp(s2)
    return float(term1 + term2 + a + math.e)


def grad_ackley(
    x: Vector, a: float = 20.0, b: float = 0.2, c: float = 2.0 * math.pi
) -> Vector:
    n = x.size
    s1 = float(np.sum(x * x)) / n
    r = math.sqrt(max(s1, 0.0))
    s2 = float(np.sum(np.cos(c * x))) / n
    exp1 = math.exp(-b * r)
    exp2 = math.exp(s2)
    if r > 1e-16:
        g1 = (a * b * exp1 / (n * r)) * x
    else:
        g1 = np.zeros_like(x)
    g2 = (exp2 * c / n) * np.sin(c * x)
    return g1 + g2


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


def griewank(x: Vector) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    return float(1.0 + np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))))


def grad_griewank(x: Vector) -> Vector:
    i = np.arange(1, x.size + 1, dtype=float)
    z = x / np.sqrt(i)
    cos_z = np.cos(z)
    sin_z = np.sin(z)
    prefix = np.cumprod(cos_z)
    suffix = np.cumprod(cos_z[::-1])[::-1]
    prod_except = np.ones_like(x)
    if x.size > 1:
        prod_except[0] = suffix[1]
        prod_except[-1] = prefix[-2]
        prod_except[1:-1] = prefix[:-2] * suffix[2:]
    return x / 2000.0 + prod_except * sin_z / np.sqrt(i)


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


def discus(x: Vector) -> float:
    return float(1e6 * x[0] * x[0] + np.sum(x[1:] * x[1:]))


def grad_discus(x: Vector) -> Vector:
    g = np.empty_like(x)
    g[0] = 2e6 * x[0]
    g[1:] = 2.0 * x[1:]
    return g


def bent_cigar(x: Vector) -> float:
    return float(x[0] * x[0] + 1e6 * np.sum(x[1:] * x[1:]))


def grad_bent_cigar(x: Vector) -> Vector:
    g = np.empty_like(x)
    g[0] = 2.0 * x[0]
    g[1:] = 2e6 * x[1:]
    return g


def zakharov(x: Vector) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    s = 0.5 * float(i @ x)
    return float(np.sum(x * x) + s * s + s**4)


def grad_zakharov(x: Vector) -> Vector:
    i = np.arange(1, x.size + 1, dtype=float)
    s = 0.5 * float(i @ x)
    return 2.0 * x + i * s + 2.0 * i * s**3


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


def ellipsoidal(x: Vector, cond: float = 1e6) -> float:
    n = x.size
    if n == 1:
        return float(x[0] * x[0])
    exponents = np.linspace(0.0, 1.0, n)
    weights = cond ** exponents
    return float(np.sum(weights * x * x))


def grad_ellipsoidal(x: Vector, cond: float = 1e6) -> Vector:
    n = x.size
    if n == 1:
        return 2.0 * x
    exponents = np.linspace(0.0, 1.0, n)
    weights = cond ** exponents
    return 2.0 * weights * x


def get_problem(name: str) -> ProblemSpec:
    table = {
        "ackley": (
            "Ackley",
            ackley,
            grad_ackley,
            r"f(x)= -a\exp\!\left(-b\sqrt{\tfrac1n\sum x_i^2}\right)"
            r" -\exp\!\left(\tfrac1n\sum\cos(cx_i)\right)+a+e,\ a=20,\ b=0.2,\ c=2\pi.",
        ),
        "rastrigin": (
            "Rastrigin",
            rastrigin,
            grad_rastrigin,
            r"f(x)=10n+\sum_{i=1}^n\left(x_i^2-10\cos(2\pi x_i)\right).",
        ),
        "schwefel": (
            "Schwefel",
            schwefel,
            grad_schwefel,
            r"f(x)=418.9829\,n-\sum_{i=1}^n x_i\sin(\sqrt{|x_i|}).",
        ),
        "griewank": (
            "Griewank",
            griewank,
            grad_griewank,
            r"f(x)=1+\frac{1}{4000}\sum_{i=1}^n x_i^2-\prod_{i=1}^n\cos(x_i/\sqrt{i}).",
        ),
        "levy": (
            "Levy",
            levy,
            grad_levy,
            r"f(x)=\sin^2(\pi w_1)+\sum_{i=1}^{n-1}(w_i-1)^2[1+10\sin^2(\pi w_i+1)]"
            r"+(w_n-1)^2[1+\sin^2(2\pi w_n)],\ w_i=1+(x_i-1)/4.",
        ),
        "discus": (
            "Discus",
            discus,
            grad_discus,
            r"f(x)=10^6 x_1^2+\sum_{i=2}^n x_i^2.",
        ),
        "bent_cigar": (
            "Bent Cigar",
            bent_cigar,
            grad_bent_cigar,
            r"f(x)=x_1^2+10^6\sum_{i=2}^n x_i^2.",
        ),
        "zakharov": (
            "Zakharov",
            zakharov,
            grad_zakharov,
            r"f(x)=\sum_{i=1}^n x_i^2+\left(\frac12\sum_{i=1}^n i x_i\right)^2"
            r"+\left(\frac12\sum_{i=1}^n i x_i\right)^4.",
        ),
        "dixon_price": (
            "Dixon-Price",
            dixon_price,
            grad_dixon_price,
            r"f(x)=(x_1-1)^2+\sum_{i=2}^n i(2x_i^2-x_{i-1})^2.",
        ),
        "ellipsoidal": (
            "Ellipsoidal",
            ellipsoidal,
            grad_ellipsoidal,
            r"f(x)=\sum_{i=1}^n\left(10^6\right)^{\frac{i-1}{n-1}} x_i^2.",
        ),
    }
    if name not in table:
        raise KeyError(f"Unknown problem {name!r}")
    return table[name]


def make_x0(dim: int, problem_key: str, seed_offset: int) -> Vector:
    from config import PROBLEM_SEED

    rng = np.random.default_rng(PROBLEM_SEED[problem_key] + dim + seed_offset)
    return rng.uniform(-3.0, 3.0, size=dim).astype(float)
