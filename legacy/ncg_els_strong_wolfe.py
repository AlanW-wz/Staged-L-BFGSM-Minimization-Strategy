# -*- coding: utf-8 -*-
"""
Nonlinear Conjugate Gradient (NCG) with ELS beta update + strong Wolfe line search.

Implemented from the provided algorithm snapshot:
  d_0 = -g_0
  r_{k+1} = r_k + alpha_k d_k
  g_{k+1} = grad U(r_{k+1})
  d_{k+1} = -g_{k+1} + beta_{k+1} d_k

ELS coefficient (using user's parameters):
  mu = 1/4, v = 1/2, omega = 4/5

beta_k^{ELS} =
  ( ||g_k||^2 - (||g_k|| / ||g_{k-1}||) * |g_k^T g_{k-1}| )
  / ( mu * |g_k^T d_{k-1}| + omega * ||g_{k-1}||^2 + v * d_{k-1}^T y_{k-1} )

where y_{k-1} = g_k - g_{k-1}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
import time

import numpy as np
from scipy.optimize import line_search

Vector = np.ndarray
ObjectiveFn = Callable[[Vector], float]
GradientFn = Callable[[Vector], Vector]


@dataclass
class NCGResult:
    x: Vector
    f: float
    grad_norm: float
    n_iter: int
    converged: bool
    f_history: List[float] = field(default_factory=list)
    x_history: List[Vector] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)


def beta_els(
    gk: Vector,
    gkm1: Vector,
    dkm1: Vector,
    *,
    mu: float = 0.25,
    v: float = 0.5,
    omega: float = 0.8,
    eps: float = 1e-16,
) -> float:
    """Compute ELS beta_k coefficient."""
    ngk = float(np.linalg.norm(gk))
    ngkm1 = float(np.linalg.norm(gkm1))
    if ngkm1 <= eps:
        return 0.0

    num = ngk * ngk - (ngk / ngkm1) * abs(float(gk @ gkm1))

    ykm1 = gk - gkm1
    den = (
        mu * abs(float(gk @ dkm1))
        + omega * (ngkm1 * ngkm1)
        + v * float(dkm1 @ ykm1)
    )

    if den <= eps:
        # restart to steepest descent when denominator is non-positive/degenerate
        return 0.0

    return float(num / den)


def strong_wolfe_step(
    f: ObjectiveFn,
    grad_f: GradientFn,
    xk: Vector,
    dk: Vector,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    maxiter: int = 80,
) -> float:
    """Get step size by strong Wolfe line search; fallback Armijo backtracking."""
    gk = grad_f(xk)
    gtd = float(gk @ dk)
    if gtd >= 0.0:
        return 1e-8

    out = line_search(
        f,
        grad_f,
        xk,
        dk,
        gfk=gk,
        old_fval=float(f(xk)),
        c1=c1,
        c2=c2,
        maxiter=maxiter,
        amax=1e50,
    )
    alpha = out[0]
    if alpha is not None and alpha > 0:
        return float(alpha)

    # fallback: Armijo backtracking
    alpha = 1.0
    fk = float(f(xk))
    for _ in range(60):
        xn = xk + alpha * dk
        if float(f(xn)) <= fk + c1 * alpha * gtd:
            return alpha
        alpha *= 0.5
    return alpha


def ncg_els(
    f: ObjectiveFn,
    grad_f: GradientFn,
    x0: Vector,
    *,
    max_iter: int = 5000,
    eps_f: float = 1e-10,
    eps_g: float = 1e-8,
    mu: float = 0.25,
    v: float = 0.5,
    omega: float = 0.8,
    collect_grad_norm_history: bool = False,
    collect_time_history: bool = False,
) -> NCGResult:
    """
    NCG with ELS beta update and strong Wolfe line search.

    Convergence check follows the provided algorithm spirit:
      f_{k+1} <= f_k and |(f_{k+1}-f_k)/f_{k+1}| <= eps_f and ||g_k|| <= eps_g
    """
    xk = np.asarray(x0, dtype=float).copy()
    gk = grad_f(xk)
    dk = -gk

    fk = float(f(xk))
    f_hist = [fk]
    x_hist = [xk.copy()]
    grad_norm_history: List[float] = []
    time_history: List[float] = []
    t_start = time.perf_counter()
    if collect_grad_norm_history:
        grad_norm_history.append(float(np.linalg.norm(gk)))
    if collect_time_history:
        time_history.append(0.0)

    for k in range(max_iter):
        ngk = float(np.linalg.norm(gk))
        if collect_grad_norm_history and k > 0:
            grad_norm_history.append(ngk)
        if collect_time_history and k > 0:
            time_history.append(float(time.perf_counter() - t_start))
        if ngk <= eps_g:
            return NCGResult(
                x=xk,
                f=fk,
                grad_norm=ngk,
                n_iter=k,
                converged=True,
                f_history=f_hist,
                x_history=x_hist,
                grad_norm_history=grad_norm_history,
                time_history=time_history,
            )

        # restart if not a descent direction
        if float(gk @ dk) >= 0.0:
            dk = -gk

        alpha = strong_wolfe_step(f, grad_f, xk, dk)
        x_next = xk + alpha * dk
        f_next = float(f(x_next))
        g_next = grad_f(x_next)

        # stopping criteria in the style of your figure
        rel_drop = abs((f_next - fk) / max(abs(f_next), 1e-16))
        if f_next <= fk and rel_drop <= eps_f and float(np.linalg.norm(g_next)) <= eps_g:
            f_hist.append(f_next)
            x_hist.append(x_next.copy())
            gn = float(np.linalg.norm(g_next))
            if collect_grad_norm_history:
                grad_norm_history.append(gn)
            if collect_time_history:
                time_history.append(float(time.perf_counter() - t_start))
            return NCGResult(
                x=x_next,
                f=f_next,
                grad_norm=gn,
                n_iter=k + 1,
                converged=True,
                f_history=f_hist,
                x_history=x_hist,
                grad_norm_history=grad_norm_history,
                time_history=time_history,
            )

        beta = beta_els(g_next, gk, dk, mu=mu, v=v, omega=omega)
        d_next = -g_next + beta * dk

        # safeguard: if update loses descent, restart
        if float(g_next @ d_next) >= 0.0:
            d_next = -g_next

        xk, fk, gk, dk = x_next, f_next, g_next, d_next
        f_hist.append(fk)
        x_hist.append(xk.copy())

    return NCGResult(
        x=xk,
        f=fk,
        grad_norm=float(np.linalg.norm(gk)),
        n_iter=max_iter,
        converged=False,
        f_history=f_hist,
        x_history=x_hist,
        grad_norm_history=grad_norm_history,
        time_history=time_history,
    )


# ---------- demo ----------
def rosenbrock(x: Vector) -> float:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x: Vector) -> Vector:
    dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    dfdy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy], dtype=float)


if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0], dtype=float)
    result = ncg_els(
        rosenbrock,
        grad_rosenbrock,
        x0,
        max_iter=10000,
        eps_f=1e-12,
        eps_g=1e-8,
        mu=0.25,   # 1/4
        v=0.5,     # 1/2
        omega=0.8, # 4/5
    )
    print("Converged:", result.converged)
    print("Iterations:", result.n_iter)
    print("x*:", result.x)
    print("f(x*):", result.f)
    print("||grad||:", result.grad_norm)
