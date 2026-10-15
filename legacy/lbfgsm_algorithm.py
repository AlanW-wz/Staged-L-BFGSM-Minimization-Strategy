# -*- coding: utf-8 -*-
"""
Algorithm LBFGSM: modified inverse L-BFGS (pseudocode from your reference figure).

Structure (matches the figure):
  - index set I, at most m stored pairs (s_j, y_j);
  - omega_k = min(c0, c1 * ||g_k||^c2);
  - pick gamma_k in ([gamma^- , gamma^+] intersect [omega_k, 1/omega_k]) if nonempty,
    else in [omega_k, 1/omega_k];
  - H_k^(0) = gamma_k * I;
  - direction d_k = -H_k g_k using only pairs with q_j >= omega_k (two-loop, chronological);
  - alpha_k > 0 by line search (default: Armijo; strong_wolfe_line_search for Wolfe).

Please confirm from your PDF (see user-facing questions in the assistant message):
  (1) exact q(s,y), (2) Definition 2.1 for gamma bounds if not as below, (3) line search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import time

import numpy as np

Vector = np.ndarray
ObjectiveFn = Callable[[Vector], float]
GradientFn = Callable[[Vector], Vector]
QFn = Callable[[Vector, Vector], float]


@dataclass
class CorrectionPair:
    """One stored pair (s_j, y_j) with iteration index j and cached q_j."""

    index: int
    s: Vector
    y: Vector
    q: float


@dataclass
class LBFGSMResult:
    x: Vector
    f: float
    grad_norm: float
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)
    # x_history[k] = iterate after k updates (x_history[0] is x0); len == len(history)
    x_history: Optional[List[Vector]] = None
    grad_norm_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)


def _scalar_interval_intersection(
    lo1: float, hi1: float, lo2: float, hi2: float
) -> Optional[Tuple[float, float]]:
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    if lo <= hi:
        return lo, hi
    return None


def choose_gamma_in_interval(
    preferred: float, interval: Tuple[float, float]
) -> float:
    """Pick gamma in [lo, hi] by clamping `preferred`."""
    lo, hi = interval
    return float(np.clip(preferred, lo, hi))


def default_q_fn(s: Vector, y: Vector) -> float:
    """
    Placeholder q(s,y) -- NOT in the figure; override with `q_fn` from the paper.

    Default: min( (y^T s)/||s||^2 , (y^T s)/||y||^2 ).
    """
    ys = float(y @ s)
    sn = float(np.linalg.norm(s))
    yn = float(np.linalg.norm(y))
    if sn < 1e-16 or yn < 1e-16:
        return 0.0
    return min(ys / (sn * sn), ys / (yn * yn))


def update_gamma_bounds(s: Vector, y: Vector) -> Tuple[float, float]:
    """
    Candidate Definition 2.1 (same as formulas in your local derivation tex):

      gamma^- = (y^T s) / ||y||^2 ,   gamma^+ = ||s||^2 / (y^T s).
    """
    ys = float(y @ s)
    yn2 = float(y @ y)
    if ys <= 1e-16 or yn2 <= 1e-16:
        return 0.0, float("inf")
    gamma_minus = ys / yn2
    sn2 = float(s @ s)
    gamma_plus = sn2 / ys
    return gamma_minus, gamma_plus


def select_gamma_k(
    gk: Vector,
    gamma_minus: float,
    gamma_plus: float,
    c0: float,
    c1: float,
    c2: float,
    last_preferred_gamma: Optional[float],
) -> float:
    """
    omega_k = min(c0, c1 * ||gk||^c2).
    Try [gamma^-, gamma^+] intersect [omega, 1/omega]; else [omega, 1/omega].
    Preferred: last (s^T y)/(y^T y) if available, else geom. mean of interval ends.
    """
    gnorm = float(np.linalg.norm(gk))
    omega = min(c0, c1 * (gnorm**c2))
    omega = max(omega, 1e-16)
    inv_omega = 1.0 / omega

    inter = _scalar_interval_intersection(
        gamma_minus, gamma_plus, omega, inv_omega
    )
    if inter is not None:
        lo, hi = inter
    else:
        lo, hi = omega, inv_omega

    if last_preferred_gamma is not None and np.isfinite(last_preferred_gamma):
        preferred = last_preferred_gamma
    else:
        preferred = float(np.sqrt(max(lo * hi, 1e-16)))

    return choose_gamma_in_interval(preferred, (lo, hi))


def two_loop_filtered(
    gk: Vector,
    pairs: Sequence[CorrectionPair],
    gamma_k: float,
) -> Vector:
    """L-BFGS two-loop recursion; `pairs` oldest-to-newest."""
    q = gk.copy()
    stack: List[Tuple[Vector, Vector, float, float]] = []

    for p in reversed(pairs):
        s, y = p.s, p.y
        ys = float(y @ s)
        if ys <= 1e-16:
            continue
        rho = 1.0 / ys
        alpha = rho * float(s @ q)
        q = q - alpha * y
        stack.append((s, y, rho, alpha))

    r = gamma_k * q
    for s, y, rho, alpha in reversed(stack):
        beta = rho * float(y @ r)
        r = r + s * (alpha - beta)
    return -r


def armijo_backtracking(
    f: ObjectiveFn,
    grad_f: GradientFn,
    xk: Vector,
    dk: Vector,
    alpha0: float = 1.0,
    rho: float = 0.5,
    sigma: float = 1e-4,
    max_backtracks: int = 60,
) -> float:
    alpha = alpha0
    fk = f(xk)
    gk = grad_f(xk)
    gtd = float(gk @ dk)
    if gtd >= 0.0:
        return 1e-8
    for _ in range(max_backtracks):
        if f(xk + alpha * dk) <= fk + sigma * alpha * gtd:
            return alpha
        alpha *= rho
    return alpha


def strong_wolfe_line_search(
    f: ObjectiveFn,
    grad_f: GradientFn,
    xk: Vector,
    dk: Vector,
    wolfe_c1: float = 1e-4,
    wolfe_c2: float = 0.9,
    maxiter: int = 80,
) -> float:
    """
    Strong Wolfe line search (Powell's modification) via scipy.optimize.line_search.

    Satisfies (standard notation, descent direction d, phi(alpha)=f(x+alpha d)):
      Armijo:   f(x+alpha d) <= f(x) + wolfe_c1 * alpha * d^T grad f(x)
      Curvature: |d^T grad f(x+alpha d)| <= wolfe_c2 * |d^T grad f(x)|

    Falls back to Armijo if SciPy is missing or the search fails.
    """
    gk = grad_f(xk)
    fk = float(f(xk))
    gtd = float(gk @ dk)
    if gtd >= 0.0:
        return 1e-8
    try:
        from scipy.optimize import line_search as _line_search
    except ImportError:
        return armijo_backtracking(f, grad_f, xk, dk)
    out = _line_search(
        f,
        grad_f,
        xk,
        dk,
        gfk=gk,
        old_fval=fk,
        c1=wolfe_c1,
        c2=wolfe_c2,
        amax=1e50,
        maxiter=maxiter,
    )
    alpha = out[0]
    if alpha is None:
        return armijo_backtracking(f, grad_f, xk, dk)
    return float(alpha)


def lbfgsm(
    f: ObjectiveFn,
    grad_f: GradientFn,
    x0: Vector,
    m: int,
    c0: float,
    c1: float,
    c2: float,
    tol: float = 1e-9,
    max_iter: int = 2000,
    q_fn: QFn = default_q_fn,
    line_search_fn: Callable[
        [ObjectiveFn, GradientFn, Vector, Vector], float
    ] = armijo_backtracking,
    collect_x_history: bool = False,
    collect_grad_norm_history: bool = False,
    collect_time_history: bool = False,
) -> LBFGSMResult:
    """LBFGSM main loop (figure + bound update; see module doc)."""
    xk = np.asarray(x0, dtype=float).copy()
    pairs: List[CorrectionPair] = []
    gamma_minus = 0.0
    gamma_plus = float("inf")
    last_preferred_gamma: Optional[float] = None
    history: List[float] = [float(f(xk))]
    x_history: Optional[List[Vector]] = [xk.copy()] if collect_x_history else None
    grad_norm_history: List[float] = []
    time_history: List[float] = []
    t_start = time.perf_counter()

    for k in range(max_iter):
        gk = grad_f(xk)
        gnorm = float(np.linalg.norm(gk))
        if collect_grad_norm_history:
            grad_norm_history.append(gnorm)
        if collect_time_history:
            time_history.append(float(time.perf_counter() - t_start))
        if gnorm <= tol:
            return LBFGSMResult(
                x=xk,
                f=float(f(xk)),
                grad_norm=gnorm,
                n_iter=k,
                converged=True,
                history=history,
                x_history=x_history,
                grad_norm_history=grad_norm_history,
                time_history=time_history,
            )

        omega_k = min(c0, c1 * (gnorm**c2))
        omega_k = max(omega_k, 1e-16)

        gamma_k = select_gamma_k(
            gk, gamma_minus, gamma_plus, c0, c1, c2, last_preferred_gamma
        )

        filtered = [p for p in pairs if p.q >= omega_k]
        dk = two_loop_filtered(gk, filtered, gamma_k)

        alpha_k = line_search_fn(f, grad_f, xk, dk)
        sk = alpha_k * dk
        x_next = xk + sk
        yk = grad_f(x_next) - gk

        if float(yk @ sk) > 0.0:
            qk = float(q_fn(sk, yk))
            pairs.append(CorrectionPair(index=k, s=sk.copy(), y=yk.copy(), q=qk))
            gamma_minus, gamma_plus = update_gamma_bounds(sk, yk)
            ys = float(yk @ sk)
            yy = float(yk @ yk)
            if yy > 1e-16:
                last_preferred_gamma = ys / yy
            else:
                last_preferred_gamma = None
        else:
            gamma_minus, gamma_plus = 0.0, float("inf")
            last_preferred_gamma = None

        if len(pairs) > m:
            oldest = min(p.index for p in pairs)
            pairs = [p for p in pairs if p.index != oldest]

        xk = x_next
        history.append(float(f(xk)))
        if x_history is not None:
            x_history.append(xk.copy())

    gf = grad_f(xk)
    return LBFGSMResult(
        x=xk,
        f=float(f(xk)),
        grad_norm=float(np.linalg.norm(gf)),
        n_iter=max_iter,
        converged=False,
        history=history,
        x_history=x_history,
        grad_norm_history=grad_norm_history,
        time_history=time_history,
    )


def _demo() -> None:
    from lbfgs_from_tex import grad_rosenbrock, rosenbrock

    x0 = np.array([-1.2, 1.0], dtype=float)
    res = lbfgsm(
        rosenbrock,
        grad_rosenbrock,
        x0,
        m=8,
        c0=0.9,
        c1=1.0,
        c2=1.0,
        tol=1e-8,
        max_iter=3000,
    )
    print("LBFGSM demo (defaults may need your paper's q / line search):")
    print("  converged:", res.converged)
    print("  iters:", res.n_iter)
    print("  x:", res.x)
    print("  f:", res.f)
    print("  ||g||:", res.grad_norm)


if __name__ == "__main__":
    _demo()
