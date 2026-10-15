"""
L-BFGS implementation based on formulas in `公式推导.tex`.

Key points mapped from the document:
1) Search direction is computed by the two-loop recursion:
   d_k = -H_k g_k, with H_k^0 = gamma_k I.
2) Keep only the latest m curvature pairs (s_i, y_i).
3) Use backtracking line search with Armijo condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


Vector = np.ndarray
ObjectiveFn = Callable[[Vector], float]
GradientFn = Callable[[Vector], Vector]


@dataclass
class LBFGSResult:
    x: Vector
    f: float
    grad_norm: float
    n_iter: int
    converged: bool
    history: List[float]


def two_loop_recursion(
    gk: Vector,
    s_history: List[Vector],
    y_history: List[Vector],
    gamma_k: float,
) -> Vector:
    """
    Compute direction d_k = -H_k g_k using L-BFGS two-loop recursion.
    """
    q = gk.copy()
    # Keep valid pairs and their alpha/rho in the same order as the backward loop.
    valid_pairs: List[Tuple[Vector, Vector, float, float]] = []

    # First loop: backward
    for s, y in zip(reversed(s_history), reversed(y_history)):
        ys = float(y @ s)
        if ys <= 1e-16:
            # Skip non-positive/degenerate curvature pair for stability.
            continue
        rho = 1.0 / ys
        alpha = rho * float(s @ q)
        q = q - alpha * y
        valid_pairs.append((s, y, rho, alpha))

    # Initial Hessian multiplication: r = H_k^0 q = gamma_k * q
    r = gamma_k * q

    # Second loop: forward (reverse back the stored loop order)
    for s, y, rho, alpha in reversed(valid_pairs):
        beta = rho * float(y @ r)
        r = r + s * (alpha - beta)

    dk = -r
    return dk


def armijo_backtracking(
    f: ObjectiveFn,
    grad_f: GradientFn,
    xk: Vector,
    dk: Vector,
    alpha0: float = 1.0,
    rho: float = 0.5,
    sigma: float = 1e-4,
    max_backtracks: int = 50,
) -> float:
    """
    Backtracking line search under Armijo condition:
    f(xk + alpha*dk) <= f(xk) + sigma * alpha * gk^T dk
    """
    alpha = alpha0
    fk = f(xk)
    gk = grad_f(xk)
    gTd = float(gk @ dk)

    if gTd >= 0:
        # Direction is not descent; force tiny step.
        return 1e-8

    for _ in range(max_backtracks):
        x_new = xk + alpha * dk
        if f(x_new) <= fk + sigma * alpha * gTd:
            return alpha
        alpha *= rho
    return alpha


def lbfgs(
    f: ObjectiveFn,
    grad_f: GradientFn,
    x0: Vector,
    m: int = 10,
    tol: float = 1e-6,
    max_iter: int = 500,
    alpha0: float = 1.0,
    rho: float = 0.5,
    sigma: float = 1e-4,
) -> LBFGSResult:
    """
    Limited-memory BFGS optimizer.
    """
    xk = x0.astype(float).copy()
    s_history: List[Vector] = []
    y_history: List[Vector] = []
    history: List[float] = [f(xk)]

    for k in range(max_iter):
        gk = grad_f(xk)
        gnorm = float(np.linalg.norm(gk))
        if gnorm <= tol:
            return LBFGSResult(
                x=xk,
                f=f(xk),
                grad_norm=gnorm,
                n_iter=k,
                converged=True,
                history=history,
            )

        # gamma_k = (s^T y) / (y^T y), from last pair; fallback 1.0
        if s_history:
            s_last = s_history[-1]
            y_last = y_history[-1]
            yTy = float(y_last @ y_last)
            if yTy > 1e-16:
                gamma_k = float((s_last @ y_last) / yTy)
            else:
                gamma_k = 1.0
        else:
            gamma_k = 1.0

        dk = two_loop_recursion(gk, s_history, y_history, gamma_k)

        # Armijo backtracking line search
        alpha_k = armijo_backtracking(
            f, grad_f, xk, dk, alpha0=alpha0, rho=rho, sigma=sigma
        )

        x_next = xk + alpha_k * dk
        g_next = grad_f(x_next)

        sk = x_next - xk
        yk = g_next - gk

        # Curvature condition: s_k^T y_k > 0
        if float(sk @ yk) > 1e-12:
            s_history.append(sk)
            y_history.append(yk)
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)

        xk = x_next
        history.append(f(xk))

    g_final = grad_f(xk)
    return LBFGSResult(
        x=xk,
        f=f(xk),
        grad_norm=float(np.linalg.norm(g_final)),
        n_iter=max_iter,
        converged=False,
        history=history,
    )


def rosenbrock(x: Vector) -> float:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x: Vector) -> Vector:
    dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    dfdy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy], dtype=float)


if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0], dtype=float)
    result = lbfgs(
        f=rosenbrock,
        grad_f=grad_rosenbrock,
        x0=x0,
        m=8,
        tol=1e-8,
        max_iter=2000,
        alpha0=1.0,
        rho=0.5,
        sigma=1e-4,
    )

    print("Converged:", result.converged)
    print("Iterations:", result.n_iter)
    print("x*:", result.x)
    print("f(x*):", result.f)
    print("||grad||:", result.grad_norm)
