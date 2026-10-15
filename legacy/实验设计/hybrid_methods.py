from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lbfgsm_algorithm import (
    CorrectionPair,
    armijo_backtracking,
    default_q_fn,
    lbfgsm,
    select_gamma_k,
    strong_wolfe_line_search,
    two_loop_filtered,
    update_gamma_bounds,
)
from ncg_els_strong_wolfe import beta_els, ncg_els

Vector = np.ndarray
Objective = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]


@dataclass
class MethodResult:
    method: str
    problem: str
    dimension: int
    converged: bool
    iterations: int
    f_x: float
    grad_norm: float
    elapsed_seconds: float
    x_norm: float
    switch_sd_to_mid_iter: Optional[int] = None
    switch_mid_to_final_iter: Optional[int] = None
    grad_history: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)


def rastrigin(x: Vector, a: float = 10.0) -> float:
    n = x.size
    return float(a * n + np.sum(x * x - a * np.cos(2.0 * np.pi * x)))


def grad_rastrigin(x: Vector, a: float = 10.0) -> Vector:
    return 2.0 * x + 2.0 * np.pi * a * np.sin(2.0 * np.pi * x)


def _all_last(vals: List[float], window: int, predicate: Callable[[float], bool]) -> bool:
    if len(vals) < window:
        return False
    return all(predicate(v) for v in vals[-window:])


def _two_loop_standard(gk: Vector, s_hist: Sequence[Vector], y_hist: Sequence[Vector]) -> Vector:
    q = gk.copy()
    stack: List[Tuple[Vector, Vector, float, float]] = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        ys = float(y @ s)
        if ys <= 1e-16:
            continue
        rho = 1.0 / ys
        alpha = rho * float(s @ q)
        q = q - alpha * y
        stack.append((s, y, rho, alpha))
    if s_hist:
        s_last = s_hist[-1]
        y_last = y_hist[-1]
        yTy = float(y_last @ y_last)
        gamma = float((s_last @ y_last) / yTy) if yTy > 1e-16 else 1.0
    else:
        gamma = 1.0
    r = gamma * q
    for s, y, rho, alpha in reversed(stack):
        beta = rho * float(y @ r)
        r = r + s * (alpha - beta)
    return -r


def run_sd_ncg(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    switch_iter = None
    alpha_hist: List[float] = []
    prev_g = None
    prev_d = None
    t_start = time.perf_counter()

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, switch_iter
        if phase == "SD" and k >= 10:
            grad_ok = (ng / max(g0, 1e-16)) <= 0.2
            stable_ok = _all_last(alpha_hist, 3, lambda a: a > 1e-6)
            ortho_ok = False
            if prev_g is not None:
                ortho = abs(float(gk @ prev_g)) / max(ng * ng, 1e-16)
                ortho_ok = ortho < 0.2
            if grad_ok and stable_ok and ortho_ok:
                phase = "NCG"
                switch_iter = k
        if phase == "SD" or prev_g is None or prev_d is None:
            dk = -gk
        else:
            beta = beta_els(gk, prev_g, prev_d, mu=0.25, v=0.5, omega=0.8)
            dk = -gk + max(0.0, beta) * prev_d
            if float(gk @ dk) >= -1e-12:
                dk = -gk
        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        x_next = xk + alpha * dk
        g_next = grad_f(x_next)
        alpha_hist.append(float(alpha))
        prev_g, prev_d = gk, dk
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, switch_iter


def run_sd_cg(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    switch_iter = None
    alpha_hist: List[float] = []
    prev_g = None
    prev_d = None
    t_start = time.perf_counter()
    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, switch_iter
        if phase == "SD" and k >= 10:
            grad_ok = (ng / max(g0, 1e-16)) <= 0.2
            stable_ok = _all_last(alpha_hist, 3, lambda a: a > 1e-6)
            ortho_ok = False
            if prev_g is not None:
                ortho = abs(float(gk @ prev_g)) / max(ng * ng, 1e-16)
                ortho_ok = ortho < 0.2
            if grad_ok and stable_ok and ortho_ok:
                phase = "CG"
                switch_iter = k
        if phase == "SD" or prev_g is None or prev_d is None:
            dk = -gk
        else:
            den = float(prev_g @ prev_g)
            beta = float((gk @ gk) / den) if den > 1e-16 else 0.0
            dk = -gk + beta * prev_d
            if float(gk @ dk) >= -1e-12:
                dk = -gk
        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        x_next = xk + alpha * dk
        g_next = grad_f(x_next)
        alpha_hist.append(float(alpha))
        prev_g, prev_d = gk, dk
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, switch_iter


def run_sd_lbfgsm(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    m: int = 8,
    c0: float = 0.9,
    c1: float = 1.0,
    c2: float = 1.0,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    switch_iter = None
    alpha_hist: List[float] = []
    pairs: List[CorrectionPair] = []
    gamma_minus, gamma_plus = 0.0, float("inf")
    last_gamma = None
    last_sty_ok = False
    t_start = time.perf_counter()

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, switch_iter
        if phase == "SD" and k >= 10:
            if (ng / max(g0, 1e-16)) <= 0.2 and _all_last(alpha_hist, 3, lambda a: a > 1e-6) and last_sty_ok:
                phase = "LBFGSM"
                switch_iter = k

        if phase == "LBFGSM":
            omega_k = max(min(c0, c1 * (ng**c2)), 1e-16)
            gamma_k = select_gamma_k(gk, gamma_minus, gamma_plus, c0, c1, c2, last_gamma)
            filt = [p for p in pairs if p.q >= omega_k]
            dk = two_loop_filtered(gk, filt, gamma_k)
            if float(gk @ dk) >= -1e-12:
                dk = -gk
        else:
            dk = -gk
        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        sk = alpha * dk
        x_next = xk + sk
        g_next = grad_f(x_next)
        yk = g_next - gk
        sty = float(sk @ yk)
        last_sty_ok = sty > 1e-12
        if sty > 0.0:
            qk = float(default_q_fn(sk, yk))
            pairs.append(CorrectionPair(index=k, s=sk.copy(), y=yk.copy(), q=qk))
            gamma_minus, gamma_plus = update_gamma_bounds(sk, yk)
            yy = float(yk @ yk)
            last_gamma = float((yk @ sk) / yy) if yy > 1e-16 else None
            if len(pairs) > m:
                oldest = min(p.index for p in pairs)
                pairs = [p for p in pairs if p.index != oldest]
        alpha_hist.append(float(alpha))
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, switch_iter


def run_sd_lbfgs_standard(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    m: int = 8,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    switch_iter = None
    alpha_hist: List[float] = []
    s_hist: List[Vector] = []
    y_hist: List[Vector] = []
    last_sty_ok = False
    t_start = time.perf_counter()

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, switch_iter
        if phase == "SD" and k >= 10:
            grad_ok = (ng / max(g0, 1e-16)) <= 0.2
            stable_ok = _all_last(alpha_hist, 3, lambda a: a > 1e-6)
            if grad_ok and stable_ok and last_sty_ok:
                phase = "LBFGS"
                switch_iter = k

        if phase == "LBFGS":
            dk = _two_loop_standard(gk, s_hist, y_hist)
            if float(gk @ dk) >= -1e-12:
                dk = -gk
        else:
            dk = -gk

        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        sk = alpha * dk
        x_next = xk + sk
        g_next = grad_f(x_next)
        yk = g_next - gk
        sty = float(sk @ yk)
        last_sty_ok = sty > 1e-12
        if sty > 1e-16:
            s_hist.append(sk.copy())
            y_hist.append(yk.copy())
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        alpha_hist.append(float(alpha))
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, switch_iter


def run_sd_bfgs_standard(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    switch_iter = None
    alpha_hist: List[float] = []
    s_hist: List[Vector] = []
    y_hist: List[Vector] = []
    last_sty_ok = False
    t_start = time.perf_counter()

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, switch_iter
        if phase == "SD" and k >= 10:
            grad_ok = (ng / max(g0, 1e-16)) <= 0.2
            stable_ok = _all_last(alpha_hist, 3, lambda a: a > 1e-6)
            if grad_ok and stable_ok and last_sty_ok:
                phase = "BFGS"
                switch_iter = k

        if phase == "BFGS":
            # Full-memory BFGS direction represented by all stored curvature pairs.
            dk = _two_loop_standard(gk, s_hist, y_hist)
            if float(gk @ dk) >= -1e-12:
                dk = -gk
        else:
            dk = -gk

        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        sk = alpha * dk
        x_next = xk + sk
        g_next = grad_f(x_next)
        yk = g_next - gk
        sty = float(sk @ yk)
        last_sty_ok = sty > 1e-12
        if sty > 1e-16:
            s_hist.append(sk.copy())
            y_hist.append(yk.copy())
        alpha_hist.append(float(alpha))
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, switch_iter


def run_lbfgs_standard(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    m: int = 8,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float]:
    xk = x0.copy()
    gk = grad_f(xk)
    s_hist: List[Vector] = []
    y_hist: List[Vector] = []
    t_start = time.perf_counter()

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng

        dk = _two_loop_standard(gk, s_hist, y_hist)
        if float(gk @ dk) >= -1e-12:
            dk = -gk
        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        sk = alpha * dk
        x_next = xk + sk
        g_next = grad_f(x_next)
        yk = g_next - gk
        if float(sk @ yk) > 1e-16:
            s_hist.append(sk.copy())
            y_hist.append(yk.copy())
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng


def run_sd_mid_final(
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
    mid: str,
    final: str,
    strategy5: bool = False,
    c0: float = 0.9,
    c1: float = 1.0,
    c2: float = 1.0,
    grad_hist: Optional[List[float]] = None,
    time_hist: Optional[List[float]] = None,
) -> Tuple[bool, int, Vector, float, float, Optional[int], Optional[int]]:
    xk = x0.copy()
    gk = grad_f(xk)
    g0 = float(np.linalg.norm(gk))
    phase = "SD"
    sw1 = None
    sw2 = None
    alpha_hist: List[float] = []
    curv_hist: List[float] = []
    res_hist: List[float] = []
    prev_g = None
    prev_d = None
    s_hist: List[Vector] = []
    y_hist: List[Vector] = []
    cg_like_iters = 0
    last_probe = -10**9
    gamma_minus, gamma_plus = 0.0, float("inf")
    last_gamma = None
    t_start = time.perf_counter()

    def mid_direction() -> Vector:
        if mid == "NCG":
            if prev_g is None or prev_d is None:
                return -gk
            b = beta_els(gk, prev_g, prev_d, mu=0.25, v=0.5, omega=0.8)
            d = -gk + max(0.0, b) * prev_d
            return d if float(gk @ d) < -1e-12 else -gk
        # CG FR
        if prev_g is None or prev_d is None:
            return -gk
        den = float(prev_g @ prev_g)
        b = float((gk @ gk) / den) if den > 1e-16 else 0.0
        d = -gk + b * prev_d
        return d if float(gk @ d) < -1e-12 else -gk

    for k in range(max_iter):
        ng = float(np.linalg.norm(gk))
        if grad_hist is not None:
            grad_hist.append(ng)
        if time_hist is not None:
            time_hist.append(float(time.perf_counter() - t_start))
        if ng <= tol:
            return True, k, xk, float(f(xk)), ng, sw1, sw2
        if phase == "SD" and k >= 10:
            grad_ok = (ng / max(g0, 1e-16)) <= 0.2
            stable_ok = _all_last(alpha_hist, 3, lambda a: a > 1e-6)
            ortho_ok = False
            if prev_g is not None:
                ortho = abs(float(gk @ prev_g)) / max(ng * ng, 1e-16)
                ortho_ok = ortho < 0.2
            if grad_ok and stable_ok and ortho_ok:
                phase = mid
                sw1 = k
        if phase == mid:
            cg_like_iters += 1
            grad_ok = (ng / max(g0, 1e-16)) <= 0.005
            stable_ok = _all_last(alpha_hist, 6, lambda a: a > 1e-6)
            curv_ok = _all_last(curv_hist, 6, lambda c: c > 0.05)
            res_ok = _all_last(res_hist, 6, lambda r: r <= 1e-6)
            pair_ok = len(s_hist) >= 5
            if strategy5:
                can_probe = (
                    cg_like_iters >= 200
                    and (ng / max(g0, 1e-16)) <= 0.02
                    and pair_ok
                    and _all_last(alpha_hist, 6, lambda a: a > 1e-6)
                    and (k - last_probe) >= 50
                )
                if can_probe:
                    # 5-step cheap probe
                    f_mid = float(f(xk))
                    x_t = xk.copy()
                    g_t = gk.copy()
                    pg_t = None if prev_g is None else prev_g.copy()
                    pd_t = None if prev_d is None else prev_d.copy()
                    for _ in range(5):
                        if mid == "NCG":
                            if pg_t is None or pd_t is None:
                                d_t = -g_t
                            else:
                                b_t = beta_els(g_t, pg_t, pd_t, mu=0.25, v=0.5, omega=0.8)
                                d_t = -g_t + max(0.0, b_t) * pd_t
                                if float(g_t @ d_t) >= -1e-12:
                                    d_t = -g_t
                        else:
                            if pg_t is None or pd_t is None:
                                d_t = -g_t
                            else:
                                den_t = float(pg_t @ pg_t)
                                b_t = float((g_t @ g_t) / den_t) if den_t > 1e-16 else 0.0
                                d_t = -g_t + b_t * pd_t
                                if float(g_t @ d_t) >= -1e-12:
                                    d_t = -g_t
                        a_t = strong_wolfe_line_search(f, grad_f, x_t, d_t)
                        xn_t = x_t + a_t * d_t
                        gn_t = grad_f(xn_t)
                        pg_t, pd_t = g_t, d_t
                        x_t, g_t = xn_t, gn_t
                    f_mid = float(f(x_t))

                    x_l = xk.copy()
                    g_l = gk.copy()
                    s_l = [s.copy() for s in s_hist]
                    y_l = [y.copy() for y in y_hist]
                    for _ in range(5):
                        d_l = _two_loop_standard(g_l, s_l, y_l)
                        if float(g_l @ d_l) >= -1e-12:
                            d_l = -g_l
                        a_l = strong_wolfe_line_search(f, grad_f, x_l, d_l)
                        s_step = a_l * d_l
                        xn_l = x_l + s_step
                        gn_l = grad_f(xn_l)
                        y_step = gn_l - g_l
                        if float(s_step @ y_step) > 1e-16:
                            s_l.append(s_step.copy())
                            y_l.append(y_step.copy())
                            if len(s_l) > 8:
                                s_l.pop(0)
                                y_l.pop(0)
                        x_l, g_l = xn_l, gn_l
                    f_final = float(f(x_l))
                    last_probe = k
                    if f_final + 1e-12 < f_mid:
                        phase = final
                        sw2 = k
            else:
                if grad_ok and stable_ok and curv_ok and res_ok and pair_ok:
                    phase = final
                    sw2 = k

        if phase == "SD":
            dk = -gk
        elif phase == mid:
            dk = mid_direction()
        else:
            # standard LBFGS for CG branch, filtered LBFGSM for NCG branch
            if final == "LBFGS":
                dk = _two_loop_standard(gk, s_hist, y_hist)
            else:
                omega_k = max(min(c0, c1 * (ng**c2)), 1e-16)
                gamma_k = select_gamma_k(gk, gamma_minus, gamma_plus, c0, c1, c2, last_gamma)
                pairs = [
                    CorrectionPair(index=i, s=s_hist[i], y=y_hist[i], q=float(default_q_fn(s_hist[i], y_hist[i])))
                    for i in range(len(s_hist))
                ]
                filt = [p for p in pairs if p.q >= omega_k]
                dk = two_loop_filtered(gk, filt, gamma_k)
            if float(gk @ dk) >= -1e-12:
                dk = -gk

        alpha = strong_wolfe_line_search(f, grad_f, xk, dk)
        sk = alpha * dk
        x_next = xk + sk
        g_next = grad_f(x_next)
        yk = g_next - gk
        f_next = float(f(x_next))
        sty = float(sk @ yk)
        sn = float(np.linalg.norm(sk))
        yn = float(np.linalg.norm(yk))
        curv = sty / (sn * yn) if sn > 1e-16 and yn > 1e-16 else -1.0
        resid = abs(f_next - float(f(xk)) - float(gk @ sk) - 0.5 * float(sk @ yk)) / max(1.0, abs(float(f(xk))))
        if sty > 1e-16:
            s_hist.append(sk.copy())
            y_hist.append(yk.copy())
            if len(s_hist) > 8:
                s_hist.pop(0)
                y_hist.pop(0)
            gamma_minus, gamma_plus = update_gamma_bounds(sk, yk)
            yy = float(yk @ yk)
            last_gamma = float((yk @ sk) / yy) if yy > 1e-16 else None
        alpha_hist.append(float(alpha))
        curv_hist.append(float(curv))
        res_hist.append(float(resid))
        prev_g, prev_d = gk, dk
        xk, gk = x_next, g_next
    ng = float(np.linalg.norm(gk))
    return False, max_iter, xk, float(f(xk)), ng, sw1, sw2


def run_method_on_problem(
    method: str,
    problem: str,
    f: Objective,
    grad_f: Gradient,
    x0: Vector,
    max_iter: int,
    tol: float,
) -> Tuple[MethodResult, Vector]:
    dim = x0.size
    grad_hist: List[float] = []
    time_hist: List[float] = []
    start = time.perf_counter()
    if method == "lbfgsm_armijo":
        res = lbfgsm(
            f,
            grad_f,
            x0,
            m=8,
            c0=0.9,
            c1=1.0,
            c2=1.0,
            tol=tol,
            max_iter=max_iter,
            line_search_fn=armijo_backtracking,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method,
            problem,
            dim,
            res.converged,
            res.n_iter,
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )
        return out, res.x.copy()
    if method == "lbfgsm_wolfe":
        res = lbfgsm(
            f,
            grad_f,
            x0,
            m=8,
            c0=0.9,
            c1=1.0,
            c2=1.0,
            tol=tol,
            max_iter=max_iter,
            line_search_fn=strong_wolfe_line_search,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method,
            problem,
            dim,
            res.converged,
            res.n_iter,
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )
        return out, res.x.copy()
    if method == "lbfgsm_wolfe_c025_c10":
        res = lbfgsm(
            f,
            grad_f,
            x0,
            m=8,
            c0=0.25,
            c1=1.0,
            c2=10.0,
            tol=tol,
            max_iter=max_iter,
            line_search_fn=strong_wolfe_line_search,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method,
            problem,
            dim,
            res.converged,
            res.n_iter,
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )
        return out, res.x.copy()
    if method == "lbfgsm_wolfe_c025":
        res = lbfgsm(
            f,
            grad_f,
            x0,
            m=8,
            c0=0.25,
            c1=1.0,
            c2=1.0,
            tol=tol,
            max_iter=max_iter,
            line_search_fn=strong_wolfe_line_search,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method,
            problem,
            dim,
            res.converged,
            res.n_iter,
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )
        return out, res.x.copy()
    if method == "lbfgs_standard":
        conv, iters, x, fx, gn = run_lbfgs_standard(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), None, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "ncg_wolfe":
        res = ncg_els(
            f,
            grad_f,
            x0,
            max_iter=max_iter,
            eps_f=1e-12,
            eps_g=tol,
            mu=0.25,
            v=0.5,
            omega=0.8,
            collect_grad_norm_history=True,
            collect_time_history=True,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method,
            problem,
            dim,
            res.converged,
            res.n_iter,
            float(res.f),
            float(res.grad_norm),
            float(elapsed),
            float(np.linalg.norm(res.x)),
            None,
            None,
            list(res.grad_norm_history),
            list(res.time_history),
        )
        return out, res.x.copy()
    if method == "sd_ncg":
        conv, iters, x, fx, gn, sw = run_sd_ncg(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_lbfgsm":
        conv, iters, x, fx, gn, sw = run_sd_lbfgsm(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_lbfgsm_c025_c10":
        conv, iters, x, fx, gn, sw = run_sd_lbfgsm(
            f, grad_f, x0, max_iter, tol, c0=0.25, c1=1.0, c2=10.0,
            grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_lbfgsm_c025":
        conv, iters, x, fx, gn, sw = run_sd_lbfgsm(
            f, grad_f, x0, max_iter, tol, c0=0.25, c1=1.0, c2=1.0,
            grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_lbfgs_standard":
        conv, iters, x, fx, gn, sw = run_sd_lbfgs_standard(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_bfgs_standard":
        conv, iters, x, fx, gn, sw = run_sd_bfgs_standard(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_ncg_lbfgsm":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="NCG", final="LBFGSM",
            strategy5=False, grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_ncg_lbfgsm_c052":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="NCG", final="LBFGSM",
            strategy5=False, c0=0.5, c1=1.0, c2=2.0,
            grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_ncg_lbfgsm_c025":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="NCG", final="LBFGSM",
            strategy5=False, c0=0.25, c1=1.0, c2=1.0,
            grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_ncg_lbfgsm_c025_c10":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="NCG", final="LBFGSM",
            strategy5=False, c0=0.25, c1=1.0, c2=10.0,
            grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_ncg_lbfgsm_strategy5":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="NCG", final="LBFGSM",
            strategy5=True, grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_cg_standard":
        conv, iters, x, fx, gn, sw1 = run_sd_cg(
            f, grad_f, x0, max_iter, tol, grad_hist=grad_hist, time_hist=time_hist
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, None, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_cg_lbfgs":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="CG", final="LBFGS",
            strategy5=False, grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    if method == "sd_cg_lbfgs_strategy5":
        conv, iters, x, fx, gn, sw1, sw2 = run_sd_mid_final(
            f, grad_f, x0, max_iter, tol, mid="CG", final="LBFGS",
            strategy5=True, grad_hist=grad_hist, time_hist=time_hist,
        )
        elapsed = time.perf_counter() - start
        out = MethodResult(
            method, problem, dim, conv, iters, fx, gn, float(elapsed),
            float(np.linalg.norm(x)), sw1, sw2, list(grad_hist), list(time_hist),
        )
        return out, x.copy()
    raise ValueError(method)
