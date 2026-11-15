"""Six local-quench methods, with a shared joint-call budget and line search.

stage[0] is the initial method. Each subsequent stage labels the direction used
to reach that accepted endpoint. Switch indices indicate the already accepted
iteration and its cumulative evaluation count, before taking the new direction.
"""
from collections import deque
from math import isfinite, sqrt

import numpy as np


SD, NCG, LBFGSM, LBFGS = range(4)
STAGE_NAMES = ("SD", "NCG", "LBFGSM", "LBFGS")
METHODS = {
    "M1": (LBFGS,),
    "M2": (NCG, LBFGSM),
    "M3": (SD, LBFGSM),
    "M4": (SD, NCG),
    "M5": (SD, NCG, LBFGSM),
    "M6": (SD, LBFGS),
}


class _BudgetExhausted(Exception):
    pass


def _line_search(evaluate, x, f, g, direction, slope):
    """AQN doubling/bisection strong Wolfe followed by bounded Armijo, all phases."""
    if not isfinite(slope) or slope >= 0:
        return None, "non_descent"
    cache = {}

    def trial(alpha):
        cached = cache.get(alpha)
        if cached is not None:
            return cached
        point = x + alpha * direction
        value, gradient = evaluate(point)
        derivative = float(gradient @ direction)
        sufficient = (isfinite(value) and isfinite(derivative)
                      and value <= f + 1e-4 * alpha * slope)
        outcome = point, value, gradient, alpha, derivative, sufficient
        cache[alpha] = outcome
        return outcome

    prev_a, prev_f, a = 0.0, f, 1.0
    bracket = None
    for spent in range(1, 81):
        t = trial(a)
        if not t[5] or (spent > 1 and t[1] >= prev_f):
            bracket = prev_a, a, prev_f
            break
        if abs(t[4]) <= -0.9 * slope:
            return t, "wolfe"
        if t[4] >= 0:
            bracket = a, prev_a, t[1]
            break
        prev_a, prev_f, a = a, t[1], min(2 * a, 1e50)

    if bracket is not None:
        lo, hi, flo = bracket
        for _ in range(80 - spent):
            a = 0.5 * (lo + hi)
            t = trial(a)
            if not t[5] or t[1] >= flo:
                hi = a
            else:
                if abs(t[4]) <= -0.9 * slope:
                    return t, "wolfe"
                if t[4] * (hi - lo) >= 0:
                    hi = lo
                lo, flo = a, t[1]
            if abs(hi - lo) < 1e-16 * max(1.0, abs(lo)):
                break

    a = 1.0
    for _ in range(60):
        t = trial(a)
        if t[5]:
            return t, "armijo_fallback"
        a *= 0.5
    return None, "line_search_failed"


def _qn_direction(g, gnorm, memory, modified):
    gamma = 1.0
    pairs = memory
    if memory:
        _, _, sy, ss, yy, _ = memory[-1]
        if yy > 1e-16:
            gamma = sy / yy
        if modified:
            omega = max(1e-16, min(0.25, gnorm))
            lo, hi = omega, 1.0 / omega
            if yy > 1e-16:
                left, right = max(lo, sy / yy), min(hi, ss / sy)
                if left <= right:
                    lo, hi = left, right
            gamma = min(hi, max(lo, gamma))
            pairs = [p for p in memory if p[5] >= omega]

    q = g.copy()
    alphas = []
    for s, y, sy, _, _, _ in reversed(pairs):
        a = float(s @ q) / sy
        alphas.append(a)
        q -= a * y
    q *= gamma
    for (s, y, sy, _, _, _), a in zip(pairs, reversed(alphas)):
        q += (a - float(y @ q) / sy) * s
    return -q


def _ncg_direction(g, gnorm, prev_g, prev_norm, prev_d):
    if prev_g is None or prev_norm <= 1e-16:
        return -g
    numerator = gnorm * gnorm - gnorm / prev_norm * abs(float(g @ prev_g))
    denominator = (0.25 * abs(float(g @ prev_d)) + 0.8 * prev_norm ** 2
                   + 0.5 * float(prev_d @ (g - prev_g)))
    beta = max(0.0, numerator / denominator) if denominator > 1e-16 else 0.0
    return -g + beta * prev_d


def _json_number(v):
    return float(v) if isfinite(v) else None


def solve(value_grad, x0, method, max_iterations=2000, max_evaluations=20000, tol=1e-6):
    """Return (JSON-safe metrics, NumPy trajectories) for one initial point."""
    stages = METHODS[method]
    if max_evaluations < 1 or max_iterations < 0 or not isfinite(tol) or tol < 0:
        raise ValueError("Require a positive evaluation budget and nonnegative iteration cap and tolerance")
    x = np.array(x0, dtype=np.float64, copy=True)
    if x.ndim != 1 or not x.size:
        raise ValueError("x0 must be a nonempty vector")

    stage = stages[0]
    stage_i = 0
    stage_steps = [0] * 4
    stage_evals = [0] * 4
    calls = 0

    def evaluate(point):
        nonlocal calls
        if calls >= max_evaluations:
            raise _BudgetExhausted
        calls += 1
        stage_evals[stage] += 1
        value, gradient = value_grad(point)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != point.shape:
            raise ValueError("value_grad must return a gradient with the shape of x0")
        return float(value), gradient

    memory = deque(maxlen=8)
    alphas = deque(maxlen=6)
    cosines = deque(maxlen=6)
    residuals = deque(maxlen=6)
    needs_memory = LBFGSM in stages or LBFGS in stages
    needs_late_gate = NCG in stages and LBFGSM in stages
    switches = []
    fallbacks = restarts = k = 0
    prev_g = prev_d = None
    prev_norm = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        f, g = evaluate(x)
        initial_f = f
        gnorm = initial_norm = float(np.linalg.norm(g))
        grad_inf = initial_grad_inf = float(np.max(np.abs(g)))
        trace_evals, trace_grad = [calls], [grad_inf]
        trace_stage, trace_f = [stage], [f]
        trace_l2 = [gnorm]
        while True:
            if not (isfinite(f) and isfinite(gnorm) and isfinite(grad_inf)):
                reason = "nonfinite"
                break
            if grad_inf <= tol:
                reason = "converged"
                break
            if k >= max_iterations:
                reason = "iteration_limit"
                break
            if calls >= max_evaluations:
                reason = "evaluation_limit"
                break

            # History spans phases, matching the archived direct-switch solver.
            # Both gates may pass at the same accepted endpoint.
            while stage_i + 1 < len(stages):
                if stage == SD:
                    ready = (k >= 10 and gnorm <= 0.2 * max(initial_norm, 1e-16)
                             and min(list(alphas)[-3:]) > 1e-6
                             and abs(float(g @ prev_g)) / max(gnorm * gnorm, 1e-16) < 0.2)
                else:
                    ready = (gnorm <= 0.005 * max(initial_norm, 1e-16)
                             and len(memory) >= 5 and len(alphas) == 6
                             and min(alphas) > 1e-6 and min(cosines) > 0.05
                             and max(residuals) <= 1e-6)
                if not ready:
                    break
                stage_i += 1
                new_stage = stages[stage_i]
                switches.append({"from": STAGE_NAMES[stage], "to": STAGE_NAMES[new_stage],
                                 "iteration": k, "evaluations": calls})
                stage = new_stage

            if stage == SD:
                direction = -g
            elif stage == NCG:
                direction = _ncg_direction(g, gnorm, prev_g, prev_norm, prev_d)
            else:
                direction = _qn_direction(g, gnorm, memory, stage == LBFGSM)
            slope = float(g @ direction)
            if stage != SD and (not isfinite(slope) or slope >= -1e-12):
                direction, slope = -g, -float(g @ g)
                restarts += 1
            try:
                trial, status = _line_search(evaluate, x, f, g, direction, slope)
            except _BudgetExhausted:
                reason = "evaluation_limit"
                break
            if trial is None:
                reason = status
                break
            x_new, f_new, g_new, alpha = trial[:4]
            if needs_memory:
                s = alpha * direction
                y = g_new - g
                sy, ss, yy = float(s @ y), float(s @ s), float(y @ y)
                if sy > 1e-16 and ss > 0 and yy > 0 and isfinite(sy + ss + yy):
                    quality = min(sy / ss, sy / yy) if min(ss, yy) >= 1e-32 else 0.0
                    memory.append((s, y, sy, ss, yy, quality))
                if needs_late_gate and stage != LBFGSM:
                    cosine = sy / (sqrt(ss) * sqrt(yy)) if min(ss, yy) > 1e-32 else -1.
                    residual = abs(f_new - f - float(g @ s) - 0.5 * sy) / max(1., abs(f))
                    cosines.append(cosine if isfinite(cosine) else -1.)
                    residuals.append(residual if isfinite(residual) else np.inf)

            alphas.append(alpha)
            prev_g, prev_d, prev_norm = g, direction, gnorm
            x, f, g = x_new, f_new, g_new
            gnorm = float(np.linalg.norm(g))
            grad_inf = float(np.max(np.abs(g)))
            k += 1
            stage_steps[stage] += 1
            fallbacks += status == "armijo_fallback"
            trace_evals.append(calls)
            trace_grad.append(grad_inf)
            trace_stage.append(stage)
            trace_l2.append(gnorm)
            trace_f.append(f)

    result = dict(method=method, converged=(reason == "converged"), reason=reason,
                  evaluations=calls, iterations=k, initial_f=_json_number(initial_f),
                  final_f=_json_number(f), initial_grad_inf=_json_number(initial_grad_inf),
                  final_grad_inf=_json_number(grad_inf), initial_grad_l2=_json_number(initial_norm),
                  final_grad_l2=_json_number(gnorm),
                  phase_iterations=dict(zip(STAGE_NAMES, stage_steps)),
                  phase_evaluations=dict(zip(STAGE_NAMES, stage_evals)),
                  switches=switches, fallbacks=fallbacks, restarts=restarts)
    trace = dict(evals=np.asarray(trace_evals, dtype=np.int32),
                 grad_inf=np.asarray(trace_grad, dtype=float),
                 grad_l2=np.asarray(trace_l2, dtype=float),
                 stage=np.asarray(trace_stage, dtype=np.int8),
                 f=np.asarray(trace_f, dtype=float))
    return result, trace

