# -*- coding: utf-8 -*-
"""Experiment configuration for the 实验设计 benchmark suite."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DESIGN_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SOLUTIONS_DIR = RESULTS_DIR / "solutions"
RAW_DIR = RESULTS_DIR / "raw"

DIMENSIONS = [10000, 50000, 100000, 500000]

METHODS = [
    "lbfgsm_armijo",
    "lbfgsm_wolfe",
    "ncg_wolfe",
    "sd_ncg",
    "sd_lbfgsm",
    "sd_ncg_lbfgsm",
    "sd_ncg_lbfgsm_strategy5",
    "sd_cg_standard",
    "sd_cg_lbfgs",
    "sd_cg_lbfgs_strategy5",
]

METHOD_LABELS = {
    "lbfgsm_armijo": "LBFGSM + Armijo",
    "lbfgsm_wolfe": "LBFGSM + Wolfe",
    "ncg_wolfe": "NCG (ELS) + Wolfe",
    "sd_ncg": "SD $\\rightarrow$ NCG",
    "sd_lbfgsm": "SD $\\rightarrow$ LBFGSM",
    "sd_ncg_lbfgsm": "SD $\\rightarrow$ NCG $\\rightarrow$ LBFGSM",
    "sd_ncg_lbfgsm_strategy5": "SD $\\rightarrow$ NCG $\\rightarrow$ LBFGSM (S5)",
    "sd_cg_standard": "SD $\\rightarrow$ CG",
    "sd_cg_lbfgs": "SD $\\rightarrow$ CG $\\rightarrow$ L-BFGS",
    "sd_cg_lbfgs_strategy5": "SD $\\rightarrow$ CG $\\rightarrow$ L-BFGS (S5)",
}

# Baselines vs hybrid groups for plots / discussion
BASELINE_METHODS = ["lbfgsm_wolfe", "ncg_wolfe", "lbfgsm_armijo"]
HYBRID_METHODS = [
    "sd_ncg",
    "sd_lbfgsm",
    "sd_ncg_lbfgsm",
    "sd_ncg_lbfgsm_strategy5",
    "sd_cg_standard",
    "sd_cg_lbfgs",
    "sd_cg_lbfgs_strategy5",
]

PLOT_METHODS = [
    "lbfgsm_wolfe",
    "ncg_wolfe",
    "sd_lbfgsm",
    "sd_ncg_lbfgsm",
    "sd_ncg_lbfgsm_strategy5",
]

PROBLEMS = [
    "ackley",
    "rastrigin",
    "schwefel",
    "griewank",
    "levy",
    "discus",
    "ellipsoidal",
    "bent_cigar",
    "dixon_price",
]

PROBLEM_SEED = {
    "ackley": 1100,
    "rastrigin": 1200,
    "schwefel": 1300,
    "griewank": 1350,
    "levy": 1375,
    "discus": 1400,
    "ellipsoidal": 1500,
    "bent_cigar": 1600,
    "dixon_price": 1700,
}

GRAD_TOL = 1e-8

MAX_ITER = {
    10000: 300,
    50000: 150,
    100000: 100,
    500000: 80,
}

LBFGSM_M = 8
LBFGSM_C0 = 0.9
LBFGSM_C1 = 1.0
LBFGSM_C2 = 1.0
