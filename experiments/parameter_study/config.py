"""Frozen numerical design of the 112-run cautious-update study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = (10000, 50000, 100000, 500000)
BUDGETS = dict(zip(DIMENSIONS, (300, 150, 100, 80)))
GRADIENT_TOLERANCE = 1e-8
MEMORY = 8
PROBLEMS = {
    "dixon_price": {"name": "Dixon-Price", "base_seed": 1700},
    "levy": {"name": "Levy", "base_seed": 1375},
    "levy_scaled": {"name": "Scaled Levy", "base_seed": 3100},
    "levy_shifted": {"name": "Shifted Levy", "base_seed": 3200},
    "rastrigin": {"name": "Rastrigin", "base_seed": 1200},
    "schwefel": {"name": "Schwefel", "base_seed": 1300},
    "trig_chain": {"name": "Trigonometric chain", "base_seed": 3500},
}
# Archived identifiers remain the machine-readable keys of the CSV data.
SETTINGS = {
    "lbfgsm_wolfe_c000": (0.0, 0.0, 0.0),
    "lbfgsm_wolfe": (0.9, 1.0, 1.0),
    "lbfgsm_wolfe_c025": (0.25, 1.0, 1.0),
    "lbfgsm_wolfe_c025_c10": (0.25, 1.0, 10.0),
}


def setting_label(method):
    return "L-BFGSM (" + ", ".join(f"{v:g}" for v in SETTINGS[method]) + ")"
