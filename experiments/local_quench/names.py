"""Stable archival identifiers and descriptive names used in new reports.

Numeric function IDs remain part of the SeedSequence and must not be reordered.
M identifiers remain compatible with the measured archival CSVs.
"""

METHOD_LABELS = {
    "M1": "standard L-BFGS",
    "M2": "NCG--L-BFGSM",
    "M3": "SD--L-BFGSM",
    "M4": "SD--NCG",
    "M5": "SD--NCG--L-BFGSM",
    "M6": "SD--standard L-BFGS",
}
METHOD_ALIASES = dict(zip(
    ("lbfgs", "ncg-lbfgsm", "sd-lbfgsm", "sd-ncg", "sd-ncg-lbfgsm", "sd-lbfgs"),
    METHOD_LABELS,
))
FUNCTION_LABELS = {
    1: "Styblinski--Tang sum",
    2: "Two-coordinate three-hump camel blocks",
    3: "Two-coordinate Bohachevsky-2 blocks",
    4: "Eight-coordinate Griewank blocks",
    5: "Mixed/scaled quadratic--cosine variant",
    6: "Mixed/scaled Rastrigin",
    7: "Two-coordinate Bohachevsky-3 blocks",
    8: "Corrugated open Rosenbrock chain",
    9: "Eight-coordinate corrugated Rosenbrock blocks",
    10: "Heterogeneous periodic double-well chain",
}
METHOD_NAMES = METHOD_LABELS


def method_id(value):
    """Accept a descriptive CLI name or an archival method identifier."""
    result = METHOD_ALIASES.get(value.lower(), value.upper())
    if result not in METHOD_LABELS:
        raise ValueError(f"Unknown method {value!r}; choose {', '.join(METHOD_ALIASES)}")
    return result
