# main.py
import numpy as np
import sympy as sp
from function_utils import (
    gen_all_complexity, XS, x, gen_values, validate, plot_example
)

EPS = 1e-6
RNG = np.random.default_rng(42)

# ------------------------------------------------------------
# Helper: recursively initialize affine parameters
# ------------------------------------------------------------
def init_affine_values(expr, XS, rng):
    """
    Initialize all a_i, b_i symbols in expr recursively from inside to outside.
    Returns: expr_filled or None if impossible.
    """
    # Identify all a_i, b_i symbols in increasing order
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    param_dict = {}

    for sym in symbols:
        # Random initialization
        if sym.name.startswith("a"):
            param_dict[sym] = rng.uniform(0.1, 2.0)
        elif sym.name.startswith("b"):
            param_dict[sym] = rng.normal(0, 1.0)

    filled = expr.subs(param_dict)

    # Check power bases > EPS
    ys = gen_values(filled)
    if np.any(ys < -1e10) or np.any(np.isnan(ys)):
        return None

    return filled, param_dict
