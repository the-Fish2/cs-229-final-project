import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.utils.function_utils import gen_all_complexity, XS, validate, plot_example
from src.gd_fit_affine.init_affine import init_affine_values, RNG

# ------------------------------------------------------------
# Generate all complexity-2 expressions
# ------------------------------------------------------------
complexity = 2
all_exprs = gen_all_complexity(complexity)

print(f"Generated {len(all_exprs)} complexity-{complexity} expressions")

# ------------------------------------------------------------
# Initialize each expression numerically on XS
# ------------------------------------------------------------
initialized = []
for expr, m in all_exprs:
    res = init_affine_values(expr, XS, RNG)
    if res is not None:
        expr_filled, param_dict = res
        if validate(expr_filled):
            initialized.append((expr_filled, param_dict))

print(f"{len(initialized)} expressions successfully initialized and valid")

# ------------------------------------------------------------
# Optional: plot the first few
# ------------------------------------------------------------
for i, (expr_filled, params) in enumerate(initialized[:5]):
    print(f"Expression {i}: {expr_filled}")
    plot_example(expr_filled)
