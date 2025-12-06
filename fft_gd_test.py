# ftt_gd_test.py
import numpy as np
import sympy as sp
from function_utils import gen_random_example, XS
from fft_gd import fft_fit

# Generate a target function
RNG = np.random.default_rng(123)
f_expr = gen_random_example(RNG, complexity=2)
f_sampled = np.array([f_expr.subs(sp.symbols('x'), xi) for xi in XS], dtype=np.float64)
print('Sampled function: ', f_expr)

# Fit best function
best_expr, best_params, best_loss = fft_fit(f_sampled)
print("Best expression:", best_expr)
print("Best parameters:", best_params)