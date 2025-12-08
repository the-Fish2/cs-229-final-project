

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import sympy as sp
import os
import multiprocessing as mp

from utils.function_utils import gen_random_example, XS
from gd_fit import fit_best_function
import tensorflow as tf

def worker(seedseq):
    # Independent RNG per process
    RNG = np.random.default_rng(seedseq)

    # Generate function
    f_expr = gen_random_example(RNG, complexity=1)

    # Sample it
    XS_syms = sp.symbols('x')
    f_sampled = np.array([f_expr.subs(XS_syms, xi) for xi in XS], dtype=np.float64)

    # Fit
    best_expr, best_params, best_loss = fit_best_function(f_sampled)
    return f_expr, best_expr, best_loss

if __name__ == "__main__":
    # Prevent BLAS/NumPy oversubscription (optional but recommended)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Use spawn for safety across platforms
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    results = []
    num_tasks = 100

    # Independent seeds for reproducibility
    seeds = np.random.SeedSequence(12345).spawn(num_tasks)

    # Use all CPUs (or set a specific number)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, s) for s in seeds]
        for future in as_completed(futures):
            f_expr, best_expr, best_loss = future.result()
            print("Sampled function:", f_expr)
            print("Best expression:", best_expr)
            print("Best loss:", best_loss)
            results.append([f_expr, best_expr, best_loss])

    print("DONE")
