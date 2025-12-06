# Utilities for generating and using data for functions

# Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import re
import pickle

SIMPLIFICATION_THRESHOLD = 6

# Domain under consideration
DOMAIN_LOW = -10
DOMAIN_HIGH = 10
SAMPLES = 5000

# Coefficients to scale / add functions when generating
# Picked from uniform log distribution
COEFF_LOW = 0.1
COEFF_HIGH = 10
# Picked with normal distribution
BIAS_VAR = 1
XS = np.linspace(DOMAIN_LOW, DOMAIN_HIGH, SAMPLES)

# def softplus(x):
#     print(x)
#     return np.log(1 + np.exp(x))

# Functions under consideration
x = sp.symbols('x')
y = sp.symbols('y')
# We hope that functions raised to a constant (x**3) will be processable by taking log
UNARY = [sp.exp(x), sp.log(x), sp.sin(x), sp.cos(x), sp.tan(x)]
BINARY = [x + y, x - y, x * y, x / y, x ** y]

def gen_coeff(rng: np.random.Generator) -> np.float64:
    return rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(COEFF_LOW), np.log(COEFF_HIGH)))

def gen_bias(rng: np.random.Generator) -> np.float64:
    return rng.normal(0, BIAS_VAR)

def random_affine(rng: np.random.Generator, expr: sp.Expr):
    return gen_coeff(rng) * expr + gen_bias(rng)

import numpy as np
import sympy as sp

def gen_values(expr: sp.Expr, XS) -> np.ndarray:
    """
    Evaluate a SymPy expression over an array XS safely.
    Replaces invalid, infinite, or complex values with np.nan.
    """
    # print("THIS EXPR IS " , expr)

    # if 'log' in get_string(expr):
    #     DOMAIN_LOW = 0
    #     XS = np.linspace(DOMAIN_LOW, DOMAIN_HIGH, SAMPLES//2)
    # else:
    #     XS = np.linspace(DOMAIN_LOW, DOMAIN_HIGH, SAMPLES//2)
    print("hello?")
    # Convert symbolic expression to numeric function
    f = sp.lambdify(sp.symbols('x'), expr, modules=['numpy'])
    # Evaluate safely
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ys = f(XS)
        ys = np.broadcast_to(ys, XS.shape)
        ys = np.array(ys, dtype=np.complex128)  # allow complex temporarily

        # Replace NaN, inf, and complex numbers with np.nan
        ys[~np.isfinite(ys)] = np.nan
        ys[np.iscomplex(ys)] = np.nan
        ys = ys.real  # convert back to float
    
    print("YS", ys)
    return ys

def validate(expr: sp.Expr) -> bool:
    return not np.any(np.isnan(gen_values(expr, XS)))

# Issue: Constants are identified as zero complexity, same as linear functions; maybe should be lower

# Generate a single example in SymPy; parameter will always be x
def gen_random_example(rng: np.random.Generator, complexity: int) -> sp.Expr:
    # Assign zero complexity to linear functions
    if complexity == 0:
        return random_affine(rng, x)
    
    args = rng.choice(np.arange(1, 3))
    if args == 1:
        # Apply a unary function on a single function of one less complexity
        while True:
            func_a = gen_random_example(rng, complexity - 1)
            ret = random_affine(rng, rng.choice(UNARY).subs(x, func_a))
            # To prevent getting a lower complexity function, only return if the complexity is actually greater
            if len(get_string(ret)) + SIMPLIFICATION_THRESHOLD >= len(get_string(func_a)) and validate(ret): return ret
    else:
        # Apply a binary function on two functions whose complexities sum to one less
        a_complexity = rng.choice(np.arange(0, complexity))
        while True:
            func_a, func_b = gen_random_example(rng, a_complexity), gen_random_example(rng, complexity - 1 - a_complexity)
            ret = random_affine(rng, rng.choice(BINARY).subs(x, func_a).subs(y, func_b))
            if len(get_string(ret)) + SIMPLIFICATION_THRESHOLD >= len(get_string(func_a)) + len(get_string(func_b)) and validate(ret): return ret
    
# ------------------------------------------------------------
# New utilities for generating all expressions of given complexity
# ------------------------------------------------------------

# Create indexed affine symbols
def affine_symbols(i: int):
    return sp.symbols(f"a{i} b{i}")

def apply_affine(expr: sp.Expr, i: int):
    ai, bi = affine_symbols(i)
    return ai * expr + bi

# Main generator
def gen_all_complexity(complexity: int, next_id: int = 0):
    """
    Generate ALL expressions of the given complexity.
    Returns list of (expr, m) where:
        expr = sympy expression
        m    = # of affine transforms used (so symbols are x and a0..a(m-1), b0..b(m-1))
    next_id = starting index for affine parameters
    """
    results = []

    # -------------------------------------------
    # Base case: complexity = 0 → a0*x + b0
    # -------------------------------------------
    if complexity == 0:
        expr = apply_affine(x, next_id)
        return [(expr, 1)]  # one affine used

    # -------------------------------------------
    # Unary case: apply each unary function to child
    # -------------------------------------------
    # child complexity is complexity - 1
    children = gen_all_complexity(complexity - 1, next_id)

    for func in UNARY:
        for child_expr, child_m in children:
            # Substitute x inside the unary template
            new_core = func.subs(x, child_expr)
            # Apply a new affine layer using index next_id + child_m
            expr = apply_affine(new_core, next_id + child_m)
            results.append((expr, child_m + 1))

    # -------------------------------------------
    # Binary case: split complexity-1 into two parts
    # -------------------------------------------
    total_child_complexity = complexity - 1

    for cA in range(total_child_complexity + 1):
        cB = total_child_complexity - cA

        childrenA = gen_all_complexity(cA, next_id)
        for (exprA, mA) in childrenA:
            # childrenB must start after A's affine symbols
            childrenB = gen_all_complexity(cB, next_id + mA)
            for (exprB, mB) in childrenB:

                for func in BINARY:
                    # Build binary template: func(x, y)
                    new_core = func.subs({x: exprA, y: exprB})
                    # Apply a new affine layer after all child affines
                    expr = apply_affine(new_core, next_id + mA + mB)
                    results.append((expr, mA + mB + 1))

    return results

# Process any decimals that didn't get truncated
def repl(match):
    return f"{float(match.group()):.3f}"

def get_string(expr: sp.Expr) -> str:
    # Generated with GPT 
    """
    Replace all numeric constants in expr with rounded versions
    having exactly 3 decimals, and return as a string.
    """
    # Replace every Float with a rounded version
    rounded_expr = expr.xreplace({
        n: sp.Float(round(float(n), 3)) for n in expr.atoms(sp.Float)
    })

    ret = str(rounded_expr)
    ret = re.sub(r"\d+\.\d+", repl, ret)
    return str(ret)


def plot_example(expr: sp.Expr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ys = gen_values(expr, XS)

    # Clip extreme values to avoid huge spikes
    ys = np.where(np.abs(ys) > 10, np.nan, ys)
    
    # Convert expression to string with 3 decimals
    expr_str = get_string(expr)
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(XS, ys, label=expr_str)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Random SymPy Expression')
    plt.grid(True)
    plt.legend()
    plt.show()

def load_sympy_to_np_array(path: str) -> np.ndarray:
    """
    Load a pickle file containing a list of SymPy expressions
    and return a NumPy array where each row is the evaluation
    of one expression over XS using gen_values.
    """
    # Load the pickled expressions
    with open(path, "rb") as f:
        expr_list = pickle.load(f)  # expected: list of sp.Expr
    
    # Evaluate each expression and stack into a 2D array
    data_array = np.array([gen_values(expr, XS) for expr in expr_list])
    
    return data_array
