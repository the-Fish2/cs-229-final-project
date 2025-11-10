# Utilities for generating and using data for functions

# Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import re

# Domain under consideration
DOMAIN_LOW = -100
DOMAIN_HIGH = 100
SAMPLES = 400

# Coefficients to scale / add functions when generating
# Picked from uniform log distribution
COEFF_LOW = 0.1
COEFF_HIGH = 10
# Picked with normal distribution
BIAS_VAR = 1
XS = np.linspace(DOMAIN_LOW, DOMAIN_HIGH, SAMPLES)

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

def gen_values(expr: sp.Expr) -> np.ndarray:
    """
    Evaluate a SymPy expression over an array XS safely.
    Replaces invalid, infinite, or complex values with np.nan.
    """
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
    
    return ys

def validate(expr: sp.Expr) -> bool:
    return not np.any(np.isnan(gen_values(expr)))

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
            ret = random_affine(rng, rng.choice(UNARY).subs(x, gen_random_example(rng, complexity - 1)))
            if validate(ret): return ret
    else:
        # Apply a binary function on two functions whose complexities sum to one less
        a_complexity = rng.choice(np.arange(0, complexity))
        while True:
            func_a, func_b = gen_random_example(rng, a_complexity), gen_random_example(rng, complexity - 1 - a_complexity)
            ret = random_affine(rng, rng.choice(BINARY).subs(x, func_a).subs(y, func_b))
            if validate(ret): return ret
    
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
    
    ys = gen_values(expr)

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
