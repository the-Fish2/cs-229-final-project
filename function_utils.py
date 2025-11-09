# Utilities for generating and using data for functions

# Imports
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Domain under consideration
DOMAIN_LOW = -1
DOMAIN_HIGH = 1
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
    return np.exp(rng.uniform(np.log(COEFF_LOW), np.log(COEFF_HIGH)))

def gen_bias(rng: np.random.Generator) -> np.float64:
    return rng.normal(0, BIAS_VAR)

def random_affine(rng: np.random.Generator, expr: sp.Expr):
    return gen_coeff(rng) * expr + gen_bias(rng)

# Issue: Constants are identified as zero complexity, same as linear functions; maybe should be lower

# Generate a single example in SymPy; parameter will always be x
def gen_random_example(rng: np.random.Generator, complexity: int) -> sp.Expr:
    # Assign zero complexity to linear functions
    if complexity == 0:
        return gen_coeff(rng) * x
    
    args = rng.choice(np.arange(1, 3))
    if args == 1:
        # Apply a unary function on a single function of one less complexity
        return random_affine(rng, rng.choice(UNARY).subs(x, gen_random_example(rng, complexity - 1)))
    else:
        # Apply a binary function on two functions whose complexities sum to one less
        a_complexity = rng.choice(np.arange(0, complexity))
        func_a, func_b = gen_random_example(rng, a_complexity), gen_random_example(rng, complexity - 1 - a_complexity)
        return random_affine(rng, rng.choice(BINARY).subs(x, func_a).subs(y, func_b))
    
def format_expr_3dec(expr: sp.Expr) -> str:
    # Generated with GPT 
    """
    Replace all numeric constants in expr with rounded versions
    having exactly 3 decimals, and return as a string.
    """
    # Replace every Float with a rounded version
    rounded_expr = expr.xreplace({
        n: sp.Float(round(float(n), 3)) for n in expr.atoms(sp.Float)
    })
    return str(rounded_expr)

def gen_values(expr: sp.Expr) -> np.array:
    # Convert symbolic expression to numeric function
    f = sp.lambdify(x, expr, modules=['numpy'])
    
    # Evaluate expression safely, replacing undefined values with np.nan
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ys = f(XS)
        ys = np.array(ys, dtype=np.float64)
        ys[~np.isfinite(ys)] = np.nan  # set NaN and inf to np.nan
    
    return ys

def plot_example(expr: sp.Expr):
    import matplotlib.pyplot as plt
    import numpy as np
    
    ys = gen_values(expr)
    
    # Clip extreme values to avoid huge spikes
    ys = np.where(np.abs(ys) > 10, np.nan, ys)
    
    # Convert expression to string with 3 decimals
    expr_str = format_expr_3dec(expr)
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(XS, ys, label=expr_str)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Random SymPy Expression')
    plt.grid(True)
    plt.legend()
    plt.show()
