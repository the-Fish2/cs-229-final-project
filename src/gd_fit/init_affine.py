# main.py
import numpy as np
import sympy as sp
from utils.function_utils import (
    gen_all_complexity, XS, x, gen_values, validate, plot_example, y
)

EPS = 1e-6
RNG = np.random.default_rng(43)



UNARY = [sp.exp(x), sp.log(x), sp.sin(x), sp.cos(x), sp.tan(x)]
BINARY = [x + y, x - y, x * y, x / y, x ** y]

MAX_ABS = 1e10  # simple overflow guard

def validate_basic(expr, gen_values, eps=1e-8):
    # Final expression must be finite and not huge
    try:
        ys = gen_values(expr)
    except Exception:
        return False
    if np.any(~np.isfinite(ys)):
        return False
    if np.any(np.abs(ys) > MAX_ABS):
        return False

    # Denominator must not vanish (catch implicit divisions too)
    num, den = sp.together(expr).as_numer_denom()
    try:
        den_vals = gen_values(den)
    except Exception:
        return False
    if np.any(np.abs(den_vals) <= eps):
        return False

    # Function-specific domain checks
    for node in sp.preorder_traversal(expr):
        # log: argument strictly positive
        if node.func == sp.log:
            try:
                arg_vals = gen_values(node.args[0])
            except Exception:
                return False
            if np.any(arg_vals <= eps):
                return False

        # tan: avoid poles where cos(arg) ~ 0
        elif node.func == sp.tan:
            pass
            # try:
            #     cos_vals = gen_values(sp.cos(node.args[0]))
            # except Exception:
            #     return False
            # if np.any(np.abs(cos_vals) <= eps):
            #     return False

        # Pow: common real-domain issues
        elif isinstance(node, sp.Pow):
            base, exp = node.as_base_exp()
            try:
                base_vals = gen_values(base)
                exp_vals = gen_values(exp)
            except Exception:
                return False

            # 0 ** negative exponent -> division by zero
            zero_base = np.isclose(base_vals, 0.0, atol=eps)
            neg_exp = exp_vals < -eps
            if np.any(zero_base & neg_exp):
                return False

            # negative base with non-integer exponent -> complex
            neg_base = base_vals < -eps
            exp_is_integer = np.isclose(exp_vals, np.round(exp_vals), atol=1e-8)
            if np.any(neg_base & ~exp_is_integer):
                return False

    return True

MAX_ABS = 1e10
MAX_EXP_ARG = 700.0

def _is_affine_param(sym):
    return isinstance(sym, sp.Symbol) and (sym.name.startswith("a") or sym.name.startswith("b"))

def validate_node(node, gen_values, eps=1e-8):
    try:
        vals = np.asarray(gen_values(node))
    except Exception:
        return False
    if np.iscomplexobj(vals) or np.any(~np.isfinite(vals)):
        return False
    if np.any(np.abs(vals) > MAX_ABS):
        return False

    num, den = sp.together(node).as_numer_denom()
    try:
        den_vals = np.asarray(gen_values(den))
    except Exception:
        return False
    if np.any(np.abs(den_vals) <= eps):
        return False

    if node.func == sp.log:
        arg = node.args[0]
        try:
            arg_vals = np.asarray(gen_values(arg))
        except Exception:
            return False
        if np.any(arg_vals <= eps):
            return False

    elif node.func == sp.tan:
        arg = node.args[0]
        try:
            cos_vals = np.asarray(gen_values(sp.cos(arg)))
        except Exception:
            return False
        if np.any(np.abs(cos_vals) <= eps):
            return False

    elif node.func == sp.exp:
        arg = node.args[0]
        try:
            arg_vals = np.asarray(gen_values(arg))
        except Exception:
            return False
        if np.any(arg_vals > MAX_EXP_ARG):
            return False

    elif isinstance(node, sp.Pow):
        base, exp = node.as_base_exp()
        try:
            base_vals = np.asarray(gen_values(base))
            exp_vals = np.asarray(gen_values(exp))
        except Exception:
            return False
        zero_base = np.isclose(base_vals, 0.0, atol=eps)
        neg_exp = exp_vals < -eps
        if np.any(zero_base & neg_exp):
            return False
        neg_base = base_vals < -eps
        exp_is_integer = np.isclose(exp_vals, np.round(exp_vals), atol=1e-8)
        if np.any(neg_base & ~exp_is_integer):
            return False

    return True

def _log_affine_trial(arg, param_dict, rng, XS, margin=0.5):
    """
    Construct a trial for the affine inside log: a*x + b >= margin on XS.
    Picks positive slope a, then b large enough to keep arg > margin over XS.
    """
    # Find one a* and one b* symbol in the arg that aren’t set yet
    a_syms = sorted([s for s in arg.free_symbols if s.name.startswith("a") and s not in param_dict],
                    key=lambda s: s.name)
    b_syms = sorted([s for s in arg.free_symbols if s.name.startswith("b") and s not in param_dict],
                    key=lambda s: s.name)

    trial = {}

    # Decide slope a (prefer small positive to avoid wild swings)
    if a_syms:
        a_val = rng.choice([-1, 1]) * float(rng.uniform(0.2, 10)) ## change???
        trial[a_syms[0]] = a_val
    else:
        # Use existing a if present; otherwise pick a modest positive slope
        a_val = None
        for s in arg.free_symbols:
            if s.name.startswith("a"):
                if s in param_dict:
                    a_val = float(param_dict[s])
                break
        if a_val is None:
            a_val = float(rng.uniform(0.2, 10)) ## change??

    xmin = float(np.min(XS))
    xmax = float(np.max(XS))
    # Ensure min_x of a*x + b is >= margin
    if a_val >= 0:
        required_b = -a_val * xmin + margin
    else:
        required_b = -a_val * xmax + margin

    if b_syms:
        trial[b_syms[0]] = float(rng.uniform(required_b, required_b + 2.0))
    return trial



def init_affine_values(expr, XS, rng, max_restarts=100, local_tries=500):
    
    """
    Initialize all a_i, b_i symbols recursively from inside to outside.
    At each node, after assigning local params, validate that subexpression.
    For log nodes, pick a,b so that the argument is positive on XS.
    If a local assignment fails after local_tries, restart completely.
    Returns: (expr_filled, param_dict) or None if impossible.
    """
    def assign_recursive(node, param_dict):
        # 1) Initialize children first (inside -> outside)
        for arg in getattr(node, "args", ()):
            if arg.is_Atom: continue
            if isinstance(arg, sp.Add) or isinstance(arg, sp.Mul): continue
            if not assign_recursive(arg, param_dict):
                return False

        # 2) Assign this node's a_i/b_i if any are still unset
        local_params = [s for s in node.free_symbols
                        if _is_affine_param(s) and s not in param_dict]
        if local_params:
            for _ in range(local_tries):
                # Default random trial for local params
                trial = {}
                for s in local_params:
                    if s.name.startswith("a"):
                        trial[s] = rng.choice([-1, 1]) * float(rng.uniform(0.1, 10.0))
                    else:  # b*
                        trial[s] = float(rng.normal(0.0, 1.0))

                # Special handling for log: ensure its affine arg is positive on XS
                if node.func == sp.log:
                    log_arg = node.args[0]
                    trial.update(_log_affine_trial(log_arg, {**param_dict, **trial}, rng, XS, margin=max(0.5, 10*EPS)))

                node_sub = node.subs({**param_dict, **trial})
                if validate_node(node_sub, gen_values, eps=EPS):
                    param_dict.update(trial)
                    return True
            return False
        else:
            # 3) Even if no new params here, validate the node with current params
            node_sub = node.subs(param_dict)
            if not validate_node(node_sub, gen_values, eps=EPS):
                return False
            return True

    for _ in range(max_restarts):
        param_dict = {}
        if assign_recursive(expr, param_dict):
            filled = expr.subs(param_dict)
            if validate_node(filled, gen_values, eps=EPS):
                return filled, param_dict
    assert(False)
