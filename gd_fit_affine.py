# gd_fit_affine.py

import numpy as np
import sympy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from function_utils import gen_all_complexity, XS, x, gen_values, validate, get_string
from init_affine import init_affine_values

# -----------------------------
# Hyperparameters
# -----------------------------
GD_MAX_COMPLEXITY = 1
GD_COEFF_INIT_TRIES = 10
GD_LEARNING_RATE = 0.01
GD_STEPS = 150

RNG = np.random.default_rng(42)

# -----------------------------
# Convert sympy expression with a_i/b_i to TensorFlow variables
# -----------------------------
import tensorflow as tf
import sympy as sp
from function_utils import x  # make sure this is the same x symbol used in your expressions
import tensorflow as tf
import sympy as sp

def sympy_to_tf(expr, param_dict):
    """
    Given a sympy expression with a_i/b_i symbols and x,
    create TF Variables for a_i/b_i and return a TF function f_tf(x_input) 
    that can be evaluated on XS and used with GradientTape.
    """
    # Sort symbols for consistent ordering
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    
    # Create TF variables for a_i/b_i (x is not a variable)
    tf_vars = {}
    for s in symbols:
        if s != x:  # x is input
            tf_vars[s] = tf.Variable(float(param_dict[s]), dtype=tf.float32)

    # Lambdify using TensorFlow so gradients are tracked
    f_tf_lambdified = sp.lambdify([x] + [s for s in symbols if s != x], expr, modules='tensorflow')

    # Wrap into a function that takes a TF tensor x_input
    def f_tf(x_input):
        args = [x_input] + [tf_vars[s] for s in symbols if s != x]
        return f_tf_lambdified(*args)

    return f_tf, tf_vars


# -----------------------------
# Loss function: sum squared error
# -----------------------------
# def loss_fn(pred, target):
#     sq_diff = (pred - target) ** 2
#     sq_diff_clipped = tf.clip_by_value(sq_diff, 0, 100)  # clip squared differences
#     return tf.reduce_sum(sq_diff_clipped)
def loss_fn(pred, target, delta=1.0): 
    err = pred - target 
    abs_err = tf.abs(err) 
    quad = tf.minimum(abs_err, delta) 
    lin = abs_err - quad 
    return tf.reduce_sum(0.5 * tf.square(quad) + delta * lin)

# -----------------------------
# Fit a single expression to f_sampled
# -----------------------------
def fit_expr(expr_template, f_sampled):
    if 'exp' not in get_string(expr_template): return (0,0,1000000000000)
    print('fitting template', expr_template)
    best_loss = np.inf
    best_params = None

    for _ in range(GD_COEFF_INIT_TRIES):
        res = init_affine_values(expr_template, XS, RNG)
        if res is None:
            continue
        expr_filled, param_dict = res

        # Convert to TF function
        f_tf, tf_vars = sympy_to_tf(expr_template, param_dict)

        opt = tf.keras.optimizers.Adam(GD_LEARNING_RATE)
        XS_tf = tf.convert_to_tensor(XS, dtype=tf.float32)
        f_sampled_tf = tf.convert_to_tensor(f_sampled, dtype=tf.float32)

        # Track loss during gradient descent
        loss_history = []

        # Gradient descent loop
        for _ in range(GD_STEPS):
            print(tf_vars)
            with tf.GradientTape() as tape:
                preds = f_tf(XS_tf)
                preds_tf = tf.convert_to_tensor(preds, dtype=tf.float32)
                loss = loss_fn(preds_tf, f_sampled_tf)
            loss_history.append(loss.numpy())
            grads = tape.gradient(loss, list(tf_vars.values()))
            opt.apply_gradients(zip(grads, list(tf_vars.values())))

        # Plot loss curve for this initialization
        plt.figure(figsize=(5,3))
        plt.plot(loss_history)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Loss Curve")
        plt.grid(True)
        plt.show()
        print('gd to', expr_template.subs({s: tf_vars[s].numpy() for s in tf_vars}))

        # Evaluate final loss
        preds_final = f_tf(XS)
        final_loss = np.sum((preds_final - f_sampled) ** 2)

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = {s: tf_vars[s].numpy() for s in tf_vars}

    if best_params is None:
        return None, None, np.inf

    best_expr_filled = expr_template.subs(best_params)
    print('best found:', best_expr_filled)
    return best_expr_filled, best_params, best_loss

# -----------------------------
# Main fitting function: loop over all complexities
# -----------------------------
def fit_best_function(f_sampled):
    best_overall_loss = np.inf
    best_overall_expr = None
    best_overall_params = None

    for c in range(GD_MAX_COMPLEXITY + 1):
        templates = gen_all_complexity(c)
        print(f"Complexity {c}, {len(templates)} templates")

        for expr_template, m in templates:
            expr_filled, params, loss = fit_expr(expr_template, f_sampled)
            if loss < best_overall_loss:
                best_overall_loss = loss
                best_overall_expr = expr_filled
                best_overall_params = params

    print(f"Best overall loss: {best_overall_loss:.4f}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(XS, f_sampled, label="Target f(x)")
    plt.plot(XS, gen_values(best_overall_expr), label=f"Best fit, loss={best_overall_loss:.4f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Gradient Descent Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_overall_expr, best_overall_params, best_overall_loss
