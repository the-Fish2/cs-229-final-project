# gd_fit_affine.py

import plotly.graph_objects as go
import os

import numpy as np
import sympy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.function_utils import gen_all_complexity, XS, x, gen_values, validate, get_string
from gd_fit.init_affine import init_affine_values

# -----------------------------
# Hyperparameters
# -----------------------------
GD_MAX_COMPLEXITY = 1
GD_COEFF_INIT_TRIES = 10
GD_LEARNING_RATE = 0.01
GD_STEPS = 500

RNG = np.random.default_rng(42)

# -----------------------------
# Convert sympy expression with a_i/b_i to TensorFlow variables
# -----------------------------

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

import plotly.graph_objects as go
import os
def save_convergence_frames(XS, true_vals, all_preds, ind, ref_step, outdir):
    """
    Create one image per GD step.
    - true_vals: green curve
    - all_preds[k]: prediction from step k
    - earlier curves have lower opacity
    - x-range taken from XS
    - y-range taken from true_vals only
    """
    import plotly.graph_objects as go
    import os
    os.makedirs(outdir, exist_ok=True)

    num_steps = len(all_preds)

    fig = go.Figure()

    # Green curve (true expression)
    fig.add_trace(go.Scatter(
        x=XS,
        y=true_vals,
        mode='lines',
        line=dict(color='green'),
        name='Target function'
    ))

    # Blue curves: add every 5th prediction step
    for i in range(0, num_steps, 5):
        opacity = (i + 1) / num_steps
        fig.add_trace(go.Scatter(
            x=XS,
            y=all_preds[i],
            mode='lines',
            line=dict(color=f'rgba(0,0,255,{opacity})'),
            showlegend=False
        ))

    # Auto x-range from XS
    x_min, x_max = np.min(XS), np.max(XS)
    # Auto y-range from true_vals with small padding
    y_min, y_max = np.min(true_vals), np.max(true_vals)
    y_pad = (y_max - y_min) * 0.05  # 5% padding
    y_min -= y_pad
    y_max += y_pad

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        title=f"Refinement Step {ref_step}",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=800,
        height=500
    )

    fig.write_image(f"{outdir}/sample_{ind:04d}.png", scale=3)

def fit_expr(expr_template, f_sampled, img_path_base, hints=[]):


    for hint in hints:
        if hint not in get_string(expr_template):
            return (0, 0, 1e12)

    print("Fitting template:", expr_template)
    img_path = img_path_base + '_' + get_string(expr_template)
    os.makedirs(img_path, exist_ok=True)

    # ---------------------------------------------------
    # Hyperparameters for multi-stage refinement
    # ---------------------------------------------------
    ROUNDS = 4                  # number of refinement stages
    INIT_CANDIDATES = 12        # how many starting points
    KEEP_TOP = 3                # survivors per round
    STEPS_INITIAL = 120         # GD steps for round 0
    STEPS_REFINE = 80           # GD steps for rounds 1+
    NOISE_SCALE = 0.35          # relative noise added to params
    NOISE_DECAY = 0.5           # how much noise shrinks each round

    XS_tf = tf.convert_to_tensor(XS, dtype=tf.float32)
    f_sampled_tf = tf.convert_to_tensor(f_sampled, dtype=tf.float32)

    # ---------------------------------------------------
    # Helper: run GD for a fixed number of steps
    # ---------------------------------------------------
    img_ind = 0
    def run_GD(param_dict, steps, img_ind, ref_step, img_path):

        f_tf, tf_vars = sympy_to_tf(expr_template, param_dict)
        opt = tf.keras.optimizers.Adam(GD_LEARNING_RATE)

        # LOOP
        all_preds = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                preds = f_tf(XS_tf)
                loss = loss_fn(preds, f_sampled_tf)

            grads = tape.gradient(loss, list(tf_vars.values()))
            opt.apply_gradients(zip(grads, list(tf_vars.values())))

            preds = f_tf(XS_tf).numpy()
            all_preds.append(f_tf(XS_tf).numpy())

        save_convergence_frames(XS, f_sampled, all_preds, img_ind, ref_step, outdir=img_path)
        img_ind += 1

        # Final loss (numpy)
        preds_final = f_tf(XS_tf).numpy()
        final_loss = np.sum((preds_final - f_sampled)**2)

        final_params = {s: tf_vars[s].numpy() for s in tf_vars}
        return final_loss, final_params

    # ---------------------------------------------------
    # ROUND 0 → Random initialization of all candidates
    # ---------------------------------------------------
    candidates = []
    ref_step = 0
    for _ in range(INIT_CANDIDATES):
        res = init_affine_values(expr_template, XS, RNG)
        if res is None:
            continue
        _, param_dict = res

        loss, params = run_GD(param_dict, STEPS_INITIAL, img_ind=img_ind, ref_step=ref_step, img_path=img_path)
        img_ind += 1
        candidates.append((loss, params))

    if not candidates:
        return None, None, np.inf

    # Sort best → worst
    candidates.sort(key=lambda t: t[0])
    candidates = candidates[:KEEP_TOP]

    # ---------------------------------------------------
    # REFINEMENT ROUNDS
    # ---------------------------------------------------
    noise = NOISE_SCALE

    for round_idx in range(1, ROUNDS + 1):
        ref_step += 1
        print(f"Refinement round {round_idx}: best loss so far = {candidates[0][0]:.6f}")

        new_candidates = []

        # take top survivors
        for loss, params in candidates:
            new_candidates.append((loss, params))

        # create new noisy candidates around the best one
        best_loss, best_params = candidates[0]

        for _ in range(INIT_CANDIDATES - KEEP_TOP):

            # produce a noisy copy
            noisy_params = {}
            for k,v in best_params.items():
                noisy_params[k] = float(v + noise * v * RNG.normal())

            loss, fitted_params = run_GD(noisy_params, STEPS_REFINE, img_ind, ref_step=ref_step, img_path=img_path)
            img_ind += 1
            new_candidates.append((loss, fitted_params))

        # sort survivors
        new_candidates.sort(key=lambda t: t[0])
        candidates = new_candidates[:KEEP_TOP]

        # shrink noise
        noise *= NOISE_DECAY

    # ---------------------------------------------------
    # Best after all refinement rounds
    # ---------------------------------------------------
    best_loss, best_params = candidates[0]
    best_expr = expr_template.subs(best_params)

    print("Best found:", best_expr)
    return best_expr, best_params, best_loss

# -----------------------------
# Main fitting function: loop over all complexities
# -----------------------------
def fit_best_function(f_sampled, img_path_base, search_complexity=GD_MAX_COMPLEXITY, hints=[]):
    best_overall_loss = np.inf
    best_overall_expr = None
    best_overall_params = None

    for c in range(search_complexity + 1):
        templates = gen_all_complexity(c)
        print(f"Complexity {c}, {len(templates)} templates")

        for expr_template, m in templates:
            expr_filled, params, loss = fit_expr(expr_template, f_sampled, img_path_base, hints)
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
