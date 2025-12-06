# gd_fit_affine.py

import numpy as np
import sympy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from function_utils import gen_all_complexity, XS, x, gen_values, validate, get_string
from init_affine import init_affine_values

def fft_loss_fn(pred, target, delta=1.0):
    # Step 1: Perform FFT on the predicted and target signals
    pred_fft = tf.signal.fft(tf.cast(pred, tf.complex64))  # FFT on predicted signal
    target_fft = tf.signal.fft(tf.cast(target, tf.complex64))  # FFT on target signal
    
    # Step 2: Extract real and imaginary parts of the Fourier coefficients
    pred_real = tf.math.real(pred_fft)  # Real part of the FFT of predicted signal
    target_real = tf.math.real(target_fft)  # Real part of the FFT of target signal
    pred_imag = tf.math.imag(pred_fft)  # Imaginary part of the FFT of predicted signal
    target_imag = tf.math.imag(target_fft)  # Imaginary part of the FFT of target signal
    
    # Step 3: Compute the error between real and imaginary parts
    print(pred_real, target_real, pred_imag, target_imag)
    print("wahoo")
    real_err = pred_real - target_real
    imag_err = pred_imag - target_imag
    
    # Step 4: Compute absolute errors
    abs_real_err = tf.abs(real_err)
    abs_imag_err = tf.abs(imag_err)
    
    # Step 5: Apply smooth L1 loss (Huber loss) for both real and imaginary parts
    real_quad = tf.minimum(abs_real_err, delta)  # Quadratic part for real
    imag_quad = tf.minimum(abs_imag_err, delta)  # Quadratic part for imaginary
    
    real_lin = abs_real_err - real_quad  # Linear part for real
    imag_lin = abs_imag_err - imag_quad  # Linear part for imaginary
    
    # Step 6: Total loss as sum of the losses for real and imaginary parts
    loss = tf.reduce_sum(0.5 * tf.square(real_quad) + delta * real_lin) + \
           tf.reduce_sum(0.5 * tf.square(imag_quad) + delta * imag_lin)

    return loss


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

def fit_expr(expr_template, f_sampled):

    # if 'exp' not in get_string(expr_template):
    #     return (0, 0, 1e12)

    print("Fitting template:", expr_template)

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

    XS_tf = tf.convert_to_tensor(XS, dtype=tf.float32)  # Input tensor (still float32)
    f_sampled_tf = tf.convert_to_tensor(f_sampled, dtype=tf.float32)  # Target tensor (float32)

    # Ensure these tensors are treated as complex64 in the loss calculation
    XS_tf_complex = tf.cast(XS_tf, tf.complex64)  # Cast to complex64
    f_sampled_tf_complex = tf.cast(f_sampled_tf, tf.complex64)  # Cast to complex64

    # Continue with the rest of your function...
    def run_GD(param_dict, steps):
        f_tf, tf_vars = sympy_to_tf(expr_template, param_dict)
        opt = tf.keras.optimizers.Adam(GD_LEARNING_RATE)

        for _ in range(steps):
            with tf.GradientTape() as tape:
                preds = f_tf(XS_tf_complex)  # Pass complex-valued input
                # Use the FFT-based loss function that considers both real and imaginary parts
                print('wooo')
                loss = fft_loss_fn(preds, f_sampled_tf_complex)  # Use complex tensors for loss calculation

            grads = tape.gradient(loss, list(tf_vars.values()))
            opt.apply_gradients(zip(grads, list(tf_vars.values())))

        # Final loss (numpy)
        preds_final = f_tf(XS_tf).numpy()
        final_loss = np.sum((preds_final - f_sampled)**2)

        final_params = {s: tf_vars[s].numpy() for s in tf_vars}
        return final_loss, final_params

    # ---------------------------------------------------
    # ROUND 0 → Random initialization of all candidates
    # ---------------------------------------------------
    candidates = []

    for _ in range(INIT_CANDIDATES):
        res = init_affine_values(expr_template, XS, RNG)
        if res is None:
            continue
        _, param_dict = res

        loss, params = run_GD(param_dict, STEPS_INITIAL)
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

            loss, fitted_params = run_GD(noisy_params, STEPS_REFINE)
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