import function_utils
import numpy as np
import sympy as sp

x = sp.symbols('x')
def main():
    rng = np.random.default_rng(229)
    # print(function_utils.gen_values(1 / x))

    # Generate data
    n_examples = 300
    samples = function_utils.SAMPLES

    X = np.zeros((n_examples, samples))
    y = np.zeros((n_examples,))

    for i in range(n_examples):
        # print('i:', i)
        complexity = i // (n_examples // 3) + 1
        expr = function_utils.gen_random_example(rng, complexity)
        # print(function_utils.get_string(expr))
        # function_utils.plot_example(expr)  # each plot waits until closed
        X[i] = function_utils.gen_values(expr)
        # print(X[i])
        # print(np.any(np.isnan(X[i])))
        y[i] = complexity

    np.savetxt("X_comp_1_3.csv", X, delimiter=",", fmt="%.10f")
    np.savetxt("y_comp_1_3.csv", y, delimiter=",", fmt="%.10f")
    



if __name__ == "__main__":
    main()
