import function_utils
import numpy as np
import sympy as sp
import pickle

x = sp.symbols('x')

def main():
    rng = np.random.default_rng(229)
    n_examples = 4000
    samples = function_utils.SAMPLES

    X = np.zeros((n_examples, samples))
    SE = [None] * n_examples
    STR = [None] * n_examples
    y = np.zeros((n_examples,))

    for i in range(n_examples):
        complexity = i // (n_examples // 4)
        expr = function_utils.gen_random_example(rng, complexity)
        X[i] = function_utils.gen_values(expr)
        SE[i] = expr
        STR[i] = function_utils.get_string(expr)
        y[i] = complexity

    # Save numeric arrays as CSV
    np.savetxt("X_comp_1_3.csv", X, delimiter=",", fmt="%.10f")
    np.savetxt("y_comp_1_3.csv", y, delimiter=",", fmt="%.10f")
    
    # Pickle SymPy expressions and strings
    with open("SE_comp_1_3.pkl", "wb") as f:
        pickle.dump(SE, f)
    
    with open("STR_comp_1_3.pkl", "wb") as f:
        pickle.dump(STR, f)

    print("Data saved successfully.")

if __name__ == "__main__":
    main()
