import function_utils
import numpy as np

def main():
    rng = np.random.default_rng(229)
    for i in range(100):
        expr = function_utils.gen_random_example(rng, 3)
        function_utils.plot_example(expr)  # each plot waits until closed

if __name__ == "__main__":
    main()
