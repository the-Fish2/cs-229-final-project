# main.py
import sympy as sp
from function_utils import gen_all_complexity   # <-- change to your actual module name

# Pretty printing options
sp.init_printing()

def main():
    complexity = 3
    print(f"Generating all expressions of complexity = {complexity}...\n")

    exprs = gen_all_complexity(complexity)

    print(f"Total expressions: {len(exprs)}\n")

    for idx, (expr, m) in enumerate(exprs):
        print(f"=== Expression #{idx} ===")
        print(f"Affine layers used (m): {m}")
        print(f"Expression:")
        print(expr)
        print()

    # Optional: plot the first few to visually inspect them
    # from your_module import plot_example
    # for expr, _ in exprs[:5]:
    #     plot_example(expr)

if __name__ == "__main__":
    main()
