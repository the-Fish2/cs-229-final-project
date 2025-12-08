import numpy as np

def fft_fit(XS, ys, keep_ratio=1):
    N = len(ys)
    L = XS[-1] - XS[0]           # length of full period
    dx = XS[1] - XS[0]           # uniform spacing

    # --- FFT ---
    fft_coeffs = np.fft.rfft(ys)

    # --- Truncate high frequencies ---
    K = int(len(fft_coeffs) * keep_ratio)
    truncated = np.zeros_like(fft_coeffs)
    truncated[:K] = fft_coeffs[:K]

    # --- Construct evaluator ---
    def f_recon(x):
        # Map input x onto periodic [XS[0], XS[-1]]
        x = np.asarray(x)
        x2 = (x - XS[0]) % L + XS[0]

        # fractional index into Fourier domain for each x
        # evaluate via inverse Fourier series formula
        freqs = np.fft.rfftfreq(N, d=dx)  # frequencies k/L
        out = np.zeros_like(x2, dtype=np.complex128)

        for k, coef in enumerate(truncated):
            out += coef * np.exp(2j * np.pi * freqs[k] * (x2 - XS[0]))

        return out.real
    
    return f_recon