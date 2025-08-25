# Usage example:
#   python dvATGJ.py 8.0 5.0 3 1.0
# where:
#   8.0  -> n, number of quadrature points
#   5.0  -> alpha, Jacobi polynomial parameter
#   3.0  -> lambda, scaling parameter
#   1.0  -> T0, temperature parameter
import numpy as np
import argparse
from scipy.linalg import eigh
from scipy.special import gammaln

def jacobi_recurrence(n, alpha, beta):
    a = np.zeros(n)
    b = np.zeros(n)
    for k in range(n):
        if k == 0:
            a[k] = (beta - alpha) / (alpha + beta + 2)
        else:
            a[k] = (beta**2 - alpha**2) / ((2 * k + alpha + beta) * (2 * k + alpha + beta + 2))
        if k > 0:
            num = 4 * k * (k + alpha) * (k + beta) * (k + alpha + beta)
            den = (2 * k + alpha + beta)**2 * (2 * k + alpha + beta + 1) * (2 * k + alpha + beta - 1)
            b[k] = np.sqrt(num / den)
    return a, b

def rootsWeights(n, alpha, lamda, T0):
    if n <= 0 or alpha <= 0 or lamda <= 0:
        raise ValueError("n, alpha, lamda must be > 0.")
    
    beta = 0
    a, b = jacobi_recurrence(n, alpha, beta)
    J = np.diag(a) + np.diag(b[1:], 1) + np.diag(b[1:], -1)
    eigenvalues, eigenvectors = eigh(J)

    roots = 0.5 * (eigenvalues + 1)
    Rr = np.sqrt(lamda * T0 * np.tan(np.pi/2 * roots))
    
    log_ratio = gammaln(alpha + 1) - gammaln(alpha + 2)
    weights = np.exp(log_ratio) * (eigenvectors[0, :]**2)

    return Rr, weights

def save_to_txt(Rr, Ww):
    with open("R_W.txt", "w") as f:
        f.write("# Rr            Ww\n")
        for r, w in zip(Rr, Ww):
            f.write(f"{r:.8e}  {w:.8e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute AT-GJ quadrature nodes and weights")
    parser.add_argument("n", type=int, help="Number of quadrature points")
    parser.add_argument("alpha", type=float, help="Alpha parameter of Jacobi polynomial")
    parser.add_argument("lamda", type=float, help="Scaling parameter")
    parser.add_argument("T0", type=float, help="Temperature parameter")
    args = parser.parse_args()

    Rr, W = rootsWeights(args.n, args.alpha, args.lamda, args.T0)
    save_to_txt(Rr, W)

    print(f"Results written to R_W.txt")