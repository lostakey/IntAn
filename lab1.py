import numpy as np
from scipy.optimize import linprog
from itertools import product, combinations


def is_singular(M, R, delta, eps=1e-9):
    m, n = M.shape
    Rd = delta * R
    for s in product([-1, 1], repeat=n):
        s = np.array(s, dtype=float)
        Ms = M * s
        A_ub, b_ub = [], []
        for i in range(m):
            A_ub.append(Ms[i] - Rd[i]); b_ub.append(0.0)
            A_ub.append(-Ms[i] - Rd[i]); b_ub.append(0.0)
        res = linprog(
            np.zeros(n),
            A_ub=np.array(A_ub), b_ub=np.array(b_ub),
            A_eq=np.array([np.ones(n)]), b_eq=np.array([1.0]),
            bounds=[(0, None)] * n, method='highs'
        )
        if res.success:
            x = s * res.x
            if np.linalg.norm(x) > eps:
                return True, x
    return False, None


def find_delta_min(M, R, eps=1e-12):
    lo, hi = 0.0, 10.0
    while hi - lo > eps:
        mid = 0.5 * (lo + hi)
        ok, _ = is_singular(M, R, mid)
        if ok:
            hi = mid
        else:
            lo = mid
    _, x = is_singular(M, R, hi)
    return hi, x


def build_singular(M, R, delta):
    m, n = M.shape
    if n == 2:
        lows = [(M[i, 0] - delta * R[i, 0]) / (M[i, 1] + delta * R[i, 1])
                for i in range(m)]
        ups = [(M[i, 0] + delta * R[i, 0]) / (M[i, 1] - delta * R[i, 1])
               for i in range(m)]
        lam = 0.5 * (max(lows) + min(ups))
        a2 = np.zeros(m)
        for i in range(m):
            lo1 = (M[i, 0] - delta * R[i, 0]) / lam
            hi1 = (M[i, 0] + delta * R[i, 0]) / lam
            lo2 = M[i, 1] - delta * R[i, 1]
            hi2 = M[i, 1] + delta * R[i, 1]
            a2[i] = 0.5 * (max(lo1, lo2) + min(hi1, hi2))
        return np.column_stack([lam * a2, a2]), lam
    else:
        for j1, j2 in combinations(range(n), 2):
            lo = np.maximum(M[:, j1] - delta * R[:, j1],
                            M[:, j2] - delta * R[:, j2])
            hi = np.minimum(M[:, j1] + delta * R[:, j1],
                            M[:, j2] + delta * R[:, j2])
            if np.all(lo <= hi + 1e-6):
                c = 0.5 * (lo + hi)
                A = M.copy()
                A[:, j1] = c
                A[:, j2] = c
                return A, (j1, j2)
    return None, None


def in_interval(A, M, R, delta, tol=1e-6):
    return np.all(A >= M - delta * R - tol) and np.all(A <= M + delta * R + tol)


if __name__ == "__main__":

    #
    M1 = np.array([[0.95, 1.00], [1.05, 1.00], [1.10, 1.00]])
    R1a = np.ones((3, 2))
    d1a, _ = find_delta_min(M1, R1a)
    A1a, lam1 = build_singular(M1, R1a, d1a)
    print(f"Случай 1: delta_min = {d1a:.6f}, lambda = {lam1:.6f}, "
          f"rank = {np.linalg.matrix_rank(A1a)}, "
          f"в интервале = {in_interval(A1a, M1, R1a, d1a)}")
    print("A1' =\n", A1a)

    #
    R1b = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    d1b, _ = find_delta_min(M1, R1b)
    A1b, lam2 = build_singular(M1, R1b, d1b)
    print(f"\nСлучай 2: delta_min = {d1b:.6f}, lambda = {lam2:.6f}, "
          f"rank = {np.linalg.matrix_rank(A1b)}, "
          f"в интервале = {in_interval(A1b, M1, R1b, d1b)}")
    print("A1' =\n", A1b)

    #
    M2 = np.array([[1.10, 0.90, 1.10],
                   [1.40, 1.00, 0.80],
                   [0.80, 1.40, 1.20]])
    R2 = np.ones((3, 3))
    d2, _ = find_delta_min(M2, R2)
    A2, pair = build_singular(M2, R2, d2)
    print(f"\nСлучай 3: delta_min = {d2:.6f}, "
          f"det = {np.linalg.det(A2):.3e}, "
          f"в интервале = {in_interval(A2, M2, R2, d2)}")
    print("A2' =\n", A2)