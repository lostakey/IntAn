import numpy as np
from itertools import product
from pathlib import Path
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = 0.05
EPS = 1e-9
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

A_FULL = np.array([
    [[0.65, 1.25], [0.70, 1.30]],
    [[0.75, 1.35], [0.75, 1.31]],
    [[0.80, 1.40], [0.66, 1.26]],
    [[0.0, 0.0], [0.0, 0.0]],
])
B_FULL = np.array([
    [2.75, 3.15],
    [2.85, 3.25],
    [2.90, 3.30],
    [1.00, 1.20],
])


def build_system(m):
    return A_FULL[:m, :, 0].copy(), A_FULL[:m, :, 1].copy(), B_FULL[:m, 0].copy(), B_FULL[:m, 1].copy()


def generators(Alo, Ahi, blo, bhi):
    m, n = Alo.shape
    seen = set()
    rows, meta = [], []
    for i in range(m):
        for s in product((0, 1), repeat=n):
            a = np.array([Ahi[i, j] if s[j] else Alo[i, j] for j in range(n)])
            for typ, row in (("up", [bhi[i], *(-a)]), ("lo", [-blo[i], *a])):
                key = (i, typ, tuple(np.round(row, 12)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                meta.append((i + 1, typ, s))
    return np.array(rows), meta


def maximize_tol(G):
    K = len(G)
    res = linprog(
        [0.0, 0.0, -1.0],
        A_ub=np.column_stack([-G[:, 1:], np.ones(K)]),
        b_ub=G[:, 0],
        bounds=[(None, None)] * 3,
        method="highs",
    )
    return float(res.x[2]), res.x[:2] + 0.0


def tol_values(G, X, Y):
    return (G[:, 0][:, None, None] + G[:, 1][:, None, None] * X + G[:, 2][:, None, None] * Y).min(axis=0)


def tol_polygon(G, tau, T):
    if T <= EPS:
        return None
    hs = np.column_stack([-G[:, 1], -G[:, 2], -G[:, 0]])
    inter = HalfspaceIntersection(hs, tau).intersections
    hull = ConvexHull(inter)
    return inter[hull.vertices], float(hull.volume)


def analyze(Alo, Ahi, blo, bhi):
    G, meta = generators(Alo, Ahi, blo, bhi)
    T, tau = maximize_tol(G)
    poly = tol_polygon(G, tau, T)
    return {"A": (Alo, Ahi), "b": (blo, bhi), "G": G, "meta": meta, "T": T, "tau": tau, "poly": poly}


def max_tol_only(Alo, Ahi, blo, bhi):
    G, _ = generators(Alo, Ahi, blo, bhi)
    return maximize_tol(G)[0]


def shrink_matrix(Alo, Ahi, theta):
    mid = (Alo + Ahi) / 2
    rad = (Ahi - Alo) / 2 * (1 - theta)
    return mid - rad, mid + rad


def widen_b(blo, bhi, K):
    return blo - K, bhi + K


def correct_b(Alo, Ahi, blo, bhi, T):
    K = TARGET - T
    nblo, nbhi = widen_b(blo, bhi, K)
    return analyze(Alo, Ahi, nblo, nbhi), {"K": K, "theta": 0.0}


def correct_A(Alo, Ahi, blo, bhi):
    if max_tol_only(*shrink_matrix(Alo, Ahi, 1.0), blo, bhi) < TARGET:
        return None, {"theta_max_T": max_tol_only(*shrink_matrix(Alo, Ahi, 1.0), blo, bhi)}
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if max_tol_only(*shrink_matrix(Alo, Ahi, mid), blo, bhi) >= TARGET:
            hi = mid
        else:
            lo = mid
    theta = hi
    nAlo, nAhi = shrink_matrix(Alo, Ahi, theta)
    return analyze(nAlo, nAhi, blo, bhi), {"K": 0.0, "theta": theta}


def distortion(base, res, theta, K):
    m = base["A"][0].shape[0]
    radA = (base["A"][1] - base["A"][0]) / 2
    radb = (base["b"][1] - base["b"][0]) / 2
    nradA = (res["A"][1] - res["A"][0]) / 2
    nradb = (res["b"][1] - res["b"][0]) / 2
    dA = float(np.linalg.norm(radA - nradA))
    db = float(np.linalg.norm(radb - nradb))
    rel_A = dA / float(np.linalg.norm(radA))
    rel_b = db / float(np.linalg.norm(radb))
    return {
        "dA": dA,
        "db": db,
        "rel_A": rel_A,
        "rel_b": rel_b,
        "total_abs": float(np.hypot(dA, db)),
        "total_rel": float(np.hypot(rel_A, rel_b)),
        "max_dA": float(np.max(np.abs(radA - nradA))),
        "max_db": float(np.max(np.abs(radb - nradb))),
        "theta": theta,
        "K": K,
    }


def correct_Ab(Alo, Ahi, blo, bhi, base):
    best = None
    for theta in np.linspace(0.0, 1.0, 201):
        nAlo, nAhi = shrink_matrix(Alo, Ahi, theta)
        T = max_tol_only(nAlo, nAhi, blo, bhi)
        K = max(0.0, TARGET - T)
        nblo, nbhi = widen_b(blo, bhi, K)
        tmp = {"A": (nAlo, nAhi), "b": (nblo, nbhi)}
        d = distortion(base, tmp, theta, K)
        if best is None or d["total_rel"] < best[0] - 1e-12:
            best = (d["total_rel"], theta, K)
    theta, K = best[1], best[2]
    nAlo, nAhi = shrink_matrix(Alo, Ahi, theta)
    nblo, nbhi = widen_b(blo, bhi, K)
    return analyze(nAlo, nAhi, nblo, nbhi), {"K": K, "theta": theta}


def fmt_interval_matrix(lo, hi):
    lines = []
    for i in range(lo.shape[0]):
        lines.append("  " + "  ".join(f"[{lo[i, j]:.4f}, {hi[i, j]:.4f}]" for j in range(lo.shape[1])))
    return "\n".join(lines)


def fmt_interval_vector(lo, hi):
    return "\n".join(f"  [{lo[i]:.4f}, {hi[i]:.4f}]" for i in range(len(lo)))


def report(title, res):
    print("=" * 78)
    print(title)
    print("A =")
    print(fmt_interval_matrix(*res["A"]))
    print("b =")
    print(fmt_interval_vector(*res["b"]))
    T, tau = res["T"], res["tau"]
    print(f"max Tol = {T:.6f}")
    print(f"argmax Tol = ({tau[0]:.6f}, {tau[1]:.6f})")
    print(f"Tol set empty = {T < -EPS}")
    print("образующие Tol(x) = c0 + c1*x1 + c2*x2:")
    print("   k  i  тип  вершина         c0         c1         c2     g(argmax)  активна")
    for k, ((i, typ, s), row) in enumerate(zip(res["meta"], res["G"]), 1):
        g = row[0] + row[1] * tau[0] + row[2] * tau[1]
        active = abs(g - T) < 1e-7
        print(f"  {k:2d}  {i}  {typ}   {str(s):10s} {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {g:12.6f}  {'да' if active else ''}")
    if res["poly"] is not None:
        verts, area = res["poly"]
        print("вершины допускового множества:")
        for v in verts:
            print(f"  ({v[0]:.6f}, {v[1]:.6f})")
        print(f"площадь = {area:.6f}")
        print(f"брус: x1 in [{verts[:, 0].min():.6f}, {verts[:, 0].max():.6f}], "
              f"x2 in [{verts[:, 1].min():.6f}, {verts[:, 1].max():.6f}]")
    else:
        print("площадь = 0")


def plot_case(res, title, path):
    G, tau, T, poly = res["G"], res["tau"], res["T"], res["poly"]
    if poly is not None:
        verts = poly[0]
        x0, x1 = verts[:, 0].min(), verts[:, 0].max()
        y0, y1 = verts[:, 1].min(), verts[:, 1].max()
        px = max(x1 - x0, 0.05) * 0.4
        py = max(y1 - y0, 0.05) * 0.4
        x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    else:
        x0, x1 = tau[0] - 1.0, tau[0] + 1.0
        y0, y1 = tau[1] - 1.0, tau[1] + 1.0
    xs = np.linspace(x0, x1, 160)
    ys = np.linspace(y0, y1, 160)
    X, Y = np.meshgrid(xs, ys)
    Z = tol_values(G, X, Y)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
    ax1.scatter([tau[0]], [tau[1]], [T], color="red", s=60, depthshade=False,
                label=f"max Tol = {T:.4f}")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("Tol")
    ax1.set_title("Tol(x)")
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(X, Y, Z, levels=24, cmap="viridis")
    fig.colorbar(cf, ax=ax2, label="Tol")
    if Z.min() < 0 < Z.max():
        ax2.contour(X, Y, Z, levels=[0], colors="white", linewidths=1.2)
    if poly is not None:
        v = poly[0]
        ax2.fill(v[:, 0], v[:, 1], facecolor="#f2d649", edgecolor="black", linewidth=1.4, alpha=0.9,
                 label="Tol ≥ 0")
    ax2.plot([tau[0]], [tau[1]], marker="x", color="red", markersize=11, markeredgewidth=2.5,
             linestyle="none", label=f"argmax ({tau[0]:.3f}, {tau[1]:.3f})")
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title("Допусковое множество")
    ax2.legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    for idx, m in enumerate((2, 3, 4), 1):
        Alo, Ahi, blo, bhi = build_system(m)
        base = analyze(Alo, Ahi, blo, bhi)
        print("#" * 78)
        print(f"СИСТЕМА {idx}  ({m}x2)")
        report(f"Система {idx}: исходная", base)
        plot_case(base, f"Система {idx}: исходная", OUT / f"sys{idx}_original.png")

        if base["T"] >= 0:
            continue

        rb, pb = correct_b(Alo, Ahi, blo, bhi, base["T"])
        report(f"Система {idx}: b-коррекция (K = {pb['K']:.6f})", rb)
        plot_case(rb, f"Система {idx}: b-коррекция", OUT / f"sys{idx}_b.png")

        rA, pA = correct_A(Alo, Ahi, blo, bhi)
        if rA is None:
            print("=" * 78)
            print(f"Система {idx}: A-коррекция")
            print(f"max Tol при полном сужении матрицы до точечной = {pA['theta_max_T']:.6f}")
            print(f"разрешимость достигнута = False")
        else:
            report(f"Система {idx}: A-коррекция (theta = {pA['theta']:.6f})", rA)
            plot_case(rA, f"Система {idx}: A-коррекция", OUT / f"sys{idx}_A.png")

        rAb, pAb = correct_Ab(Alo, Ahi, blo, bhi, base)
        report(f"Система {idx}: Ab-коррекция (theta = {pAb['theta']:.6f}, K = {pAb['K']:.6f})", rAb)
        plot_case(rAb, f"Система {idx}: Ab-коррекция", OUT / f"sys{idx}_Ab.png")


if __name__ == "__main__":
    main()