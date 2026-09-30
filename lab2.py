import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize

class Interval:
    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)
    def __add__(self, o):
        if isinstance(o, (int, float)): return Interval(self.lo+o, self.hi+o)
        return Interval(self.lo+o.lo, self.hi+o.hi)
    __radd__ = __add__
    def __sub__(self, o):
        if isinstance(o, (int, float)): return Interval(self.lo-o, self.hi-o)
        return Interval(self.lo-o.hi, self.hi-o.lo)
    def __rsub__(self, o): return Interval(o, o) - self
    def __mul__(self, o):
        if isinstance(o, (int, float)):
            return Interval(min(self.lo*o, self.hi*o), max(self.lo*o, self.hi*o))
        p = [self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi]
        return Interval(min(p), max(p))
    __rmul__ = __mul__
    def __truediv__(self, o):
        if isinstance(o, (int, float)): return self * (1.0/o)
        return self * Interval(1.0/o.hi, 1.0/o.lo)
    def __pow__(self, n):
        if n == 0: return Interval(1, 1)
        if n % 2 == 0:
            lo = 0.0 if self.lo <= 0 <= self.hi else min(self.lo**n, self.hi**n)
            return Interval(lo, max(self.lo**n, self.hi**n))
        return Interval(self.lo**n, self.hi**n)
    def __neg__(self): return Interval(-self.hi, -self.lo)
    def rad(self): return (self.hi - self.lo)/2.0
    def mid(self): return (self.lo + self.hi)/2.0
    def hausdorff(self, o):
        return max(abs(self.lo-o.lo), abs(self.hi-o.hi))
    def intersect(self, o):
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        return Interval(lo, hi) if lo <= hi else None
    def __repr__(self): return f"[{self.lo:.6f}, {self.hi:.6f}]"


def interval_sin(x):
    a, b = x.lo, x.hi
    if b - a >= 2*math.pi: return Interval(-1, 1)
    k1 = math.ceil((a - math.pi/2)/(2*math.pi))
    max_v = 1.0 if math.pi/2 + 2*k1*math.pi <= b else max(math.sin(a), math.sin(b))
    k2 = math.ceil((a + math.pi/2)/(2*math.pi))
    min_v = -1.0 if -math.pi/2 + 2*k2*math.pi <= b else min(math.sin(a), math.sin(b))
    return Interval(min_v, max_v)

def interval_cos(x):
    return interval_sin(Interval(x.lo + math.pi/2, x.hi + math.pi/2))


def f1_scalar(x):     return x**3 - 3*x**2 + 2
def f1_int(x):        return x**3 - 3*x**2 + 2
def f1_horner_int(x): return ((x - 3)*x)*x + 2
def f1_deriv_int(x):  return 3*(x - 1)**2 - 3

def f1_slope_analytic(X, c):
    xv = (3 - c) / 2
    xs = [X.lo, X.hi] + ([xv] if X.lo <= xv <= X.hi else [])
    vals = [x*x + (c-3)*x + c*c - 3*c for x in xs]
    return Interval(min(vals), max(vals))

def f2_scalar(x):     return x**3 + 2*math.sin(x)
def f2_int(x):        return x**3 + 2*interval_sin(x)
F2_DERIV_HI = 12.0 + 2*math.cos(2)
def f2_deriv_int(x):  return Interval(2.0, F2_DERIV_HI)

def f2_slope_numerical(X, c, n=4000):
    def slope(x):
        if abs(x - c) < 1e-10:
            h = 1e-6
            return (f2_scalar(c+h) - f2_scalar(c-h)) / (2*h)
        return (f2_scalar(x) - f2_scalar(c)) / (x - c)
    xs = np.linspace(X.lo, X.hi, n)
    vals = [slope(x) for x in xs]
    r1 = minimize_scalar(slope, bounds=(X.lo, X.hi), method='bounded')
    r2 = minimize_scalar(lambda x: -slope(x), bounds=(X.lo, X.hi), method='bounded')
    return Interval(min(min(vals), r1.fun), max(max(vals), -r2.fun))

def f3_scalar(x, y):  return x**2 + 3*y**2 + 2*x*y - 2*x - 2*y
def f3_int(x, y):     return x**2 + 3*y**2 + 2*x*y - 2*x - 2*y
def f3_alg_int(x, y): return (x + y - 1)**2 + 2*y**2 - 1
def f3_dx_int(x, y):  return 2*x + 2*y - 2
def f3_dy_int(x, y):  return 6*y + 2*x - 2


def centered_form_1d(f_int, f_deriv_int, X, c):
    return f_int(Interval(c, c)) + f_deriv_int(X) * (X - c)

def slope_form_1d(f_scalar, slope_range_fn, X, c):
    fc = Interval(f_scalar(c), f_scalar(c))
    return fc + slope_range_fn(X, c) * (X - c)

def _baumann_p(dX):
    if dX.rad() == 0: return 0.0
    return max(-1.0, min(1.0, dX.mid()/dX.rad()))

def bi_centered_1d(f_int, f_deriv_int, dX, X):
    p = _baumann_p(dX)
    c_star, c_sup = X.mid() - p*X.rad(), X.mid() + p*X.rad()
    F1 = centered_form_1d(f_int, f_deriv_int, X, c_star)
    F2 = centered_form_1d(f_int, f_deriv_int, X, c_sup)
    return F1.intersect(F2), c_star, c_sup

def centered_form_2d(f_int, f_dx_int, f_dy_int, X1, X2, c1, c2):
    return (f_int(Interval(c1,c1), Interval(c2,c2))
            + f_dx_int(X1,X2)*(X1-c1) + f_dy_int(X1,X2)*(X2-c2))

def bi_centered_2d(f_int, f_dx_int, f_dy_int, dX1, dX2, X1, X2):
    p1, p2 = _baumann_p(dX1), _baumann_p(dX2)
    cs = (X1.mid() - p1*X1.rad(), X2.mid() - p2*X2.rad())
    cS = (X1.mid() + p1*X1.rad(), X2.mid() + p2*X2.rad())
    F1 = centered_form_2d(f_int, f_dx_int, f_dy_int, X1, X2, *cs)
    F2 = centered_form_2d(f_int, f_dx_int, f_dy_int, X1, X2, *cS)
    return F1.intersect(F2), cs, cS

def split_box(X1, X2, axis):
    if axis == 'x':
        m = X1.mid()
        return (Interval(X1.lo, m), X2), (Interval(m, X1.hi), X2)
    m = X2.mid()
    return (X1, Interval(X2.lo, m)), (X1, Interval(m, X2.hi))

def hull(A, B):
    return Interval(min(A.lo, B.lo), max(A.hi, B.hi))


if __name__ == "__main__":
    RAN_F1 = Interval(-2, 2)
    RAN_F2 = Interval(f2_scalar(-2), f2_scalar(2))
    RAN_F3 = Interval(-1, 16)
    L1, L2     = 9.0, 3*2*2 + 2*np.cos(2)
    L3x, L3y   = 6.0, 14.0

    X1 = Interval(0, 3)
    print("="*72); print("f1(x) = x^3 - 3x^2 + 2,  X = [0,3]"); print("="*72)
    print(f"A.   Точная область (аналит.)     : {RAN_F1}")
    print(f"B.1  Естественное                 : {f1_int(X1)}  dist={f1_int(X1).hausdorff(RAN_F1):.4f}")
    print(f"B.2  Горнер                       : {f1_horner_int(X1)}  dist={f1_horner_int(X1).hausdorff(RAN_F1):.4f}")

    for c in [0.0, 1.5, 3.0]:
        F = centered_form_1d(f1_int, f1_deriv_int, X1, c)
        print(f"B.3  Дифф. центрир. (c={c})       : {F}  dist={F.hausdorff(RAN_F1):.4f}")

    for c in [0.0, 1.5, 3.0]:
        F = slope_form_1d(f1_scalar, f1_slope_analytic, X1, c)
        print(f"B.4  Наклонная (c={c})            : {F}  dist={F.hausdorff(RAN_F1):.4f}")

    bic1, cs, cS = bi_centered_1d(f1_int, f1_deriv_int, f1_deriv_int(X1), X1)
    print(f"B.5  Бицентр. (c_*={cs:.3f}, c*={cS:.3f}) : {bic1}  dist={bic1.hausdorff(RAN_F1):.4f}")

    print(f"C.1  L = {L1}")
    print(f"C.2  L·rad(X) = {L1*X1.rad()}")
    print(f"C.3  rad F_nat = {f1_int(X1).rad()}")

    X2 = Interval(-2, 2)
    print(); print("="*72); print("f2(x) = x^3 + 2 sin x,  X = [-2,2]"); print("="*72)
    print(f"A.   Точная область (аналит.)     : {RAN_F2}")
    print(f"B.1  Естественное                 : {f2_int(X2)}  dist={f2_int(X2).hausdorff(RAN_F2):.4f}")

    for c in [-2.0, 0.0, 2.0]:
        F = centered_form_1d(f2_int, f2_deriv_int, X2, c)
        print(f"B.3  Дифф. центрир. (c={c})       : {F}  dist={F.hausdorff(RAN_F2):.4f}")

    for c in [-2.0, 0.0, 2.0]:
        F = slope_form_1d(f2_scalar, f2_slope_numerical, X2, c)
        print(f"B.4  Наклонная (c={c})            : {F}  dist={F.hausdorff(RAN_F2):.4f}")

    bic2, cs2, cS2 = bi_centered_1d(f2_int, f2_deriv_int, f2_deriv_int(X2), X2)
    print(f"B.5  Бицентр. (c_*={cs2:.3f}, c*={cS2:.3f}) : {bic2}  dist={bic2.hausdorff(RAN_F2):.4f}")

    print(f"C.1  L = {L2:.6f}")
    print(f"C.2  L·rad(X) = {L2*X2.rad():.6f}")
    print(f"C.3  rad F_nat = {f2_int(X2).rad():.6f}")

    Xa, Xb = Interval(-1, 2), Interval(-1, 2)
    print(); print("="*72); print("f3(x,y) = x²+3y²+2xy-2x-2y,  X=[-1,2]²"); print("="*72)
    print(f"E.2  Точная область (аналит.)     : {RAN_F3}")

    nat3 = f3_int(Xa, Xb);  alg3 = f3_alg_int(Xa, Xb)
    print(f"E.3  Естественное                 : {nat3}  dist={nat3.hausdorff(RAN_F3):.4f}")
    print(f"E.4  Алгебраическое               : {alg3}  dist={alg3.hausdorff(RAN_F3):.4f}")

    for (c1,c2) in [(0.5,0.5), (0.5,0.0)]:
        F = centered_form_2d(f3_int, f3_dx_int, f3_dy_int, Xa, Xb, c1, c2)
        print(f"E.5  Дифф. центрир. c=({c1},{c2})  : {F}  dist={F.hausdorff(RAN_F3):.4f}")

    dX1, dX2 = Interval(-6, 6), Interval(-10, 14)
    bic3, cs3, cS3 = bi_centered_2d(f3_int, f3_dx_int, f3_dy_int, dX1, dX2, Xa, Xb)
    print(f"E.6  Бицентр. c_*={cs3}, c*={cS3} : {bic3}  dist={bic3.hausdorff(RAN_F3):.4f}")

    for axis in ['x', 'y']:
        (P1, P2), (Q1, Q2) = split_box(Xa, Xb, axis)
        h = hull(f3_int(P1,P2), f3_int(Q1,Q2))
        print(f"E.7  Дробление по {axis}            : {h}  dist={h.hausdorff(RAN_F3):.4f}")

    print(f"E.8  L1={L3x}, L2={L3y}, оценка={L3x*Xa.rad()+L3y*Xb.rad()}")
    print(f"     rad F_nat = {nat3.rad()}")

    xs = np.linspace(0, 3, 400)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [f1_scalar(x) for x in xs]); plt.grid(True)
    plt.axhline(RAN_F1.lo, color='r', ls='--'); plt.axhline(RAN_F1.hi, color='g', ls='--')
    plt.title('f1(x) = x^3 - 3x^2 + 2'); plt.show()

    xs = np.linspace(-2, 2, 400)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [f2_scalar(x) for x in xs]); plt.grid(True)
    plt.axhline(RAN_F2.lo, color='r', ls='--'); plt.axhline(RAN_F2.hi, color='g', ls='--')
    plt.title('f2(x) = x^3 + 2 sin x'); plt.show()

    x = np.linspace(-1, 2, 100); y = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x, y); Z = f3_scalar(X, Y)
    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85)
    ax1.set_title('Поверхность f3(x,y)')
    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax2); ax2.set_title('Линии уровня')
    plt.tight_layout(); plt.show()