"""Student-t quantile without scipy: bisection on the regularized incomplete beta CDF (math only)."""
from math import lgamma, exp, log
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, 300):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300); c = 1 + aa / c; de = d * c; h *= de
        if abs(de - 1) < 1e-14: break
    return h
def _ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b
def t_cdf(t, df):
    x = df / (df + t * t); p = 0.5 * _ibeta(df / 2, 0.5, x)
    return 1 - p if t > 0 else p
def t_ppf(q, df):
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2
if __name__ == "__main__":
    print(t_ppf(0.975, 29), t_ppf(0.975, 59), t_ppf(0.975, 1e6))  # 2.0452, 2.0010, 1.95996
