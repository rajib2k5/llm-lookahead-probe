"""C6: honest-null coverage of the 95% CIs on SMALL and DEFAULT configs (+ t-interval comparison)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from worker import run, wilson
N = {"SMALL": 2000, "DEFAULT": 2000}
refuted = False
for cfg, n in N.items():
    res = run("honest", cfg, range(n))
    for key in ("cov_all", "cov_pre", "cov_post", "cov_diff", "tcov_all", "tcov_pre", "tcov_post"):
        k = sum(r[key] for r in res); lo, hi = wilson(k, n)
        print(f"{cfg} seeds 0..{n-1} {key}: {k/n:.4f}  Wilson95 [{lo:.4f}, {hi:.4f}]")
        if key == "cov_all" and hi < 0.935: refuted = True
    k200 = sum(r["cov_all"] for r in res[:200]); print(f"{cfg} seeds 0..199 cov_all: {k200/200:.3f}")
print("REFUTED" if refuted else "CONFIRMED (per pre-registered rule)")
sys.exit(1 if refuted else 0)
