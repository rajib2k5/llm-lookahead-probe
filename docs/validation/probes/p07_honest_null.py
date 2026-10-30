"""C7: honest mock base action identical across arms; mean honest gap over seeds ~ 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from worker import run
fails = []
for cfg, n in (("DEFAULT", 1000), ("SMALL", 1000)):
    res = run("honest", cfg, range(n))
    mism = sum(r["base_mismatch"] for r in res); tot = sum(r["n_ep"] for r in res)
    est = np.array([r["est"] for r in res]); t = est.mean() / (est.std(ddof=1) / np.sqrt(n))
    print(f"{cfg}: base-action mismatches {mism} of {tot} episodes; mean gap {est.mean():.3e} "
          f"sd {est.std(ddof=1):.3e} t={t:.2f}; share of seeds with est>0: {(est>0).mean():.3f}")
    print(f"{cfg}: seed0 est {est[0]:.6f}")
    if mism: fails.append(f"{cfg} mismatch")
    if abs(t) > 3: fails.append(f"{cfg} t")
print("fails:", fails); sys.exit(1 if fails else 0)
