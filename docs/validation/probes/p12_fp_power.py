"""C12 (fp) / C13 (power): verdict rates under honest null and memorising mock, SMALL config."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import run, wilson
mode = sys.argv[1]
S = "memorisation suspected"
if mode == "fp":
    bad = False
    for cfg, n in (("SMALL", 2000), ("DEFAULT", 1000)):
        res = run("honest", cfg, range(n))
        k = sum(r["verdict"] == S for r in res); lo, hi = wilson(k, n)
        k200 = sum(r["verdict"] == S for r in res[:200])
        kpre = sum(r["pre_low"] > 0 for r in res); kd = sum(r["diff_low"] > 0 for r in res)
        print(f"{cfg}: false 'suspected' {k}/{n} = {k/n:.4f} Wilson [{lo:.4f},{hi:.4f}]; seeds 0-199: {k200/200:.3f}; "
              f"P(pre low>0)={kpre/n:.4f}; P(diff low>0)={kd/n:.4f}")
        if lo > 0.05: bad = True
        if cfg == "SMALL" and k200 / 200 != 0.020: print("NOTE: seeds 0-199 do not reproduce 0.020")
    sys.exit(1 if bad else 0)
else:
    res = run("memorising", "SMALL", range(200))
    k = sum(r["verdict"] == S for r in res); print(f"power p=0.75 SMALL seeds 0-199: {k/200:.3f}")
    for p in (0.55, 0.60):
        rr = run("memorising", "SMALL", range(200), {"recall_p": p})
        print(f"power p={p} SMALL seeds 0-199: {sum(r['verdict'] == S for r in rr)/200:.3f}")
    sys.exit(0 if k / 200 >= 0.95 else 1)
