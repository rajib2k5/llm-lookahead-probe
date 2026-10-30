"""C14: constant masking effect (no cutoff dependence) must not trigger 'suspected'.
Mock: MemorisingMock with mock_cutoff far in the future (unmasked arm recalls every date, pre and post
alike), so gap_pre ~= gap_post > 0 and only the pre-post test can keep the verdict at 'no evidence'."""
import sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import run, wilson
bad = False
for cfg, n in (("SMALL", 1000), ("DEFAULT", 400)):
    res = run("memorising", cfg, range(n), {"mock_cutoff": date(2100, 1, 1), "recall_p": 0.3})
    k = sum(r["verdict"] == "memorisation suspected" for r in res); lo, hi = wilson(k, n)
    kp = sum(r["pre_low"] > 0 for r in res)
    print(f"{cfg}: gap_pre CI low>0 in {kp/n:.3f} of seeds; false 'suspected' {k}/{n} = {k/n:.4f} Wilson [{lo:.4f},{hi:.4f}]")
    if lo > 0.05: bad = True
sys.exit(1 if bad else 0)
