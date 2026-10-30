"""One targeted string mutation per name: (relative file under src/llm_lookahead_probe, old, new).
apply(root, name) asserts the old text occurs exactly once."""
import sys
from pathlib import Path

M = {
    "pnl_long_sign": ("stats.py", "return r if action == Action.LONG", "return -r if action == Action.LONG"),
    "estimate_date_mean": ("stats.py", "gap = CI(float(d_all.mean()), lo, hi)",
                           "gap = CI(float(np.mean(sums / counts)), lo, hi)"),
    "iid_bootstrap": ("stats.py", "_, inv = np.unique(clusters, return_inverse=True)",
                      "inv = np.arange(len(values))"),
    "ratio_to_date_mean": ("stats.py", "return (w @ cluster_sums) / (w @ cluster_counts)",
                           "return (w @ (cluster_sums / cluster_counts)) / w.sum(axis=1)"),
    "diff_ignores_post_var": ("stats.py", 'dlo, dhi = _percentile(draws["pre"] - draws["post"], level)',
                              'dlo, dhi = _percentile(draws["pre"] - parts["post"]["gap"]["estimate"], level)'),
    "ci_level_shrunk": ("stats.py", "lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])",
                        "lo, hi = np.quantile(draws, [0.075, 0.925])"),
    "mock_price_threshold": ("mocks.py", "a = Action.LONG if closes[-1] > closes[0] else Action.SHORT",
                             "a = Action.LONG if closes[-1] > closes[0] + 0.5 else Action.SHORT"),
    "factor_per_bar": ("masking.py", "bars = tuple(Bar(b.date + shift, b.close * f) for b in obs.bars)",
                       "bars = tuple(Bar(b.date + shift, b.close * f * (1 + 0.01 * i)) for i, b in enumerate(obs.bars))"),
    "ticker_lookahead_loose": ("masking.py", r"(?![A-Za-z0-9_])(?![.\-][A-Za-z0-9])", r"(?![A-Za-z0-9_.])(?![.\-][A-Za-z0-9])"),
    "skip_leak_check": ("probe.py", "        checker.check(prompt_m, real_prices=[b.close for b in obs_u.bars] + [next_close])\n", ""),
    "asof_checks_noop": ("pit.py", "    for b in bars:\n        if b.date > as_of:", "    return\n    for b in bars:\n        if b.date > as_of:"),
    "verdict_or": ("stats.py", 'elif parts["pre"]["gap"]["low"] > 0 and diff is not None and diff.low > 0:',
                   'elif parts["pre"]["gap"]["low"] > 0 or (diff is not None and diff.low > 0):'),
    "verdict_drop_diff": ("stats.py", 'elif parts["pre"]["gap"]["low"] > 0 and diff is not None and diff.low > 0:',
                          'elif parts["pre"]["gap"]["low"] > 0:'),
    "mock_no_recall": ("mocks.py", "remembered = self.memory.get((m.group(1), m.group(2))) if m else None",
                       "remembered = None"),
    "thinking_enabled_typo": ("live.py", 'request_extra={"thinking": {"type": "disabled"}}',
                              'request_extra={"thinking": {"type": "disable"}}'),
    "usage_field_rename": ("live.py", "tin, tout = int(resp.usage.input_tokens), int(resp.usage.output_tokens)",
                           "tin, tout = int(resp.usage.prompt_tokens), int(resp.usage.output_tokens)"),
}

def apply(root, name):
    f, old, new = M[name]
    p = Path(root) / "src" / "llm_lookahead_probe" / f
    s = p.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    p.write_text(s.replace(old, new))

if __name__ == "__main__":
    apply(sys.argv[1], sys.argv[2])
