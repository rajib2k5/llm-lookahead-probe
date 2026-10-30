"""C17 (probe-decidable part): 'claude-haiku-5-5' is in the installed SDK's Model literal list."""
import sys, typing
import anthropic
from anthropic.types import Model
def lits(t):
    out = set()
    for a in typing.get_args(t):
        if isinstance(a, str): out.add(a)
        else: out |= lits(a)
    return out
L = lits(Model)
print("anthropic", anthropic.__version__, "| literal count", len(L), "| haiku models:", sorted(x for x in L if "haiku" in x))
ok = "claude-haiku-5-5" in L; neg = "claude-haiku-5-55" in L
print("claude-haiku-5-5 in list:", ok, "| control claude-haiku-5-55 in list:", neg)
sys.exit(0 if ok and not neg else 1)
