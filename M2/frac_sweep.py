"""Ablation (design 11.5): permutation FRACTION swept inside the GATE arm.

Previously frac was varied only in the exactness harness, which answers "is it
still function-preserving?" but not "does partial permutation move the detector
more or less than full permutation?".  Composed with M1t so the metadata gate is
broken and sigma_id -- not the tier-1 shortcut -- is the deciding score.

Partial permutation is still EXACT: permuting a subset of indices among
themselves is a valid permutation of the whole index set (identity elsewhere),
so every point here remains kappa = 0 and function-preserving.
"""
import sys, json, pathlib
sys.path.insert(0, str(_ROOT) + "/M2")
sys.path.insert(0, str(_ROOT) + "/M2/transforms")
import laundry as LD
from eval_transform import EVAL, RES, PI_CUT, K_STAR

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))


PARENT = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
CHILD = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolLM2-135M-Instruct"
FRACS = [0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]

fh = (RES / f"frac_sweep_{CHILD.replace('/','--')}.jsonl").open("w")
print(f"pi* cut = {PI_CUT:.4f}  k* = {K_STAR}   (all points kappa=0, exact)\n", flush=True)
print(f"{'frac':>7}{'tier':>5}{'sigma':>9}{'sig_id':>9}{'EAS':>8}{'WVC':>8}{'rank':>6}{'pi':>7}{'E':>3}  exact  kind", flush=True)
print("-" * 84, flush=True)
for f in FRACS:
    ops = [] if f == 0.0 else [lambda m, f=f: LD.X1a_mlp_permute(m, f, 11)]
    r = EVAL(PARENT, CHILD, f"FRAC-{f:.4f}", ops, "both")
    fh.write(json.dumps({**r, "frac": f}, default=str) + "\n"); fh.flush()
    c = r["cmp"] or {}; s = r["signals"] or {}
    print(f"{f:>7.4f}{str(c.get('mfi_tier')):>5}{c.get('pipeline_score',0):>9.4f}"
          f"{c.get('identity_score',0):>9.4f}"
          f"{(s.get('eas') if s.get('eas') is not None else float('nan')):>8.4f}"
          f"{(s.get('wvc') if s.get('wvc') is not None else float('nan')):>8.4f}"
          f"{str(r['parent_rank']):>6}{(r['pi'] or 0):>7.1f}{r['E']:>3}"
          f"  {str(r['fidelity']['exact_within_tolerance']):<6} {r['kind']}", flush=True)
fh.close(); print("DONE-FRACSWEEP", flush=True)
