"""Apply the PREREGISTERED held-out split to the alignment defence (open item #2).

research_design_detailed.md:882 fixes "Development set = 135M; held-out = 500M. No rung,
threshold or ordering is tuned on held-out data." That split was never applied: the defence
threshold was read off the same pairs it was evaluated on, which the manuscript admits.

Here the threshold is fit ONLY on development pairs (<= 160M parameters) and then applied
unchanged to the held-out pairs (>= 360M). Distillation pairs are excluded from both arms
for the reason already stated in the paper: differing depth makes index-to-index layer
comparison the wrong correspondence.

Pure recomputation over M4/defence_expanded.json plus the pythia-1b negative from
M4/p_tilde_arm.jsonl. No new models, no new measurement.
"""
import json, pathlib

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))


R = pathlib.Path(_ROOT)
SIZE = {  # approximate parameter count of the PARENT, used only to assign the split
 "HuggingFaceTB/SmolLM2-135M": 135e6, "Qwen/Qwen2.5-0.5B": 500e6,
 "google-bert/bert-base-uncased": 110e6, "bigscience/bloom-560m": 560e6,
 "FacebookAI/roberta-base": 125e6, "openai-community/gpt2": 124e6,
 "EleutherAI/pythia-70m": 70e6, "EleutherAI/pythia-160m": 160e6,
 "EleutherAI/pythia-410m": 410e6, "EleutherAI/pythia-70m-deduped": 70e6,
 "EleutherAI/pythia-1b": 1.0e9,
}
DEV_MAX = 160e6          # development: <= 160M   held-out: >= 360M

rows = [r for r in json.loads((R / "M4/defence_expanded.json").read_text())
        if isinstance(r.get("lap"), (int, float)) and r["lap"] == r["lap"]]
rows = [r for r in rows if r["kind"] != "distill"]
rows.append(dict(cls="P~", kind="same-recipe", a="EleutherAI/pythia-1b",
                 b="EleutherAI/pythia-1b-deduped", lap=0.17145))

for r in rows:
    n = SIZE.get(r["a"])
    r["scale"] = n
    r["arm"] = "dev" if (n is not None and n <= DEV_MAX) else "heldout"
    r["is_pos"] = (r["cls"] == "P+")

dev = [r for r in rows if r["arm"] == "dev"]
hold = [r for r in rows if r["arm"] == "heldout"]

print("=" * 78)
print("PREREGISTERED HELD-OUT SPLIT  (develop <=160M, hold out >=360M)")
print("=" * 78)
for arm, name in ((dev, "DEVELOPMENT"), (hold, "HELD-OUT")):
    print(f"\n{name}  (n={len(arm)}: {sum(r['is_pos'] for r in arm)} derived, "
          f"{sum(not r['is_pos'] for r in arm)} negative)")
    for r in sorted(arm, key=lambda r: -r["lap"]):
        print(f"  {'derived ' if r['is_pos'] else 'negative':<9}{r['lap']:.4f}  "
              f"{r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}")

# --- threshold fit on DEVELOPMENT ONLY -------------------------------------------------
dpos = [r["lap"] for r in dev if r["is_pos"]]
dneg = [r["lap"] for r in dev if not r["is_pos"]]
if not dpos or not dneg:
    raise SystemExit("development arm lacks one class")
lo, hi = max(dneg), min(dpos)
thr = (lo + hi) / 2.0
print(f"\nTHRESHOLD FIT ON DEVELOPMENT ONLY")
print(f"  highest development negative = {lo:.4f}")
print(f"  lowest  development derived  = {hi:.4f}")
print(f"  threshold = midpoint         = {thr:.4f}   (development margin {hi-lo:+.4f})")

# --- apply UNCHANGED to held-out --------------------------------------------------------
print(f"\nAPPLIED UNCHANGED TO HELD-OUT")
tp = fp = tn = fn = 0
for r in sorted(hold, key=lambda r: -r["lap"]):
    pred = r["lap"] >= thr
    ok = (pred == r["is_pos"])
    tp += pred and r["is_pos"]; fp += pred and not r["is_pos"]
    tn += (not pred) and (not r["is_pos"]); fn += (not pred) and r["is_pos"]
    print(f"  {'derived ' if r['is_pos'] else 'negative':<9}{r['lap']:.4f}  "
          f"-> {'derived' if pred else 'not derived':<12}{'OK' if ok else 'WRONG'}   "
          f"{r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}")
n = len(hold)
acc = (tp + tn) / n if n else float("nan")
hpos = [r['lap'] for r in hold if r['is_pos']]; hneg = [r['lap'] for r in hold if not r['is_pos']]
print(f"\n  held-out n={n}   TP {tp}  TN {tn}  FP {fp}  FN {fn}   accuracy {acc:.3f}")
if hpos and hneg:
    print(f"  held-out margin (min derived - max negative) = {min(hpos)-max(hneg):+.4f}")
print(f"\n  VERDICT: {'threshold GENERALISES to held-out' if fp==0 and fn==0 else 'threshold does NOT generalise'}")

json.dump(dict(threshold=thr, dev_margin=hi-lo,
               dev=[{k: r[k] for k in ('cls','kind','a','b','lap','scale')} for r in dev],
               heldout=[{k: r[k] for k in ('cls','kind','a','b','lap','scale')} for r in hold],
               heldout_tp=tp, heldout_tn=tn, heldout_fp=fp, heldout_fn=fn,
               heldout_accuracy=acc,
               heldout_margin=(min(hpos)-max(hneg)) if (hpos and hneg) else None),
          open(R / "M4/heldout_split.json", "w"), indent=1)
print("\nwrote M4/heldout_split.json")
