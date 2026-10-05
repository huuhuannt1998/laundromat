"""E23 — the P~ arm at 2.8B and 6.9B, extending the ladder from 1b to 6.9b.

WHY THIS EXISTS. The mock review's standing objection is scale: every empirical arm
stopped at 1.4B, so "recipe convergence defeats lineage verification" was only ever
shown on small models. Pythia publishes the same recipe trained independently on the
Pile and the deduplicated Pile at 2.8b and 6.9b, so the ladder extends with no change
of protocol -- only the scale varies.

WHY IT IS RUNNABLE ON ONE 24 GB LAPTOP, contrary to the earlier N/A. Two measured
facts, not assumptions:
  (1) MPK's scanner.compare extracts model A to a small feature record, then extracts
      model B, then scores the two records. Weights are never co-resident, and the
      loader pins torch_dtype=float16 (services/model_loader.py). Peak is ONE model:
      13.8 GB at 6.9b.
  (2) The LAP defence as written in M4/p_tilde_arm.py DOES hold both models at once,
      in fp32 -- 51.5 GB at 6.9b, which is why it looked impossible. This file fixes
      that by reducing each model to its subsampled up-projections, spilling them to
      disk, and freeing the model before the second one is loaded.

EXACT COMPARABILITY WITH THE 1b LADDER. M4's lap_colnorm column-normalises, then
subsamples columns, then row-normalises. Column normalisation is per-column, so
subsampling before or after it gives identical values; this file subsamples first
only to bound memory. The stride, max_feat=1536 and the mean-over-layers reduction
are unchanged, so E23 numbers are directly comparable to the existing P~ rows.

Nothing here downloads implicitly: run fetch.py first, or the MPK call will pull
weights on demand.
"""
import os, sys, gc, json, time, pathlib, subprocess, resource
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_grad_enabled(False)

# Repo root from this file's location; LAUNDROMAT_ROOT overrides. Never hardcode an
# author path -- the clean-room audit found 21 scripts that did and read the wrong tree.
ROOT = pathlib.Path(os.environ.get("LAUNDROMAT_ROOT",
       next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())))
REPO = ROOT / "M1/oracle/model-provenance-kit"
OUT  = ROOT / "M9"
RED  = OUT / "reduced"          # spilled up-projections, one .npy per model
RED.mkdir(parents=True, exist_ok=True)

UP_KEYS = ("up_proj.weight", "dense_h_to_4h.weight", "c_fc.weight", "fc_in.weight")
MAX_FEAT = 1536                 # identical to M4/p_tilde_arm.py

# (class, a, b). Labels are documented by the publisher, never inferred:
#   P~  Pythia trains Pile and deduplicated-Pile runs independently on one recipe.
#   D   the child's own model card names the parent in base_model / prose.
PAIRS = [
    # Every label below was verified IN THE WEIGHTS with verify_label.py before use.
    # Published labels failed that check three times out of four, so none is trusted:
    #   pythia-2.8b-deduped   cos 0.997 to pythia-2.8b        -> near-copy, not an
    #                         independent run. EXCLUDED.
    #   lambda/...-synthetic-instruct  cos 0.191 to the parent its card names
    #                         -> not weight-derived. EXCLUDED.
    #   pszemraj/pythia-6.9b-HC3  cos 0.325 / 0.339 to pythia-6.9b / -deduped
    #                         -> derived from neither. EXCLUDED.
    # The verifier is calibrated: it reads 0.9999 on a true fine-tune
    # (pythia-410m-deduped -> mnoukhov/pythia410m-sft-tldr), 0.1995 on that same
    # child against the WRONG parent, and 1.0000 on identity.
    ("P~ 6.9b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-6.9b-deduped"),   # cos 0.3215: independent
    ("P~ cross-scale", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b"),    # no label claim
]


def peak_gb():
    """Peak RSS of this process. macOS reports ru_maxrss in bytes, Linux in KiB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9

def mpk(a, b):
    """MPK's own verdict. Identical invocation to M4/p_tilde_arm.py, longer timeout."""
    p = subprocess.run(
        ["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
        cwd=REPO, capture_output=True, text=True, timeout=21600,
        env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}

def reduce_model(model_id):
    """Load ONE model fp16, keep only subsampled up-projections, spill, free.

    Returns the .npy path. This is the whole reason 6.9b fits: the model is gone
    before its partner is ever loaded.
    """
    tag = model_id.replace("/", "__")
    dst = RED / f"{tag}.npy"
    if dst.exists():
        return dst
    from transformers import AutoModelForCausalLM
    t = time.monotonic()
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    mats = []
    for n, p in m.named_parameters():
        if p.ndim == 2 and any(k in n for k in UP_KEYS):
            A = p.detach()
            if A.shape[1] > MAX_FEAT:                      # same stride as M4
                f = torch.arange(0, A.shape[1], max(1, A.shape[1] // MAX_FEAT))[:MAX_FEAT]
                A = A[:, f]
            mats.append(A.clone().float().numpy())
    del m; gc.collect()
    arr = np.stack(mats) if mats else np.zeros((0, 0, 0), np.float32)
    np.save(dst, arr)
    print(f"    reduced {model_id}: {arr.shape} in {time.monotonic()-t:.0f}s, "
          f"peak {peak_gb():.1f} GB", flush=True)
    del arr, mats; gc.collect()
    return dst

def lap_colnorm(A, B):
    """Column-normalise, row-normalise, Hungarian, mean matched similarity.

    Column subsampling already happened in reduce_model; because column
    normalisation is per-column that reordering is exact, not an approximation.
    """
    if A.shape != B.shape:
        return None
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())            # float() -- E19 died on np types

def defence(a, b):
    pa, pb = reduce_model(a), reduce_model(b)
    A, B = np.load(pa, mmap_mode="r"), np.load(pb, mmap_mode="r")
    vals = []
    for i in range(min(len(A), len(B))):
        t = time.monotonic()
        s = lap_colnorm(np.asarray(A[i], np.float32), np.asarray(B[i], np.float32))
        if s is not None:
            vals.append(s)
            print(f"      layer {i:2d} LAP={s:.4f} ({time.monotonic()-t:.0f}s)", flush=True)
    return float(np.mean(vals)) if vals else float("nan")

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    res = OUT / "e23_scale.jsonl"
    done = set()
    if res.exists():                                   # resumable: 6.9b pairs are long
        for l in res.open():
            try: done.add(json.loads(l)["class"])
            except Exception: pass
    with res.open("a") as fh:
        for cls, a, b in PAIRS:
            if cls in done or (only and only not in cls):
                print(f"[skip] {cls}"); continue
            print(f"\n=== {cls}: {a}  vs  {b} ===", flush=True)
            t0 = time.monotonic()
            r = mpk(a, b)
            if "error" in r:
                print(f"  MPK ERROR {r['error'][:200]}", flush=True); continue
            s, g = r["scores"], r["signals"]
            print(f"  MPK tier={s['mfi_tier']} pipe={s['pipeline_score']:.4f} "
                  f"id={s['identity_score']:.4f} -> {s['provenance_decision']}", flush=True)
            d = defence(a, b)
            rec = {"class": cls, "a": a, "b": b, "scores": s, "signals": g,
                   "lap_defence": d, "peak_rss_gb": round(peak_gb(), 2),
                   "seconds": round(time.monotonic() - t0, 1)}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  LAP={d:.4f}  peak={rec['peak_rss_gb']:.1f} GB  "
                  f"{rec['seconds']/60:.1f} min", flush=True)
    print("\nDONE-E23")

if __name__ == "__main__":
    main()
