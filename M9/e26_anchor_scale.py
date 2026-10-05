"""E26 -- does the from-scratch displacement anchor hold at scale?

WHY. Section VII prices evasion as a FRACTION of the displacement two independent
training runs show: crossing 0.659 / anchor 1.2044 = 55%, and 1.020 / 1.2044 = 85%.
Those are the paper's headline cost figures. The anchor (M5/displacement_reference.json,
frozen) is computed from FOUR pairs, all at pythia-70m and pythia-160m -- the two
smallest scales in the paper. E23 extended the soundness ladder to 6.9B but the pricing
anchor was left at 160M, so the scale objection the paper answers elsewhere is still open
underneath its cost claim.

This recomputes embedding displacement on every independent same-recipe pair available,
70m through 6.9b, and asks whether the anchor moves with scale. If it holds, the 55/85
fractions survive a scale challenge. If it drifts, the paper must say so.

Frozen data is never touched: M5/displacement_reference.json is read only, and results
are written to a NEW file.

Memory: one model resident at a time; only the embedding matrix is retained.
"""
import os, gc, json, pathlib, warnings, logging
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch, numpy as np
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
OUT = ROOT / "M9" / "e26_anchor_scale.json"

# every independent same-recipe pair whose weights are already local
PAIRS = [
    ("70m",   "EleutherAI/pythia-70m",   "EleutherAI/pythia-70m-deduped"),
    ("70m",   "EleutherAI/pythia-70m",   "EleutherAI/pythia-70m-v0"),
    ("160m",  "EleutherAI/pythia-160m",  "EleutherAI/pythia-160m-deduped"),
    ("160m",  "EleutherAI/pythia-160m",  "EleutherAI/pythia-160m-v0"),
    ("410m",  "EleutherAI/pythia-410m",  "EleutherAI/pythia-410m-deduped"),
    ("1b",    "EleutherAI/pythia-1b",    "EleutherAI/pythia-1b-deduped"),
    ("1.4b",  "EleutherAI/pythia-1.4b",  "EleutherAI/pythia-1.4b-deduped"),
    ("6.9b",  "EleutherAI/pythia-6.9b",  "EleutherAI/pythia-6.9b-deduped"),
]

def embed(mid):
    """Return the input embedding matrix in float32. One model resident at a time."""
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16,
                                             low_cpu_mem_usage=True).eval()
    w = m.get_input_embeddings().weight.detach().float().clone()
    del m; gc.collect()
    return w

def disp(a, b):
    """||E_b - E_a||_F / ||E_a||_F, the paper's relative embedding displacement."""
    A = embed(a); B = embed(b)
    if A.shape != B.shape:
        del A, B; gc.collect(); return None
    d = float(torch.linalg.matrix_norm(B - A) / torch.linalg.matrix_norm(A))
    del A, B; gc.collect()
    return d

def main():
    rows = []
    for scale, a, b in PAIRS:
        try:
            d = disp(a, b)
        except Exception as e:
            print(f"  {scale:6} SKIP {type(e).__name__}: {str(e)[:60]}", flush=True); continue
        if d is None:
            print(f"  {scale:6} SKIP shape mismatch", flush=True); continue
        rows.append({"scale": scale, "a": a, "b": b, "displacement_embed": d})
        print(f"  {scale:6} {a.split('/')[-1]:22} vs {b.split('/')[-1]:24} d_emb = {d:.4f}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

    # reproduce the frozen anchor from this code, as a validation
    froz = json.loads((ROOT / "M5" / "displacement_reference.json").read_text())
    fmean = float(np.mean([r["displacement_embed"] for r in froz]))
    small = [r["displacement_embed"] for r in rows if r["scale"] in ("70m", "160m")]
    allv  = [r["displacement_embed"] for r in rows]
    print(f"\n  frozen anchor (4 pairs, 70m+160m)      = {fmean:.4f}")
    print(f"  recomputed here (70m+160m subset)      = {np.mean(small):.4f}  n={len(small)}")
    print(f"  ALL scales 70m-6.9b                    = {np.mean(allv):.4f}  n={len(allv)}")
    big = [r["displacement_embed"] for r in rows if r["scale"] in ("1b", "1.4b", "6.9b")]
    if big:
        print(f"  large only (1b, 1.4b, 6.9b)            = {np.mean(big):.4f}  n={len(big)}")
    for name, anch in (("frozen", fmean), ("all-scale", float(np.mean(allv)))):
        print(f"  fractions under the {name:9} anchor: rejection {0.659/anch:.1%}, evasion bar {1.020/anch:.1%}")

if __name__ == "__main__":
    main()
