"""O0 weight recovery: recover a signal weight from SCORES ALONE.

Open item #1. The paper currently states that our weight recovery divides a score delta by
a SIGNAL delta, that signal values are internal, and that the result is therefore O1 -- with
an O0 version conjectured but untested. This tests it.

THE CONSTRUCTION. An O0 adversary sees verdicts and scores, never signal values. To recover
a weight they need a perturbation whose effect on exactly one signal is analytically known.
Norm-preserving randomisation of a known fraction p of the non-embedding, non-norm weight
matrices does that:

  * the embedding is untouched      -> EAS and END cannot move (they read only the embedding)
  * norm-layer tensors are untouched-> NLF cannot move
  * each tensor's Frobenius norm is preserved exactly -> LEP (layer-energy profile) cannot move
  * a fraction p of rows is replaced by random directions -> those rows lose positional
    correspondence entirely, so WVC falls by approximately p

Hence  d(sigma_id) = w_wvc * d(WVC) ~= w_wvc * p, and the adversary recovers
w_wvc = d(sigma_id)/p using only p, which they chose, and two observed scores.

We ALSO record the true WVC from the comparison output. The adversary does not get to see
it; we use it only to check the dWVC ~= p assumption the method rests on.

Reference against the parent's own weights, so the untransformed pair scores ~1.0.
"""
import os, sys, json, shutil, copy, subprocess, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES = pathlib.Path(str(_ROOT) + "/M2/results")
WORK = pathlib.Path(str(_ROOT) + "/M2/work"); WORK.mkdir(exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
FRACS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]


def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=5400,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}


def eligible(name, param):
    """2-D projection matrices only: no embedding, no lm_head, no norm-layer vectors."""
    if param.ndim != 2:
        return False
    low = name.lower()
    if any(k in low for k in ("embed", "wte", "wpe", "lm_head", "shared")):
        return False
    if "norm" in low or "ln_" in low:
        return False
    return True


def randomise(model, p, seed=0):
    """Replace a fraction p of ROWS with random directions, preserving each row's norm and
    therefore each tensor's Frobenius norm. Layer energy is invariant by construction."""
    g = torch.Generator().manual_seed(seed)
    moved = total = 0
    for nme, prm in model.named_parameters():
        if not eligible(nme, prm):
            continue
        W = prm.data
        R = W.shape[0]
        k = int(round(p * R))
        total += R
        if k == 0:
            continue
        idx = torch.randperm(R, generator=g)[:k]
        rows = W[idx].float()
        norms = rows.norm(dim=1, keepdim=True)                 # preserve each row norm
        rnd = torch.randn(rows.shape, generator=g)
        rnd = rnd / (rnd.norm(dim=1, keepdim=True) + 1e-12) * norms
        W[idx] = rnd.to(W.dtype)
        moved += k
    return moved, total


tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL).eval()
out_rows = []
print(f"reference model: {MODEL}", flush=True)
print(f"{'p':>6}{'rows moved':>12}{'sigma_id':>10}{'EAS':>9}{'END':>9}{'LEP':>9}{'NLF':>9}{'WVC':>9}", flush=True)
print("-" * 76, flush=True)
for p_frac in FRACS:
    m = copy.deepcopy(base)
    moved, total = randomise(m, p_frac, seed=17)
    out = WORK / f"o0_p{int(p_frac*1000):04d}"
    if out.exists(): shutil.rmtree(out)
    m.save_pretrained(out); tok.save_pretrained(out)
    del m
    r = mpk(MODEL, str(out))
    shutil.rmtree(out, ignore_errors=True)
    sc, sg = r.get("scores", {}), r.get("signals", {})
    nf = lambda x: float("nan") if x is None else x
    row = dict(p=p_frac, rows_moved=moved, rows_total=total,
               sigma_id=sc.get("identity_score"), tier=sc.get("mfi_tier"),
               signals={k: sg.get(k) for k in ("eas", "nlf", "lep", "end", "wvc")})
    out_rows.append(row)
    print(f"{p_frac:>6.3f}{moved:>12}{nf(row['sigma_id']):>10.4f}"
          f"{nf(sg.get('eas')):>9.4f}{nf(sg.get('end')):>9.4f}{nf(sg.get('lep')):>9.4f}"
          f"{nf(sg.get('nlf')):>9.4f}{nf(sg.get('wvc')):>9.4f}", flush=True)

(RES / "o0_weight_recovery.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
print("\nwrote M2/results/o0_weight_recovery.jsonl", flush=True)
print("DONE-O0", flush=True)
