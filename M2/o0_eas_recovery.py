"""Second O0 recovery: the embedding-anchor weight, again from scores alone.

The first O0 experiment recovered w_WVC by randomising non-embedding rows. Its p=1.0 row
shows WVC bottoming out at 0.0027 with the embedding untouched, so the embedding contributes
essentially nothing to the positional signal. That makes the complementary construction
available: perturb ONLY the embedding and the positional signal should not move.

CONSTRUCTION. Replace a known fraction p of embedding rows with random directions of the
SAME NORM.
  * row norms are preserved  -> the embedding-norm-distribution signal cannot move
  * non-embedding tensors untouched -> the positional, layer-energy and norm-layer signals
    cannot move
  * a fraction p of embedding rows loses its geometry -> the anchor signal falls

The anchor signal reads 64 specific rows chosen by encoding fixed token strings. An O0
adversary does not know which rows those are, but randomising a fraction p of the vocabulary
hits p of them in expectation, so d(EAS) should track p up to binomial noise on 64 draws.
Whether it does is the question: if the anchor signal responds linearly, w_eas follows by
division; if it does not, that is itself a finding about how the signal is built.

Reference is the model against itself, so the untransformed pair scores 1.0.
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
MODEL = "HuggingFaceTB/SmolLM2-135M"
FRACS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]


def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=5400,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}


def randomise_embedding(model, p, seed=23):
    """Randomise a fraction p of embedding rows, preserving each row's norm."""
    g = torch.Generator().manual_seed(seed)
    emb = model.get_input_embeddings().weight
    V = emb.shape[0]
    k = int(round(p * V))
    if k == 0:
        return 0, V
    idx = torch.randperm(V, generator=g)[:k]
    rows = emb.data[idx].float()
    norms = rows.norm(dim=1, keepdim=True)
    rnd = torch.randn(rows.shape, generator=g)
    rnd = rnd / (rnd.norm(dim=1, keepdim=True) + 1e-12) * norms
    emb.data[idx] = rnd.to(emb.dtype)
    return k, V


tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL).eval()
rows = []
print(f"reference: {MODEL}   (embedding-only, norm-preserving)", flush=True)
print(f"{'p':>6}{'rows':>9}{'sigma_id':>10}{'EAS':>9}{'END':>9}{'LEP':>9}{'NLF':>9}{'WVC':>9}", flush=True)
print("-" * 70, flush=True)
for pf in FRACS:
    m = copy.deepcopy(base)
    k, V = randomise_embedding(m, pf)
    out = WORK / f"o0eas_p{int(pf*1000):04d}"
    if out.exists(): shutil.rmtree(out)
    m.save_pretrained(out); tok.save_pretrained(out); del m
    r = mpk(MODEL, str(out)); shutil.rmtree(out, ignore_errors=True)
    sc, sg = r.get("scores", {}), r.get("signals", {})
    nf = lambda x: float("nan") if x is None else x
    rows.append(dict(p=pf, rows_moved=k, vocab=V, sigma_id=sc.get("identity_score"),
                     signals={x: sg.get(x) for x in ("eas", "nlf", "lep", "end", "wvc")}))
    print(f"{pf:>6.3f}{k:>9}{nf(sc.get('identity_score')):>10.4f}{nf(sg.get('eas')):>9.4f}"
          f"{nf(sg.get('end')):>9.4f}{nf(sg.get('lep')):>9.4f}{nf(sg.get('nlf')):>9.4f}"
          f"{nf(sg.get('wvc')):>9.4f}", flush=True)

(RES / "o0_eas_recovery.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("\nwrote M2/results/o0_eas_recovery.jsonl", flush=True)
print("DONE-O0EAS", flush=True)
