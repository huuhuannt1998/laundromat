"""Reference displacements: what does 'as expensive as training from scratch' look like?

C1 claims evading the weight signals costs as much as training from scratch.  The
corpus-distance sweep measures where the detector actually fails, on an axis of
relative weight displacement ||theta_c - theta_p|| / ||theta_p||.  To price C1 we
need the scale anchored at both ends:

  kappa = 0        our exact transforms          displacement 0 (weights permuted, norm preserved)
  independent      two separate training runs    <- MEASURED HERE, the empirical 'from scratch' point
  analytic         uncorrelated weights          ~sqrt(2) = 1.414

If the detector fails at displacement well below the independent-run value, C1 is
quantitatively false at kappa > 0 -- and the counterexamples are benign published models.
"""
import os, json, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModel

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

PAIRS = [
    ("independent run (same corpus)",  "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-v0"),
    ("independent run (dedup corpus)", "EleutherAI/pythia-160m", "EleutherAI/pythia-160m-deduped"),
    ("independent run (same corpus)",  "EleutherAI/pythia-70m",  "EleutherAI/pythia-70m-v0"),
    ("independent run (dedup corpus)", "EleutherAI/pythia-70m",  "EleutherAI/pythia-70m-deduped"),
]


def disp(a, b):
    ma, mb = AutoModel.from_pretrained(a).eval(), AutoModel.from_pretrained(b).eval()
    pa, pb = dict(ma.named_parameters()), dict(mb.named_parameters())
    num = den = 0.0; enum = eden = 0.0
    for k, v in pa.items():
        if k not in pb or pb[k].shape != v.shape:
            continue
        d = (pb[k].float() - v.float()).pow(2).sum().item()
        n = v.float().pow(2).sum().item()
        num += d; den += n
        if "embed" in k:
            enum += d; eden += n
    del ma, mb
    return (num/den)**0.5 if den else float("nan"), (enum/eden)**0.5 if eden else float("nan")


out = []
print(f"{'kind':<32}{'disp':>8}{'dispEmb':>9}  pair", flush=True)
print("-" * 92, flush=True)
for kind, a, b in PAIRS:
    try:
        d, de = disp(a, b)
        out.append(dict(kind=kind, a=a, b=b, displacement=d, displacement_embed=de))
        print(f"{kind:<32}{d:>8.4f}{de:>9.4f}  {a.split('/')[-1]} | {b.split('/')[-1]}", flush=True)
    except Exception as e:
        print(f"{kind:<32} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
json.dump(out, open(str(_ROOT) + "/M5/displacement_reference.json", "w"), indent=1)
if out:
    import statistics as st
    m = st.mean(x["displacement"] for x in out)
    print(f"\n  EMPIRICAL 'trained from scratch' displacement = {m:.4f}"
          f"   (analytic uncorrelated bound ~1.4142)", flush=True)
print("DONE-DISPREF", flush=True)
