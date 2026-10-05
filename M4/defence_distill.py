"""Open items #3 and #4: fix the matrix selector, and solve distillation layer correspondence.

#3 Two of seventeen trials in defence_expanded returned null because ups() matched
   "intermediate.dense.weight" (BERT/RoBERTa) but not DistilBERT's "ffn.lin1.weight".
   One of the lost pairs was an unrelated negative, so that class was a member short.

#4 Distilled children have fewer layers than their parents, so comparing layer i to layer i
   is the wrong correspondence and put both distillation pairs inside the negative range.
   DistilBERT/DistilGPT2/DistilRoBERTa are initialised by taking every other parent layer,
   so child layer i corresponds to parent layer 2i. We evaluate BOTH mappings and report
   both, rather than picking the flattering one.
"""
import os, json, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
OUT = pathlib.Path(str(_ROOT) + "/M4/defence_distill.json")

PAIRS = [
  ("P+", "distill",   "google-bert/bert-base-uncased",       "distilbert/distilbert-base-uncased"),
  ("P+", "distill",   "FacebookAI/roberta-base",             "distilbert/distilroberta-base"),
  ("P+", "distill",   "openai-community/gpt2",               "distilbert/distilgpt2"),
  ("P-", "unrelated", "distilbert/distilbert-base-uncased",  "distilbert/distilroberta-base"),
]

# the round-4 review traced the nulls to this list; ffn.lin1 is DistilBERT's FFN input proj
KEYS = ("up_proj.weight", "dense_h_to_4h.weight", "c_fc.weight", "fc_in.weight",
        "intermediate.dense.weight", "ffn.lin1.weight")


def ups(m):
    return [p for n, p in m.named_parameters() if p.ndim == 2 and any(k in n for k in KEYS)]


def lap(A, B, max_feat=1536):
    A, B = A.float(), B.float()
    if A.shape != B.shape:
        return None
    A = A / (A.norm(dim=0, keepdim=True) + 1e-12)
    B = B / (B.norm(dim=0, keepdim=True) + 1e-12)
    if A.shape[1] > max_feat:
        f = torch.arange(0, A.shape[1], max(1, A.shape[1] // max_feat))[:max_feat]
        A, B = A[:, f], B[:, f]
    An = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    Bn = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    S = (An @ Bn.T).numpy()
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


rows = []
print(f"{'class':<5}{'idx->idx':>10}{'stride':>9}  pair", flush=True)
print("-" * 78, flush=True)
for cls, kind, a, b in PAIRS:
    try:
        ma, mb = AutoModel.from_pretrained(a).eval(), AutoModel.from_pretrained(b).eval()
        ua, ub = ups(ma), ups(mb)
        La, Lb = len(ua), len(ub)
        # mapping 1: index to index (what the earlier run did)
        v1 = [x for i in range(min(La, Lb)) if (x := lap(ua[i], ub[i])) is not None]
        # mapping 2: child layer i -> parent layer i*stride (distil init takes every k-th layer)
        stride = max(1, round(La / Lb)) if Lb else 1
        v2 = [x for i in range(Lb)
              if i * stride < La and (x := lap(ua[i * stride], ub[i])) is not None]
        del ma, mb
    except Exception as e:
        print(f"{cls:<5}  FAILED {type(e).__name__}: {str(e)[:60]}  {a} | {b}", flush=True)
        continue
    m1 = float(np.mean(v1)) if v1 else None
    m2 = float(np.mean(v2)) if v2 else None
    rows.append(dict(cls=cls, kind=kind, a=a, b=b, layers_parent=La, layers_child=Lb,
                     stride=stride, lap_index=m1, lap_stride=m2,
                     n_index=len(v1), n_stride=len(v2)))
    f = lambda x: float('nan') if x is None else x
    print(f"{cls:<5}{f(m1):>10.4f}{f(m2):>9.4f}  {a.split('/')[-1]} | {b.split('/')[-1]}"
          f"   (parent {La}L, child {Lb}L, stride {stride})", flush=True)

json.dump(rows, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT.name}", flush=True)

neg_ceiling = 0.2679   # highest same-recipe negative in the same-depth arm
print(f"\nAgainst the same-depth negative ceiling {neg_ceiling}:")
for r in rows:
    if r["cls"] != "P+":
        continue
    for tag, v in (("index->index", r["lap_index"]), ("stride-matched", r["lap_stride"])):
        if v is None: continue
        print(f"  {r['b'].split('/')[-1]:<24}{tag:<16}{v:.4f}  "
              f"{'ABOVE (separates)' if v > neg_ceiling else 'below (fails)'}")
print("DONE-DISTILL", flush=True)
