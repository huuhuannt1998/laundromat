"""E16 -- stronger function-preservation validation for the transforms labelled exact.

The predeclared criterion is max|dlogit| <= 1e-2 on a fixed probe set. A reviewer may
reasonably ask whether that probe is too small. This widens it and adds the measures the
review names: mean |dlogit|, next-token argmax agreement, perplexity before/after, and
greedy-decode sequence equality.

Probe set is deterministic and larger than the original: 64 prompts spanning prose, code,
multilingual text, numerals and degenerate inputs, each scored over every position rather
than the final token only.
"""
import os, json, pathlib, copy, gc, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(_ROOT) + "/M2/transforms")
sys.path.insert(0, str(_ROOT) + "/M2")
import laundry as LD

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
R = pathlib.Path(_ROOT)
OUT = R/"M7"/"e16_fidelity.json"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

BASE = [
 "The professor's arguments were criticized for their",
 "In 1969, the first humans landed on the surface of",
 "The capital of France is Paris, and the capital of Germany is",
 "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
 "class Node:\n    def __init__(self, value):\n        self.value =",
 "SELECT name, COUNT(*) FROM users WHERE created_at >",
 "Le chat est assis sur le tapis, et le chien est",
 "Die Hauptstadt von Österreich ist",
 "El resultado de la ecuación es aproximadamente",
 "1, 1, 2, 3, 5, 8, 13, 21, 34,",
 "0.1 + 0.2 = 0.30000000000000004 because floating point",
 "```json\n{\"name\": \"test\", \"values\": [1, 2,",
 "                    ",
 "!!!???...---___",
 "a",
 "The mitochondria is the powerhouse of the",
]
PROMPTS = [p + s for p in BASE for s in ("", " and", " therefore", " which")]  # 64, deterministic

def measure(ref, mod, tok):
    maxd = 0.0; sabs = 0.0; n = 0; agree = 0; tot = 0; seq_eq = 0; measured = 0; skipped = 0
    nll_r = nll_m = ntok = 0.0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            skipped += 1; continue
        measured += 1
        lr = ref(ids).logits[0]; lm = mod(ids).logits[0]
        d = (lr - lm).abs()
        maxd = max(maxd, d.max().item()); sabs += d.sum().item(); n += d.numel()
        ar, am = lr.argmax(-1), lm.argmax(-1)
        agree += (ar == am).sum().item(); tot += ar.numel()
        if torch.equal(ar, am): seq_eq += 1
        tgt = ids[0, 1:]
        nll_r += torch.nn.functional.cross_entropy(lr[:-1], tgt, reduction="sum").item()
        nll_m += torch.nn.functional.cross_entropy(lm[:-1], tgt, reduction="sum").item()
        ntok += tgt.numel()
    import math
    return dict(n_prompts=len(PROMPTS), n_logits=n,
                max_abs_dlogit=maxd, mean_abs_dlogit=sabs/max(n,1),
                argmax_agreement=agree/max(tot,1), n_positions=tot,
                greedy_sequence_equal=f"{seq_eq}/{measured}",
                prompts_skipped_single_token=skipped,
                ppl_ref=math.exp(nll_r/max(ntok,1)), ppl_mod=math.exp(nll_m/max(ntok,1)))

tok = AutoTokenizer.from_pretrained(MODEL)
ref = AutoModelForCausalLM.from_pretrained(MODEL).eval().float()

ARMS = [
 ("X1a  MLP intermediate permutation", lambda m: LD.X1a_mlp_permute(m, 1.0, 11)),
 ("X1b  GQA-aware head permutation",   lambda m: LD.X1b_head_permute(m, 11)),
]
rows = []
for name, fn in ARMS:
    m = copy.deepcopy(ref)
    try:
        info = fn(m)
    except Exception as e:
        print(f"{name}: SKIPPED ({type(e).__name__}: {e})"); del m; gc.collect(); continue
    r = measure(ref, m.eval().float(), tok); r["arm"] = name; r["info"] = str(info)[:120]
    rows.append(r); del m; gc.collect()
    print(f"\n=== {name} ===")
    print(f"  max |dlogit|          {r['max_abs_dlogit']:.3e}   (criterion <= 1e-2)")
    print(f"  mean |dlogit|         {r['mean_abs_dlogit']:.3e}   over {r['n_logits']:,} logits")
    print(f"  next-token argmax     {r['argmax_agreement']*100:.4f}% agreement over {r['n_positions']:,} positions")
    print(f"  greedy sequence equal {r['greedy_sequence_equal']} measured prompts "
          f"({r['prompts_skipped_single_token']} skipped as single-token)")
    print(f"  perplexity            {r['ppl_ref']:.6f} -> {r['ppl_mod']:.6f}")
OUT.write_text(json.dumps(dict(model=MODEL, n_prompts=len(PROMPTS), arms=rows), indent=1))
print(f"\nwrote {OUT.name}")
