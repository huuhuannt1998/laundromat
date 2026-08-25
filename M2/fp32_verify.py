"""fp32 exactness re-verification for the M2t compositions.

The predeclared tolerance (design M-4) is specified IN fp32: max|delta logit| <= 1e-2.
m2t_composed.py compared in the model's native dtype, where permuted accumulation
order rounds differently -- that reported X1a as "exact=False (4.4e-01)", which
contradicts the gate arm's verified 6.5e-4.  This re-runs the fidelity check the
way the design specifies.  MPK scores are unaffected either way.
"""
import os, sys, json, glob, pathlib, copy
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(_ROOT) + "/M2/transforms")
sys.path.insert(0, str(_ROOT) + "/M2")
import laundry as LD
from m2t_attack import build_perm, CHILD, PROBES

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(CHILD)
tj = json.load(open(sorted(glob.glob(str(pathlib.Path(os.path.expanduser(
    "~/.cache/huggingface/hub")) / ("models--" + CHILD.replace("/", "--")) /
    "snapshots" / "*" / "tokenizer.json")))[0]))
fixed = {a["id"] for a in tj.get("added_tokens", [])}

base = AutoModelForCausalLM.from_pretrained(CHILD).eval().float()   # fp32 reference
V = base.get_input_embeddings().weight.shape[0]
print(f"reference dtype: fp32   vocab {V}   tolerance (design M-4): max|dlogit| <= 1e-2\n")
print(f"{'variant':<28}{'max|dlogit|':>14}{'exact?':>9}")
print("-" * 52)

for name, ov, x1a in (("M2t(0.0)", 0.0, False),
                      ("X1a", None, True),
                      ("M2t(0.0) o X1a", 0.0, True)):
    m = copy.deepcopy(base)
    perm = build_perm(V, fixed, ov) if ov is not None else np.arange(V)
    if ov is not None:
        inv = np.argsort(perm)
        m.get_input_embeddings().weight.data = m.get_input_embeddings().weight.data[torch.as_tensor(inv)].clone()
        if not getattr(m.config, "tie_word_embeddings", False) and \
           hasattr(m, "lm_head") and m.lm_head.weight.shape[0] == V:
            m.lm_head.weight.data = m.lm_head.weight.data[torch.as_tensor(inv)].clone()
    if x1a:
        LD.X1a_mlp_permute(m, 1.0, 11)
    dev = []
    for s in PROBES[:8]:
        ids = tok(s, return_tensors="pt").input_ids
        old = base(ids).logits[0, -1]
        new = m(torch.as_tensor(perm[ids.numpy()])).logits[0, -1]
        dev.append((new[torch.as_tensor(perm)] - old).abs().max().item())
    d = max(dev)
    print(f"{name:<28}{d:>14.3e}{str(d <= 1e-2):>9}")
    del m
print("\n(native-dtype figures in m2t_composed.log are NOT comparable to the")
print(" design tolerance, which is defined in fp32.)")
