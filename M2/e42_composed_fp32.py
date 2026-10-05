"""E42 -- the composed arm's exactness, measured in float32 and in the checkpoint's own dtype.

Review round 7 (N5). M2/results/m2t_composed.jsonl records max|dlogit| = 0.4375 for the composed
arm M2t(0.0) o X1a, and the same 0.4375 for X1a alone, while the E15 transform matrix records X1a
on the same pair at 1.8e-4. The design tolerance (M-4) is defined in float32:
max|dlogit| <= 1e-2. M2/fp32_verify.py was written to re-run that check in float32 but cannot
run as committed: it (and m2t_attack.py, which it imports) reads _ROOT before defining it, and no
output of it is on record. Here _ROOT is supplied through builtins so both import unchanged.

This repeats fp32_verify.py's procedure unchanged (same child, same build_perm vocabulary
permutation at overlap 0.0, same X1a seed 11 at fraction 1.0, same comparison of permuted logits)
twice: with the model cast to float32, and in the dtype the checkpoint loads in by default, which
is how m2t_composed.py loaded it. Probes: m2t_composed.py's eight, and the full PROBES list.

Writes a NEW file, M2/results/e42_composed_fp32.json.
"""
import os, sys, json, glob, pathlib, copy
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
sys.path.insert(0, str(ROOT / "M2" / "transforms")); sys.path.insert(0, str(ROOT / "M2"))
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import builtins
builtins._ROOT = ROOT          # m2t_attack.py, like fp32_verify.py, reads _ROOT before defining it
import laundry as LD
from m2t_attack import build_perm, CHILD, PROBES
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(CHILD)
tj = json.load(open(sorted(glob.glob(str(pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
     / ("models--" + CHILD.replace("/", "--")) / "snapshots" / "*" / "tokenizer.json")))[0]))
fixed = {a["id"] for a in tj.get("added_tokens", [])}

def check(base, probes):
    V = base.get_input_embeddings().weight.shape[0]
    out = {}
    for name, ov, x1a in (("M2t(0.0)", 0.0, False), ("X1a", None, True), ("M2t(0.0) o X1a", 0.0, True)):
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
        for s in probes:
            ids = tok(s, return_tensors="pt").input_ids
            old = base(ids).logits[0, -1].float()
            new = m(torch.as_tensor(perm[ids.numpy()])).logits[0, -1].float()
            dev.append((new[torch.as_tensor(perm)] - old).abs().max().item())
        out[name] = dict(max_abs_logit_delta=max(dev), within_1e_2=max(dev) <= 1e-2, n_probes=len(probes))
        del m
    return out

native = AutoModelForCausalLM.from_pretrained(CHILD).eval()
native_dtype = str(next(native.parameters()).dtype)
res = dict(child=CHILD, tolerance=1e-2, config_dtype=str(getattr(native.config, "dtype", None)),
           native_load_dtype=native_dtype)
res["native"] = {"probes8": check(native, PROBES[:8]), "all": check(native, PROBES)}
fp32 = copy.deepcopy(native).float(); del native
res["float32"] = {"probes8": check(fp32, PROBES[:8]), "all": check(fp32, PROBES)}
(ROOT / "M2" / "results" / "e42_composed_fp32.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
