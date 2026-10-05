"""Finish the pythia decomposition-transfer arm (GPT-NeoX, third architecture family).

The first attempt died in X1a: laundry.py assumed a gated SwiGLU MLP and the
Llama-style .model.layers accessor.  Both are now generalised.  This script FIRST
verifies the non-gated permutation is genuinely exact on a small GPT-NeoX model --
a permutation that forgot dense_h_to_4h.bias would look plausible and be wrong --
and only then runs the 1.4b arm.
"""
import os, sys, json, shutil, copy, glob, pathlib, gc
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(_ROOT) + "/M2/transforms")
sys.path.insert(0, str(_ROOT) + "/M2")
import laundry as LD
from m2t_attack import mpk, build_perm, WORK, RES

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

PROBES = ["The professor's arguments were criticized for their",
          "In 1969, the first humans landed on the surface of",
          "The capital of France is Paris, and the capital of Germany is"]

# ---- step 1: exactness of the non-gated X1a on GPT-NeoX
print("=== X1a exactness on GPT-NeoX (pythia-160m, fp32) ===", flush=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
ref = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").eval().float()
m = copy.deepcopy(ref)
info = LD.X1a_mlp_permute(m, 1.0, 11)
dev = []
for s in PROBES:
    ids = tok(s, return_tensors="pt").input_ids
    dev.append((m(ids).logits[0, -1] - ref(ids).logits[0, -1]).abs().max().item())
d = max(dev)
print(f"  layers permuted: {info['layers']}   max|dlogit| = {d:.3e}   "
      f"{'EXACT' if d <= 1e-2 else 'NOT EXACT -- do not proceed'}", flush=True)
del m, ref; gc.collect()
if d > 1e-2:
    sys.exit("X1a is not exact on GPT-NeoX; aborting rather than reporting a broken arm.")

# ---- step 2: the 1.4b arm
PARENT, CHILD, TAG = "EleutherAI/pythia-1.4b", "EleutherAI/pythia-1.4b-deduped", "pythia-1.4b"
print(f"\n=== {TAG}  ({PARENT} -> {CHILD}) ===", flush=True)
tj = json.load(open(sorted(glob.glob(str(pathlib.Path(os.path.expanduser(
    "~/.cache/huggingface/hub")) / ("models--" + CHILD.replace("/", "--")) /
    "snapshots" / "*" / "tokenizer.json")))[0]))


def one(name, do_m2t, do_x1a):
    out = WORK / f"pa_{name}".replace(" ", "").replace("/", "_")
    if out.exists(): shutil.rmtree(out)
    t = AutoTokenizer.from_pretrained(CHILD)
    mm = AutoModelForCausalLM.from_pretrained(CHILD).eval()
    V = mm.get_input_embeddings().weight.shape[0]
    perm = np.arange(V)
    if do_m2t:
        fixed = {a["id"] for a in tj.get("added_tokens", [])}
        perm = build_perm(V, fixed, 0.0)
        inv = np.argsort(perm)
        mm.get_input_embeddings().weight.data = mm.get_input_embeddings().weight.data[torch.as_tensor(inv)].clone()
        if not getattr(mm.config, "tie_word_embeddings", False):
            head = getattr(mm, "embed_out", None) or getattr(mm, "lm_head", None)
            if head is not None and head.weight.shape[0] == V:
                head.weight.data = head.weight.data[torch.as_tensor(inv)].clone()
    if do_x1a:
        LD.X1a_mlp_permute(mm, 1.0, 11)
    mm.save_pretrained(out); t.save_pretrained(out)
    if do_m2t:
        tp = out / "tokenizer.json"; t2 = json.load(open(tp))
        t2["model"]["vocab"] = {k: int(perm[i]) for k, i in tj["model"]["vocab"].items()}
        t2["truncation"] = None; t2["padding"] = None
        json.dump(t2, open(tp, "w"))
    del mm; gc.collect()
    r = mpk(["compare", PARENT, str(out)])
    shutil.rmtree(out, ignore_errors=True)
    sc, sg = r.get("scores", {}), r.get("signals", {})
    return sc.get("identity_score"), sg


vals = {}
for nm, a, b in (("base", False, False), ("M2t", True, False),
                 ("X1a", False, True), ("M2t o X1a", True, True)):
    sid, sg = one(nm, a, b); vals[nm] = sid
    nf = lambda x: float("nan") if x is None else x
    print(f"  {nm:<12} sig_id={nf(sid):.4f}  EAS={nf(sg.get('eas')):.4f} WVC={nf(sg.get('wvc')):.4f}", flush=True)
    gc.collect()
if all(v is not None for v in vals.values()):
    pred = vals["base"] - (vals["base"]-vals["M2t"]) - (vals["base"]-vals["X1a"])
    obs = vals["M2t o X1a"]
    print(f"  ADDITIVITY: predicted {pred:.4f}   observed {obs:.4f}   residual {obs-pred:+.5f}   "
          f"{'HOLDS' if abs(obs-pred) < 0.01 else 'FAILS'}", flush=True)
    with (RES / "decomp_transfer.jsonl").open("a") as f:
        f.write(json.dumps(dict(model=TAG, parent=PARENT, child=CHILD, values=vals,
                                predicted=pred, observed=obs, residual=obs-pred,
                                arch="GPTNeoX", note="non-gated MLP; X1a generalised")) + "\n")
print("DONE-PYTHIAARM", flush=True)
