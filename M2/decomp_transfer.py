"""Does the linear decomposition transfer across scale and architecture?

At SmolLM2-135M the fraction sweep established that MPK is a LINEAR model in its
signals: sigma_id's response to X1a is exactly w_wvc * delta(WVC), and composing
an EAS-attack (M2t) with a WVC-attack (X1a) is additive to 4 decimals
(predicted 0.8864, observed 0.8864).

That is the analytic backbone of Exit 3, so it must be shown to hold at more than
one point.  This replaces the design's Stage 4 ("<=20 frozen compositions at
1.4B"), which tests breadth; additivity-transfer tests the claim that actually
carries the argument, at a fraction of the cost.

Four points per model: baseline, M2t (EAS term), X1a (WVC term), M2t o X1a.
Predicted composed = base - (base - M2t) - (base - X1a).  Report the residual.
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

PAIRS = [("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M"),
         ("Qwen/Qwen2.5-0.5B",          "Qwen/Qwen2.5-0.5B-Instruct",          "Qwen2.5-0.5B"),
         ("EleutherAI/pythia-1.4b",     "EleutherAI/pythia-1.4b-deduped",      "pythia-1.4b")]


def vocab_of(child):
    g = sorted(glob.glob(str(pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub")) /
               ("models--" + child.replace("/", "--")) / "snapshots" / "*" / "tokenizer.json")))
    return json.load(open(g[0])) if g else None


def one(parent, child, name, do_m2t, do_x1a):
    out = WORK / f"dt_{name}".replace("/", "_").replace(" ", "")
    if out.exists(): shutil.rmtree(out)
    tok = AutoTokenizer.from_pretrained(child)
    m = AutoModelForCausalLM.from_pretrained(child).eval()
    V = m.get_input_embeddings().weight.shape[0]
    tj = vocab_of(child)
    perm = np.arange(V)
    if do_m2t and tj is not None:
        fixed = {a["id"] for a in tj.get("added_tokens", [])}
        perm = build_perm(V, fixed, 0.0)
        inv = np.argsort(perm)
        m.get_input_embeddings().weight.data = m.get_input_embeddings().weight.data[torch.as_tensor(inv)].clone()
        if not getattr(m.config, "tie_word_embeddings", False) and \
           hasattr(m, "lm_head") and m.lm_head.weight.shape[0] == V:
            m.lm_head.weight.data = m.lm_head.weight.data[torch.as_tensor(inv)].clone()
    if do_x1a:
        LD.X1a_mlp_permute(m, 1.0, 11)
    m.save_pretrained(out); tok.save_pretrained(out)
    if do_m2t and tj is not None:
        tp = out / "tokenizer.json"
        t2 = json.load(open(tp))
        t2["model"]["vocab"] = {t: int(perm[i]) for t, i in tj["model"]["vocab"].items()}
        t2["truncation"] = None; t2["padding"] = None
        json.dump(t2, open(tp, "w"))
    del m; gc.collect()
    r = mpk(["compare", parent, str(out)])
    shutil.rmtree(out, ignore_errors=True)
    sc, sg = r.get("scores", {}), r.get("signals", {})
    return sc.get("identity_score"), sg


fh = (RES / "decomp_transfer.jsonl").open("a")
for parent, child, tag in PAIRS:
    print(f"\n=== {tag}   ({parent} -> {child}) ===", flush=True)
    try:
        vals = {}
        for nm, m2t, x1a in (("base", False, False), ("M2t", True, False),
                             ("X1a", False, True), ("M2t o X1a", True, True)):
            sid, sg = one(parent, child, f"{tag}-{nm}", m2t, x1a)
            vals[nm] = sid
            nf = lambda x: float("nan") if x is None else x
            print(f"  {nm:<12} sig_id={nf(sid):.4f}  EAS={nf(sg.get('eas')):.4f} "
                  f"WVC={nf(sg.get('wvc')):.4f}", flush=True)
        if all(v is not None for v in vals.values()):
            pred = vals["base"] - (vals["base"] - vals["M2t"]) - (vals["base"] - vals["X1a"])
            obs = vals["M2t o X1a"]
            print(f"  ADDITIVITY: predicted {pred:.4f}   observed {obs:.4f}   "
                  f"residual {obs-pred:+.5f}   {'HOLDS' if abs(obs-pred)<0.01 else 'FAILS'}", flush=True)
            fh.write(json.dumps(dict(model=tag, parent=parent, child=child, values=vals,
                                     predicted=pred, observed=obs, residual=obs-pred)) + "\n")
            fh.flush()
    except Exception as e:
        print(f"  {tag} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    gc.collect()
fh.close(); print("\nDONE-DECOMPTRANSFER", flush=True)
