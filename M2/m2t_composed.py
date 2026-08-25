"""M2t o X1a -- the composition the frontier says is required.

The fraction-sweep decomposition (M2/frac_sweep.py) established:
  * EAS is bit-identically invariant to MLP permutation  -> X1a cannot touch it
  * 100% of sigma_id's response to X1a is the WVC term   -> X1a kills WVC only
  * destroying WVC alone leaves sigma_id >= 0.7898       -> no evasion
  * destroying EAS alone leaves sigma_id >= 0.6393       -> no evasion
  * destroying BOTH puts sigma_id at ~0.4299             -> BELOW the 0.5298 bar

M2t attacks EAS by index-reordering the embedding; X1a attacks WVC by permuting
the MLP.  They are orthogonal by construction, both exact, both kappa = 0.
This is the composition that decides gate G1.
"""
import os, sys, json, shutil, copy, subprocess, pathlib, glob, statistics as st
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(_ROOT) + "/M2/transforms")
sys.path.insert(0, str(_ROOT) + "/M2")
import laundry as LD
from m2t_attack import mpk, build_perm, RES, WORK, PROBES, PARENT, CHILD, NULL, PI_CUT

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)


def run(name, overlap, do_x1a, meta):
    out = WORK / name.replace(".", "_").replace("/", "_")
    if out.exists(): shutil.rmtree(out)
    tok = AutoTokenizer.from_pretrained(CHILD)
    model = AutoModelForCausalLM.from_pretrained(CHILD).eval()
    ref = copy.deepcopy(model)
    applied = []
    tj_path = sorted(glob.glob(str(pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub")) /
                     ("models--" + CHILD.replace("/", "--")) / "snapshots" / "*" / "tokenizer.json")))[0]
    tkjson = json.load(open(tj_path)); vocab = tkjson["model"]["vocab"]
    fixed = {a["id"] for a in tkjson.get("added_tokens", [])}
    V = model.get_input_embeddings().weight.shape[0]
    perm = build_perm(V, fixed, overlap) if overlap is not None else np.arange(V)
    if overlap is not None:
        inv = np.argsort(perm)
        e = model.get_input_embeddings().weight.data
        model.get_input_embeddings().weight.data = e[torch.as_tensor(inv)].clone()
        if not getattr(model.config, "tie_word_embeddings", False) and \
           hasattr(model, "lm_head") and model.lm_head.weight.shape[0] == V:
            model.lm_head.weight.data = model.lm_head.weight.data[torch.as_tensor(inv)].clone()
        applied.append(f"M2t(overlap={overlap})")
    if do_x1a:
        applied.append(LD.X1a_mlp_permute(model, 1.0, 11))
    dev = []
    for s in PROBES[:8]:
        ids = tok(s, return_tensors="pt").input_ids
        old = ref(ids).logits[0, -1].float()
        new = model(torch.as_tensor(perm[ids.numpy()])).logits[0, -1].float()
        dev.append((new[torch.as_tensor(perm)] - old).abs().max().item())
    maxdev = max(dev)
    model.save_pretrained(out); tok.save_pretrained(out)
    tp = out / "tokenizer.json"; tj = json.load(open(tp))
    tj["model"]["vocab"] = {t: int(perm[i]) for t, i in vocab.items()}
    tj["truncation"] = None; tj["padding"] = None
    json.dump(tj, open(tp, "w"))
    if meta:
        applied.append(LD.M1t_metadata(out, mode="both"))
        applied.append(LD.M3t_tokenizer_config(out))
    del model, ref
    cmp_ = mpk(["compare", PARENT, str(out)])
    scan = mpk(["scan", str(out), "--top-k", "300", "--threshold", "0.0"])
    sc, sg = cmp_.get("scores", {}), cmp_.get("signals", {})
    ms = scan.get("matches", []); ps = PARENT.split("/")[-1]
    idx = [i for i, x in enumerate(ms, 1) if x["model_id"] == ps]
    rank = idx[0] if idx else None
    sp = sc.get("pipeline_score")
    pi = 100.0 * sum(1 for s in NULL if s <= sp) / len(NULL) if sp is not None else None
    E = 1 if (rank is not None and rank > 5 and pi is not None and pi <= 75) else 0
    kind = "SUCCESS" if (E and maxdev <= 1e-2) else ("DESTRUCTIVE" if E else "RESISTED")
    shutil.rmtree(out, ignore_errors=True)
    print(f"{name:<34} tier={sc.get('mfi_tier')} sig={sc.get('pipeline_score')} "
          f"sig_id={sc.get('identity_score')} EAS={sg.get('eas')} END={sg.get('end')} "
          f"WVC={sg.get('wvc')} rank={rank} pi={pi} E={E} exact={maxdev<=1e-2} "
          f"({maxdev:.1e}) {kind}  {sc.get('provenance_decision')}", flush=True)
    return dict(name=name, applied=applied, overlap=overlap, x1a=do_x1a, meta=meta,
                max_abs_logit_delta=maxdev, exact=bool(maxdev <= 1e-2), scores=sc,
                signals=sg, parent_rank=rank, pi=pi, E=E, kind=kind, kappa=0.0)


CASES = [("M2t(0.0) alone",            0.0,  False, False),
         ("X1a alone",                 None, True,  False),
         ("M2t(0.0) o X1a",            0.0,  True,  False),
         ("M2t(0.0) o X1a o M1t/M3t",  0.0,  True,  True)]
fh = (RES / "m2t_composed.jsonl").open("w")
print(f"pi* cut = {PI_CUT:.4f}\n", flush=True)
for n, ov, x, m in CASES:
    try:
        fh.write(json.dumps(run(n, ov, x, m), default=str) + "\n"); fh.flush()
    except Exception as e:
        print(f"{n}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
fh.close(); print("DONE-M2TCOMP", flush=True)
