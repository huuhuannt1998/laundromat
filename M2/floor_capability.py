"""Measure the capability cost of the all-signal floor construction (reviewer C-5).

Section VI-E exhibits a zero-compute construction whose identity score (0.3255) sits below
the evasion bar (0.5298). A reviewer's sharpest objection is: 'you say no zero-compute
transform reaches the bar, then you exhibit one below it.' The answer is that the floor
construction is a diagnostic, not an attack -- it is not capability preserving. The paper
asserts this in passing but never measures it. This script measures it.

Reports max|delta logit| and relative perplexity change against the predeclared M-5
capability requirement (max|delta logit| <= 1e-2).
"""
import os, json, copy, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

MODEL = "HuggingFaceTB/SmolLM2-135M"
RES = pathlib.Path(str(_ROOT) + "/M2/results")

def is_norm(n): return ("norm" in n.lower()) or ("ln_" in n.lower())
def is_emb(n):  return any(k in n.lower() for k in ("embed","wte","wpe","lm_head","shared"))
def is_proj(n,pm): return pm.ndim==2 and not is_emb(n) and not is_norm(n)

def rand_rows(W_, g):
    rows=W_.float(); rnd=torch.randn(rows.shape, generator=g)
    nrm=rows.norm(dim=1,keepdim=True)
    rnd=rnd/(rnd.norm(dim=1,keepdim=True)+1e-12)*nrm
    return rnd.to(W_.dtype)

def apply_all(m,g):
    e=m.get_input_embeddings().weight; e.data=rand_rows(e.data,g)          # EAS
    for n,pm in m.named_parameters():
        if is_proj(n,pm): pm.data=rand_rows(pm.data,g)                     # WVC
    e=m.get_input_embeddings().weight                                       # END
    r=e.data.float(); e.data=(r/(r.norm(dim=1,keepdim=True)+1e-12)).to(e.dtype)
    seen={}
    for n,pm in m.named_parameters():                                       # LEP
        if pm.ndim!=2 or is_emb(n): continue
        li=n.split(".")[2] if len(n.split("."))>2 else "0"
        seen.setdefault(li, 0.25+1.75*torch.rand(1,generator=g).item())
        pm.data=(pm.data.float()*seen[li]).to(pm.dtype)
    for n,pm in m.named_parameters():                                       # NLF
        if is_norm(n): pm.data=(torch.rand(pm.shape,generator=g)*2.0).to(pm.dtype)

tok=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForCausalLM.from_pretrained(MODEL).eval()
TEXT=("The transformer architecture has become the dominant approach for sequence modelling. "
      "Attention allows the model to weigh distant tokens. Provenance verification asks whether "
      "one set of weights was derived from another, which matters for supply-chain assurance.")
ids=tok(TEXT, return_tensors="pt").input_ids

def ppl_and_logits(m):
    out=m(ids, labels=ids)
    return float(torch.exp(out.loss)), out.logits

ppl_ref, lg_ref = ppl_and_logits(base)
m=copy.deepcopy(base); g=torch.Generator().manual_seed(101); apply_all(m,g)
ppl_new, lg_new = ppl_and_logits(m)
mad=float((lg_new-lg_ref).abs().max())
rec=dict(arm="ALL", model=MODEL, n_tokens=int(ids.numel()),
         ppl_ref=ppl_ref, ppl_new=ppl_new,
         rel_ppl_delta=(ppl_new-ppl_ref)/ppl_ref,
         max_abs_logit_delta=mad,
         capability_threshold=1e-2,
         capability_preserving=bool(mad<=1e-2))
(RES/"floor_capability.jsonl").write_text(json.dumps(rec)+"\n")
print(json.dumps(rec, indent=2), flush=True)
print(f"\nppl {ppl_ref:.2f} -> {ppl_new:.2f}  ({rec['rel_ppl_delta']*100:+.1f}%)", flush=True)
print(f"max|dlogit| {mad:.4g} vs requirement <= 1e-2 -> capability preserving: {rec['capability_preserving']}", flush=True)
print("DONE-FLOORCAP", flush=True)
