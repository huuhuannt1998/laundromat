"""Q2 — test my own claim that untied-embedding models admit an EXACT per-row embedding rescaling.

Reasoning that now makes me doubt it: in a residual transformer the UNNORMALIZED embedding enters
the residual stream (h = x + attn(norm(x))), so scaling x is not absorbed by the norm -- the residual
term scales too. The exception would be an architecture with an embedding-normalizing layer BEFORE
the residual stream (BLOOM has word_embeddings_layernorm) AND an untied output head.
Predict: NOT exact for pythia (untied, no embedding LN) and NOT exact for bloom (has embedding LN, but tied).
"""
import os,sys,copy,json,torch,pathlib
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM, AutoTokenizer
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
PROBES=["The capital of France is Paris, and the capital of Germany is",
        "In 1969, the first humans landed on the surface of the",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
        "Photosynthesis converts light energy into chemical energy stored in"]*12

def per_row_scale(m, log_sigma=0.5, seed=0):
    e=m.get_input_embeddings().weight
    g=torch.Generator().manual_seed(seed)
    c=torch.exp(torch.randn(e.shape[0],1,generator=g)*log_sigma)
    e.copy_((e.float()*c).to(e.dtype)); return {"t":"per_row_scale","log_sigma":log_sigma}
def global_scale(m, c=2.0):
    e=m.get_input_embeddings().weight
    e.copy_((e.float()*c).to(e.dtype)); return {"t":"global_scale","c":c}

for MODEL in ["EleutherAI/pythia-160m","bigscience/bloom-560m","HuggingFaceTB/SmolLM2-135M"]:
    tok=AutoTokenizer.from_pretrained(MODEL)
    ref=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).eval()
    cfg=ref.config
    has_emb_ln = any("word_embeddings_layernorm" in n or "embed_layer_norm" in n
                     for n,_ in ref.named_modules())
    print(f"\n{MODEL}  tie={cfg.tie_word_embeddings}  embedding_layernorm={has_emb_ln}")
    for name,fn in [("per_row_scale(0.5)",lambda m: per_row_scale(m,0.5,3)),
                    ("global_scale(x2)",  lambda m: global_scale(m,2.0))]:
        m=copy.deepcopy(ref); fn(m)
        r=LD.verify_exact(ref,m,tok,PROBES)
        tag="EXACT" if r["exact_within_tolerance"] else "NOT EXACT"
        print(f"   {name:<20} max|dlogit|={r['max_abs_logit_delta']:.4e}  "
              f"ppl {r['ppl_ref']:.4f} -> {r['ppl_new']:.4f}  rel={r['rel_ppl_delta']:.3e}   {tag}")
        del m
    del ref
