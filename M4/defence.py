"""M4 core experiment — can a permutation-invariant weight signature be BOTH
laundering-resistant AND sound?

M2 established the duality: MPK's EAS resists laundering because it measures a stable
consequence of the training RECIPE, and false-positives on independent same-recipe models
for that same reason. WVC is the converse -- it is the only signal that tracks weight
inheritance (0.10 on independent same-recipe, 0.99 on a fine-tune), but it is the signal
most easily moved by a FREE permutation (0.997 -> 0.245 under X1a o X1b o X2).

So the defence question is whether some statistic is simultaneously:
  (a) SOUND    -- high on genuine derivation, LOW on independent-same-recipe pairs, and
  (b) ROBUST   -- unmoved by free weight-space permutation.
Singular values are invariant to row and column permutation (permutation matrices are
orthogonal) while remaining value-dependent, so a spectral signature is the natural candidate.
This is also GhostSpec's mechanism, so the same run tests whether the PUBLISHED defence
inherits MPK's recipe-convergence unsoundness.
"""
import os,sys,json,copy,math,pathlib,torch,numpy as np
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
from transformers import AutoModelForCausalLM
import laundry as LD

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)
RES=pathlib.Path(str(_ROOT) + "/M4"); RES.mkdir(exist_ok=True)
TOPK=64

def layer_mats(model):
    """Canonical per-layer matrices, architecture-agnostic."""
    out=[]
    for n,p in model.named_parameters():
        if p.ndim!=2: continue
        if any(t in n for t in ("embed","lm_head","wte","wpe","word_embeddings")): continue
        out.append((n,p))
    return out

def spectral_fp(model):
    """Per-matrix top-k singular values, L2-normalised. Permutation-invariant by construction."""
    sig=[]
    for n,p in layer_mats(model):
        w=p.float()
        if min(w.shape)<8: continue
        s=torch.linalg.svdvals(w)[:TOPK]
        s=s/ (s.norm()+1e-12)
        sig.append(s.numpy())
    return sig

def raw_fp(model):
    """MPK-style positional subsample: deterministic stride over raw values."""
    sig=[]
    for n,p in layer_mats(model):
        w=p.float().flatten()
        if w.numel()<64: continue
        idx=torch.arange(0,w.numel(),max(1,w.numel()//4096))[:4096]
        sig.append(w[idx].numpy())
    return sig

def resample(v,L):
    x=np.linspace(0,1,len(v)); return np.interp(np.linspace(0,1,L),x,v)

def compare_fp(a,b):
    """Mean cosine over aligned matrices; depth mismatch handled by resampling the layer axis."""
    L=min(len(a),len(b))
    if L==0: return float("nan")
    ai=[a[int(i*len(a)/L)] for i in range(L)]; bi=[b[int(i*len(b)/L)] for i in range(L)]
    cs=[]
    for x,y in zip(ai,bi):
        n=min(len(x),len(y)); x2,y2=resample(x,n),resample(y,n)
        d=(np.linalg.norm(x2)*np.linalg.norm(y2))
        if d>0: cs.append(float(np.dot(x2,y2)/d))
    return float(np.mean(cs)) if cs else float("nan")

def load(mid, launder=False):
    m=AutoModelForCausalLM.from_pretrained(mid).eval()
    if launder:
        LD.X1a_mlp_permute(m,1.0,11); LD.X1b_head_permute(m,11); LD.X2_norm_scale(m,0.5,11)
    return m

PAIRS=[
 ("IDENTITY","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M",False),
 ("P+ derived","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",False),
 ("P+ derived","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",False),
 ("P~ SAME-RECIPE INDEP","EleutherAI/pythia-160m","EleutherAI/pythia-160m-deduped",False),
 ("P- unrelated","HuggingFaceTB/SmolLM2-135M","EleutherAI/pythia-160m",False),
 ("P- unrelated","gpt2","bigscience/bloom-560m",False),
 ("LAUNDERED (X1a.X1b.X2)","HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct",True),
 ("LAUNDERED (X1a.X1b.X2)","Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",True),
]
rows=[]
print(f"{'class':<24}{'pair':<52}{'SVD-spec':>10}{'raw-pos':>10}")
print("-"*98)
for cls,a,b,launder in PAIRS:
    ma=load(a); mb=load(b,launder)
    sa,sb=spectral_fp(ma),spectral_fp(mb)
    ra,rb=raw_fp(ma),raw_fp(mb)
    sv=compare_fp(sa,sb); rw=compare_fp(ra,rb)
    rows.append({"class":cls,"a":a,"b":b,"laundered":launder,"svd_spectral":sv,"raw_positional":rw})
    print(f"{cls:<24}{(a.split('/')[-1]+' | '+b.split('/')[-1]):<52}{sv:>10.4f}{rw:>10.4f}")
    del ma,mb
(RES/"defence_signals.json").write_text(json.dumps(rows,indent=1))
print("\nDONE-DEFENCE")
