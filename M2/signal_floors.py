"""Measure the FLOOR of every identity signal, and the true minimum attainable identity score.

WHY. The EAS arm of the O0 recovery showed the anchor signal bottoming out at 0.5013 rather
than 0 -- it is an affinely rescaled cosine. If the other four signals also have floors, then
"complete signal destruction" is not reachable, and the paper's evasion-bar extrapolation
(sigma_id -> 0.5298 at 85% of a from-scratch run) rests on a limit that cannot be attained.
That extrapolation is a claim in section VI and this experiment tests it directly.

CONSTRUCTION. For each signal, apply the most aggressive perturbation that targets it, and
read all five signals plus sigma_id. Then apply everything at once: the ALL row is the
empirical minimum attainable identity score for a model of this shape.

Reference is the model against itself, so an untransformed pair scores 1.0 on every signal.
"""
import os, json, shutil, copy, subprocess, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES  = pathlib.Path(str(_ROOT) + "/M2/results")
WORK = pathlib.Path(str(_ROOT) + "/M2/work"); WORK.mkdir(exist_ok=True)
MODEL = "HuggingFaceTB/SmolLM2-135M"
W = {"eas":0.36, "wvc":0.21, "end":0.19, "lep":0.16, "nlf":0.08}

def mpk(a, b):
    p = subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],
        cwd=REPO, capture_output=True, text=True, timeout=5400,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-300:]}

def is_norm(n):  return ("norm" in n.lower()) or ("ln_" in n.lower())
def is_emb(n):   return any(k in n.lower() for k in ("embed","wte","wpe","lm_head","shared"))
def is_proj(n,pm): return pm.ndim==2 and not is_emb(n) and not is_norm(n)

def rand_rows(W_, g, preserve_norm=True):
    rows = W_.float()
    rnd = torch.randn(rows.shape, generator=g)
    if preserve_norm:
        nrm = rows.norm(dim=1, keepdim=True)
        rnd = rnd/(rnd.norm(dim=1,keepdim=True)+1e-12)*nrm
    else:
        rnd = rnd * 0.02
    return rnd.to(W_.dtype)

def t_eas(m,g):                       # destroy embedding geometry, keep row norms
    e=m.get_input_embeddings().weight; e.data=rand_rows(e.data,g,True)
def t_wvc(m,g):                       # destroy projection-row correspondence, keep norms
    for n,pm in m.named_parameters():
        if is_proj(n,pm): pm.data=rand_rows(pm.data,g,True)
def t_end(m,g):                       # flatten the embedding NORM DISTRIBUTION
    e=m.get_input_embeddings().weight
    r=e.data.float(); r=r/(r.norm(dim=1,keepdim=True)+1e-12)*1.0
    e.data=r.to(e.dtype)
def t_lep(m,g):                       # rescale each layer differently -> new energy profile
    seen={}
    for n,pm in m.named_parameters():
        if pm.ndim!=2 or is_emb(n): continue
        li=n.split(".")[2] if len(n.split("."))>2 else "0"
        seen.setdefault(li, 0.25+1.75*torch.rand(1,generator=g).item())
        pm.data=(pm.data.float()*seen[li]).to(pm.dtype)
def t_nlf(m,g):                       # randomise norm-layer vectors
    for n,pm in m.named_parameters():
        if is_norm(n):
            pm.data=(torch.rand(pm.shape,generator=g)*2.0).to(pm.dtype)

ARMS=[("none",[]),("EAS",[t_eas]),("WVC",[t_wvc]),("END",[t_end]),
      ("LEP",[t_lep]),("NLF",[t_nlf]),("ALL",[t_eas,t_wvc,t_end,t_lep,t_nlf])]

tok=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForCausalLM.from_pretrained(MODEL).eval()
rows=[]
print(f"reference: {MODEL}", flush=True)
print(f"{'arm':>6}{'sigma_id':>10}{'EAS':>9}{'WVC':>9}{'END':>9}{'LEP':>9}{'NLF':>9}  verdict", flush=True)
print("-"*80, flush=True)
for name,fns in ARMS:
    g=torch.Generator().manual_seed(101)
    m=copy.deepcopy(base)
    for f in fns: f(m,g)
    out=WORK/f"floor_{name}"
    if out.exists(): shutil.rmtree(out)
    m.save_pretrained(out); tok.save_pretrained(out); del m
    r=mpk(MODEL,str(out)); shutil.rmtree(out,ignore_errors=True)
    sc,sg=r.get("scores",{}),r.get("signals",{})
    nf=lambda x: float("nan") if x is None else x
    rows.append(dict(arm=name, sigma_id=sc.get("identity_score"),
                     tier=sc.get("mfi_tier"), pipeline=sc.get("pipeline_score"),
                     verdict=sc.get("provenance_decision"),
                     signals={k:sg.get(k) for k in ("eas","wvc","end","lep","nlf")}))
    print(f"{name:>6}{nf(sc.get('identity_score')):>10.4f}{nf(sg.get('eas')):>9.4f}"
          f"{nf(sg.get('wvc')):>9.4f}{nf(sg.get('end')):>9.4f}{nf(sg.get('lep')):>9.4f}"
          f"{nf(sg.get('nlf')):>9.4f}  {sc.get('provenance_decision')}", flush=True)

(RES/"signal_floors.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
# predicted floor from per-signal minima vs measured ALL
mins={k:min([r["signals"][k] for r in rows if r["signals"][k] is not None]) for k in W}
pred=sum(W[k]*mins[k] for k in W)
allrow=[r for r in rows if r["arm"]=="ALL"][0]
print(f"\nper-signal floors: "+", ".join(f"{k}={mins[k]:.4f}" for k in W), flush=True)
print(f"predicted min sigma_id (sum w_i * floor_i) = {pred:.4f}", flush=True)
print(f"measured  min sigma_id (ALL arm)           = {allrow['sigma_id']}", flush=True)
print(f"evasion bar (null p75)                     = 0.5298", flush=True)
print("DONE-FLOORS", flush=True)
