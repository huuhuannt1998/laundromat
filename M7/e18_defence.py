"""E18 -- expand the alignment-defence evaluation and give it a family-disjoint holdout.

Three deficiencies the review names, addressed here:
  (a) 18 pairs is thin, and the classes are unevenly covered;
  (b) there is no same-family-independent (N-FAM) negative class at all, only
      same-recipe (P~) and unrelated (P-);
  (c) the holdout splits on PARAMETER COUNT, which put every distillation pair in
      development. A family-disjoint split -- develop on some parent families, test on
      entirely unseen ones -- is what actually measures generalisation.

Distillation pairs are stride-matched (child layer i against parent layer i*stride);
every other pair is matched index-to-index. Which rule was used is recorded per row.

Writes a NEW file. No existing result file is touched.
"""
import os, json, pathlib, gc, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
import numpy as np, torch
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

OUT = pathlib.Path(str(_ROOT) + "/M7/e18_defence.json")
B, R, G = "google-bert/bert-base-uncased", "FacebookAI/roberta-base", "openai-community/gpt2"
P = "EleutherAI/pythia-"

PAIRS = [
 # cls, kind, parent, child, parent-family tag
 ("P+","fine-tune", B,"textattack/bert-base-uncased-SST-2","bert"),
 ("P+","fine-tune", B,"textattack/bert-base-uncased-imdb","bert"),
 ("P+","fine-tune", B,"csarron/bert-base-uncased-squad-v1","bert"),
 ("P+","cont-pre",  B,"ProsusAI/finbert","bert"),
 ("P+","distill",   B,"distilbert/distilbert-base-uncased","bert"),
 ("P+","distill",   "google-bert/bert-base-cased","distilbert/distilbert-base-cased","bert-cased"),
 ("P+","distill",   "google-bert/bert-base-multilingual-cased",
                    "distilbert/distilbert-base-multilingual-cased","mbert"),
 ("P+","cont-pre",  R,"ehsanaghaei/SecureBERT","roberta"),
 ("P+","cont-pre",  R,"allenai/biomed_roberta_base","roberta"),
 ("P+","cont-pre",  R,"allenai/cs_roberta_base","roberta"),
 ("P+","cont-pre",  R,"allenai/news_roberta_base","roberta"),
 ("P+","cont-pre",  R,"allenai/reviews_roberta_base","roberta"),
 ("P+","cont-pre",  R,"cardiffnlp/twitter-roberta-base","roberta"),
 ("P+","distill",   R,"distilbert/distilroberta-base","roberta"),
 ("P+","distill",   G,"distilbert/distilgpt2","gpt2"),
 ("P+","fine-tune", "HuggingFaceTB/SmolLM2-135M","HuggingFaceTB/SmolLM2-135M-Instruct","smollm2"),
 ("P+","fine-tune", "Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct","qwen"),
 ("P+","multitask", "bigscience/bloom-560m","bigscience/bloomz-560m","bloom"),
 # independently trained, same recipe and size
 ("P~","same-recipe", P+"70m",  P+"70m-deduped","pythia"),
 ("P~","same-recipe", P+"160m", P+"160m-deduped","pythia"),
 ("P~","same-recipe", P+"410m", P+"410m-deduped","pythia"),
 ("P~","same-recipe", P+"1b",   P+"1b-deduped","pythia"),
 ("P~","same-recipe", P+"70m",  P+"70m-v0","pythia"),
 ("P~","same-recipe", P+"160m", P+"160m-v0","pythia"),
 ("P~","same-recipe", P+"410m", P+"410m-v0","pythia"),
 ("P~","same-recipe", P+"70m-deduped", P+"70m-deduped-v0","pythia"),
 # same architecture family, independently PRETRAINED (the class that was missing)
 ("N~","same-family", B,"nlpaueb/legal-bert-base-uncased","bert"),
 ("N~","same-family", B,"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract","bert"),
 # unrelated
 ("P-","unrelated",  B, R,"bert"),
 ("P-","unrelated",  "distilbert/distilbert-base-uncased","distilbert/distilroberta-base","bert"),
 ("P-","unrelated",  G, P+"160m","gpt2"),
]

def ups(m):
    return [p for n,p in m.named_parameters() if p.ndim==2 and any(
        k in n for k in ("up_proj.weight","dense_h_to_4h.weight","c_fc.weight",
                         "fc_in.weight","intermediate.dense.weight","ffn.lin1.weight"))]

def lap_colnorm(A,Bm,max_feat=1536):
    A,Bm=A.float(),Bm.float()
    if A.shape!=Bm.shape: return None
    A =A /(A.norm(dim=0,keepdim=True)+1e-12)
    Bm=Bm/(Bm.norm(dim=0,keepdim=True)+1e-12)
    if A.shape[1]>max_feat:
        f=torch.arange(0,A.shape[1],max(1,A.shape[1]//max_feat))[:max_feat]
        A,Bm=A[:,f],Bm[:,f]
    An=A /(A.norm(dim=1,keepdim=True)+1e-12)
    Bn=Bm/(Bm.norm(dim=1,keepdim=True)+1e-12)
    S=(An@Bn.T).numpy()
    r,c=linear_sum_assignment(-S)
    return float(S[r,c].mean())

done={}
if OUT.exists():
    for r in json.loads(OUT.read_text()): done[(r["a"],r["b"])]=r
rows=list(done.values())

print(f"{'cls':<4}{'kind':<12}{'lap':>9} {'match':<7} pair",flush=True)
print("-"*94,flush=True)
for cls,kind,a,b,fam in PAIRS:
    if (a,b) in done:
        r=done[(a,b)]
        print(f"{cls:<4}{kind:<12}{(r['lap'] or float('nan')):>9.4f} {r.get('match',''):<7} cached",flush=True)
        continue
    t0=time.time()
    try:
        ma=AutoModel.from_pretrained(a).eval(); mb=AutoModel.from_pretrained(b).eval()
        ua,ub=ups(ma),ups(mb)
        if kind=="distill" and len(ua)>len(ub) and len(ub)>0 and len(ua)%len(ub)==0:
            stride=len(ua)//len(ub); idx=[(i*stride,i) for i in range(len(ub))]; mrule=f"stride{stride}"
        else:
            idx=[(i,i) for i in range(min(len(ua),len(ub)))]; mrule="index"
        vals=[v for (i,j) in idx if (v:=lap_colnorm(ua[i],ub[j])) is not None]
        del ma,mb; gc.collect()
        lap=float(np.mean(vals)) if vals else None
    except Exception as e:
        print(f"{cls:<4}{kind:<12}{'FAIL':>9} {'-':<7} {a}|{b} ({type(e).__name__})",flush=True)
        rows.append(dict(cls=cls,kind=kind,a=a,b=b,family=fam,lap=None,
                         match=None,n_layers=0,error=type(e).__name__))
        OUT.write_text(json.dumps(rows,indent=1)); continue
    rows.append(dict(cls=cls,kind=kind,a=a,b=b,family=fam,lap=lap,match=mrule,
                     n_layers=len(vals),secs=round(time.time()-t0,1)))
    OUT.write_text(json.dumps(rows,indent=1))
    print(f"{cls:<4}{kind:<12}{(lap if lap is not None else float('nan')):>9.4f} {mrule:<7} "
          f"{a.split('/')[-1]} | {b.split('/')[-1]}",flush=True)
print(f"\nwrote {OUT.name} (n={len(rows)})",flush=True)
print("DONE-E18",flush=True)
