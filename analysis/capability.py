"""M-5 capability battery — the predeclared instrument I kept deferring.

Every rho in this project so far is perplexity on a small custom corpus. The design specified
LAMBADA / ARC-Easy / HellaSwag subsets, and without them no capability-retention claim is
properly supported. CPU-feasible at <=500M with bounded subsets.

  LAMBADA   last-word accuracy (greedy)
  ARC-Easy  multiple choice by length-normalised loglikelihood
  HellaSwag multiple choice by length-normalised loglikelihood
"""
import os,sys,json,copy,pathlib,torch
os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"]="1"
sys.path.insert(0,str(_ROOT/"M2"/"transforms"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import laundry as LD
torch.set_grad_enabled(False)
N=int(os.environ.get("NEX","150"))

def load_tasks():
    t={}
    try:
        d=load_dataset("EleutherAI/lambada_openai","en",split=f"test[:{N}]")
        t["lambada"]=[(x["text"].rsplit(" ",1)[0], " "+x["text"].rsplit(" ",1)[1]) for x in d]
    except Exception as e: print("  lambada unavailable:",type(e).__name__)
    try:
        d=load_dataset("allenai/ai2_arc","ARC-Easy",split=f"test[:{N}]")
        t["arc_easy"]=[({"q":x["question"],"ch":x["choices"]["text"],
                         "gold":x["choices"]["label"].index(x["answerKey"])})
                       for x in d if x["answerKey"] in x["choices"]["label"]]
    except Exception as e: print("  arc unavailable:",type(e).__name__)
    try:
        d=load_dataset("Rowan/hellaswag",split=f"validation[:{N}]")
        t["hellaswag"]=[({"q":x["ctx"],"ch":x["endings"],"gold":int(x["label"])})
                        for x in d if str(x["label"]).isdigit()]
    except Exception as e: print("  hellaswag unavailable:",type(e).__name__)
    return t

@torch.no_grad()
def ll(model,tok,ctx,cont):
    a=tok(ctx,return_tensors="pt")["input_ids"]
    b=tok(cont,return_tensors="pt",add_special_tokens=False)["input_ids"]
    ids=torch.cat([a,b],dim=1)[:, -512:]
    lg=model(ids).logits.float().log_softmax(-1)
    n=b.shape[1]
    tgt=ids[0,-n:]
    lp=lg[0,-n-1:-1].gather(-1,tgt[:,None]).sum().item()
    return lp, n

def evaluate(model,tok,tasks):
    out={}
    if "lambada" in tasks:
        c=0
        for ctx,gold in tasks["lambada"]:
            ids=tok(ctx,return_tensors="pt")["input_ids"][:, -512:]
            nxt=model(ids).logits[0,-1].argmax().item()
            c+= tok.decode([nxt]).strip()==gold.strip()
        out["lambada"]=c/len(tasks["lambada"])
    for name in ("arc_easy","hellaswag"):
        if name not in tasks: continue
        c=0
        for x in tasks[name]:
            sc=[]
            for ch in x["ch"]:
                lp,n=ll(model,tok,x["q"]," "+ch); sc.append(lp/max(n,1))
            c+= int(max(range(len(sc)),key=lambda i:sc[i])==x["gold"])
        out[name]=c/len(tasks[name])
    out["mean"]=sum(v for k,v in out.items() if k!="mean")/max(1,len([k for k in out if k!="mean"]))
    return out

if __name__=="__main__":
    C=sys.argv[1] if len(sys.argv)>1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"loading tasks (N={N} each)…"); tasks=load_tasks()
    print("tasks:",{k:len(v) for k,v in tasks.items()})
    tok=AutoTokenizer.from_pretrained(C)
    ref=AutoModelForCausalLM.from_pretrained(C).eval()
    VARIANTS=[("baseline",None),
              ("X1a.X1b.X2 (EXACT)",lambda m:[LD.X1a_mlp_permute(m,1.0,11),LD.X1b_head_permute(m,11),LD.X2_norm_scale(m,0.5,11)]),
              ("Q1 int8",lambda m: LD.Q1_quant_roundtrip(m,8,True)),
              ("Q1 int4",lambda m: LD.Q1_quant_roundtrip(m,4,True)),
              ("N1emb sigma=0.10",lambda m: LD.N1_embed_noise(m,0.10,11)),
              ("N1emb sigma=0.40",lambda m: LD.N1_embed_noise(m,0.40,11)),
              ("X5 anchor sigma=1.0",None)]
    base=None; res=[]
    hdr=["variant"]+list(tasks)+["mean","rho"]
    print("\n"+"".join(f"{h:>13}" if h!="variant" else f"{h:<24}" for h in hdr)); print("-"*100)
    for name,fn in VARIANTS:
        if name.startswith("X5"): continue
        m=copy.deepcopy(ref)
        if fn: fn(m)
        s=evaluate(m,tok,tasks); del m
        if base is None: base=s["mean"]
        s["rho"]=s["mean"]/base if base else 0
        res.append({"variant":name,**s})
        print(f"{name:<24}"+"".join(f"{s.get(k,0):>13.4f}" for k in list(tasks)+["mean","rho"]),flush=True)
    (_ROOT/"analysis"/"capability.json").write_text(json.dumps(res,indent=1))
    print("DONE-CAPABILITY")
