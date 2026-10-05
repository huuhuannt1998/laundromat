"""Head-vs-tail: is the 60.6% declaration rate an artifact of sampling the popular head?
Compares the download-ranked head against a recency-ranked slice (a very different population).
Metadata only."""
import os,json,re,pathlib,sys
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
api=HfApi()
DERIV=re.compile(r"(instruct|chat|-sft|-dpo|-orpo|-rlhf|finetune|fine-tune|-ft-|lora|qlora|"
                 r"merge|slerp|distil|quantis|quantiz|-gguf|-awq|-gptq|-bnb|uncensored|abliterat)",re.I)
def survey(sort,n,label):
    rows=[]
    for m in api.list_models(pipeline_tag="text-generation",sort=sort,limit=n,cardData=True,full=False):
        cd=m.card_data.to_dict() if getattr(m,"card_data",None) else {}
        bm=cd.get("base_model")
        if isinstance(bm,str): bm=[bm]
        rows.append({"id":m.id,"declared":bool(bm),"n_base":len(bm) if bm else 0,
                     "implies":bool(DERIV.search(m.id))})
    N=len(rows); dec=sum(r["declared"] for r in rows); imp=sum(r["implies"] for r in rows)
    impdec=sum(1 for r in rows if r["implies"] and r["declared"])
    multi=sum(1 for r in rows if r["n_base"]>1)
    print(f"{label:<26} n={N:<5} declare={100*dec/N:5.1f}%   implies-deriv={100*imp/N:5.1f}%   "
          f"of-implied-declared={100*impdec/max(1,imp):5.1f}%   multi-parent={multi}")
    return rows,{"label":label,"n":N,"declare_pct":100*dec/N,"implies_pct":100*imp/N,
                 "of_implied_declared_pct":100*impdec/max(1,imp),"multi":multi}
out=[]
for sort,label in [("downloads","head (by downloads)"),
                   ("lastModified","recent (by lastModified)"),
                   ("likes","by likes"),
                   ("createdAt","newest (by createdAt)")]:
    try:
        rows,s=survey(sort,3000,label); out.append(s)
        pathlib.Path(f"claim_{sort}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    except Exception as e:
        print(f"{label:<26} ERROR {type(e).__name__}: {str(e)[:80]}")
pathlib.Path("claim_survey2.json").write_text(json.dumps(out,indent=1))
print("DONE-CLAIM2")
