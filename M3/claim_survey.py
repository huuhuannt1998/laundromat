"""M3 — the CLAIM side of claimed-vs-derived lineage divergence, at hub scale.

M1 observed that only 2 of 6 canonical, uncontested derivations declare a machine-readable
`base_model` (jrn_01M0E5P2JWPMJY8EYRJ9BCZKKS) and hypothesised that the dominant divergence mode
at hub scale is ABSENCE of a claim rather than a FALSE claim -- two different failure modes with
different attestation consequences, which would inflate any headline number if conflated.

Fingerprinting the hub is impossible on this hardware. The CLAIM side is not: it is metadata only,
no weights downloaded. So M3 measures what an automated ML-BOM builder would actually be able to
read, which is the half that determines whether the attestation stack has anything to attest.
"""
import os,json,collections,pathlib,re,sys
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

api=HfApi()
RES=pathlib.Path(str(_ROOT) + "/M3"); RES.mkdir(exist_ok=True)
N=int(sys.argv[1]) if len(sys.argv)>1 else 3000

# Name patterns that strongly imply the model is a DERIVATIVE of something.
DERIV=re.compile(r"(instruct|chat|-sft|-dpo|-orpo|-rlhf|finetune|fine-tune|-ft-|lora|qlora|"
                 r"merge|slerp|distil|quantis|quantiz|-gguf|-awq|-gptq|-bnb|uncensored|abliterat)",re.I)
rows=[]; seen=0
for m in api.list_models(pipeline_tag="text-generation", sort="downloads",
                         limit=N, cardData=True, full=False):
    seen+=1
    cd = m.card_data.to_dict() if getattr(m,"card_data",None) else {}
    bm = cd.get("base_model")
    if isinstance(bm,str): bm=[bm]
    declared = bool(bm)
    name_implies = bool(DERIV.search(m.id))
    rows.append({"id":m.id,"declared":declared,"n_base":len(bm) if bm else 0,
                 "name_implies_derivative":name_implies,
                 "downloads":getattr(m,"downloads",0) or 0,
                 "tags":[t for t in (m.tags or []) if t in ("merge","lora","peft")]})
(RES/"claim_survey.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

n=len(rows)
dec=sum(r["declared"] for r in rows)
imp=sum(r["name_implies_derivative"] for r in rows)
imp_undec=sum(1 for r in rows if r["name_implies_derivative"] and not r["declared"])
imp_dec  =sum(1 for r in rows if r["name_implies_derivative"] and r["declared"])
noimp_dec=sum(1 for r in rows if not r["name_implies_derivative"] and r["declared"])
print("="*88); print(f"M3 CLAIM SURVEY — {n} text-generation models, ranked by downloads"); print("="*88)
print(f"declare base_model                         : {dec:>5} / {n}  ({100*dec/n:.1f}%)")
print(f"name implies a derivative                  : {imp:>5} / {n}  ({100*imp/n:.1f}%)")
print()
print(f"  name implies derivative AND declared     : {imp_dec:>5}  ({100*imp_dec/max(1,imp):.1f}% of implied)")
print(f"  name implies derivative BUT NOT declared : {imp_undec:>5}  ({100*imp_undec/max(1,imp):.1f}% of implied)  <-- CLAIM ABSENT")
print(f"  no name signal but declared              : {noimp_dec:>5}")
print()
# multi-parent (merges) — Constitution S10.2 open problem
multi=[r for r in rows if r["n_base"]>1]
print(f"declared with MULTIPLE base_models (merges): {len(multi):>5}  "
      f"({100*len(multi)/max(1,dec):.1f}% of declarers)   max parents = {max([r['n_base'] for r in multi], default=0)}")
# declaration rate by popularity decile
rows_sorted=sorted(rows,key=lambda r:-r["downloads"])
print("\ndeclaration rate by download decile (1 = most downloaded):")
for d in range(10):
    seg=rows_sorted[d*n//10:(d+1)*n//10]
    if seg: print(f"  decile {d+1}: {100*sum(x['declared'] for x in seg)/len(seg):5.1f}%  (n={len(seg)})")
print("\nDONE-CLAIM")
