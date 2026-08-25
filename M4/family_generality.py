"""Does the EAS-high-on-NON-DERIVED effect generalise beyond Pythia?

All same-recipe evidence so far comes from one family (Pythia, Pile vs deduplicated Pile).
Same-family-DIFFERENT-SIZE pairs are the other controlled case of "same recipe, no derivation"
-- Cisco's Constitution S8 lists them explicitly under Independent (Llama-2-7B vs Llama-2-13B,
XLM-R base vs large). Each size is pretrained independently; neither is derived from the other.
Five families here, so the claim stops resting on Pythia.
"""
import os,json,subprocess,pathlib
os.environ["HF_HUB_DISABLE_XET"]="1"

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES=pathlib.Path(str(_ROOT) + "/M4")
def mpk(a,b):
    p=subprocess.run(["uv","run","provenancekit","compare",a,b,"--json","--no-cache"],
        cwd=REPO,capture_output=True,text=True,timeout=7200,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":(p.stderr or "")[-160:]}
PAIRS=[
 ("GPT-2",      "gpt2",                        "gpt2-medium"),
 ("OPT",        "facebook/opt-125m",           "facebook/opt-350m"),
 ("Cerebras",   "cerebras/Cerebras-GPT-111M",  "cerebras/Cerebras-GPT-256M"),
 ("SmolLM2",    "HuggingFaceTB/SmolLM2-135M",  "HuggingFaceTB/SmolLM2-360M"),
 ("Pythia",     "EleutherAI/pythia-160m",      "EleutherAI/pythia-410m"),
 ("BLOOM",      "bigscience/bloom-560m",       "bigscience/bloom-1b1"),
]
out=(RES/"family_generality.jsonl").open("w")
print("Same-family / different-size pairs = 'same recipe, independently pretrained, NO derivation'")
print("Cisco Constitution S8 lists this category under Independent.\n")
print(f"{'family':<12}{'pair':<44}{'tier':>5}{'pipe':>8}{'sig_id':>8}{'EAS':>8}{'WVC':>8}  verdict")
print("-"*104)
for fam,a,b in PAIRS:
    r=mpk(a,b)
    if "error" in r:
        print(f"{fam:<12}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44}  ERROR {r['error'][:50]}"); continue
    s=r["scores"]; g=r["signals"]
    out.write(json.dumps({"family":fam,"a":a,"b":b,"scores":s,"signals":g})+"\n"); out.flush()
    print(f"{fam:<12}{a.split('/')[-1]+' | '+b.split('/')[-1]:<44}{s['mfi_tier']:>5}{s['pipeline_score']:>8.4f}"
          f"{s['identity_score']:>8.4f}{(g.get('eas') or 0):>8.4f}"
          f"{(0 if g.get('wvc') is None else g['wvc']):>8.4f}  {s['provenance_decision']}",flush=True)
out.close(); print("\nDONE-FAMGEN")
