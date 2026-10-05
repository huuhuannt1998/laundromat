"""M2 Stage-0 T1/T2 — wide scans to build the frozen null N and measure DB density."""
import json, subprocess, pathlib, os, sys

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT  = pathlib.Path(str(_ROOT) + "/M2/results/wide_scans.jsonl")
TARGETS = [
 ("HuggingFaceTB/SmolLM2-135M",          None,                    "gate parent (in DB)"),
 ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M",          "gate child"),
 ("Qwen/Qwen2.5-0.5B",                   None,                    "gate parent (in DB)"),
 ("Qwen/Qwen2.5-0.5B-Instruct",          "Qwen2.5-0.5B",          "gate child"),
 ("gpt2",                                None,                    "in DB"),
 ("distilgpt2",                          "gpt2",                  "distilled child"),
 ("EleutherAI/pythia-160m",              None,                    "not in DB"),
 ("EleutherAI/pythia-160m-deduped",      None,                    "P~ member"),
 ("bigscience/bloom-560m",               None,                    "not in DB (bloomz is)"),
 ("google-bert/bert-base-uncased",       None,                    "in DB"),
]
def run(m):
    p=subprocess.run(["uv","run","provenancekit","scan",m,"--json","--top-k","300","--threshold","0.0"],
        cwd=REPO,capture_output=True,text=True,timeout=3600,
        env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    if p.returncode!=0: return {"error":p.stderr[-400:]}
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":"nojson"}
with OUT.open("w") as fh:
    for m,parent,note in TARGETS:
        r=run(m)
        fh.write(json.dumps({"target":m,"true_parent":parent,"note":note,"result":r})+"\n"); fh.flush()
        ms=r.get("matches",[])
        if "error" in r:
            print(f"{m:<40} ERROR {r['error'][:90]}",flush=True); continue
        scores=[x["scores"]["pipeline_score"] for x in ms]
        pr=""
        if parent:
            idx=[i for i,x in enumerate(ms,1) if x["model_id"]==parent]
            pr=f"  parent_rank={idx[0] if idx else 'ABSENT'}"
        print(f"{m:<40} candidates={len(ms):>3}  max={max(scores) if scores else 0:.4f} "
              f"min={min(scores) if scores else 0:.4f}  n>0.75={sum(s>0.75 for s in scores):>3}"
              f"  n>0.65={sum(s>0.65 for s in scores):>3}{pr}",flush=True)
print("DONE")
