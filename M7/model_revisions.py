"""Record the exact Hugging Face revision (commit sha) of every model the paper touches.

The artifact checklist requires model revisions and hashes. A model id alone is not a
pin: repositories are mutable. This resolves each id to the commit sha currently in the
local cache, which is the object the measurements were actually made against.
"""
import json, pathlib, re, os
_ROOT = pathlib.Path(os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
HUB = pathlib.Path.home()/".cache"/"huggingface"/"hub"

# ONLY the models this paper measures -- the corpus, the defence pairs, the second-substrate
# pairs and the AWM pairs. Not the vendor's fingerprint database, which lists thousands of
# models we never touch and whose revisions are not ours to attest.
ids=set()
def take(v):
    if isinstance(v,str) and v.count("/")==1 and " " not in v and not v.startswith(("http","M1/","M2/","M4/","M5/","M6/","M7/","figures/")):
        ids.add(v)
SRC = [("M6/gold_lineage_manifest.json", ("parent","child")),
       ("M7/e18_defence.json",           ("a","b")),
       ("M7/e19_layer_correspondence.json",("a","b")),
       ("M7/awm/e3_awm_results.jsonl",   ("model_a","model_b")),
       ("M4/defence_expanded.json",      ("a","b")),
       ("M4/p_tilde_arm.jsonl",          ("a","b")),
       ("M4/power_fix.jsonl",            ("a","b")),
       ("M5/sweep_bert.jsonl",           ("parent","child")),
       ("M5/corpus_distance.jsonl",      ("parent","child")),
       ("M1/results/tier3_positives.jsonl",("parent","child")),
       ("M1/results/mpk_crashes.jsonl",  ("parent","child")),
       ("M6/e12_pythia_ladder.jsonl",    ("model",))]
for rel, keys in SRC:
    f=_ROOT/rel
    if not f.exists(): continue
    txt=f.read_text(errors="ignore")
    docs=[]
    try:
        docs = [json.loads(l) for l in txt.splitlines() if l.strip()] if rel.endswith(".jsonl") else json.loads(txt)
    except Exception: continue
    if isinstance(docs,dict): docs=docs.get("rows") or docs.get("members") or [docs]
    for d in docs:
        if not isinstance(d,dict): continue
        for k in keys: take(d.get(k))
for extra in ("EleutherAI/pythia-160m","openai-community/gpt2","HuggingFaceTB/SmolLM2-135M",
              "Qwen/Qwen2.5-0.5B","bigscience/bloom-560m"):
    ids.add(extra)

rows=[]
for mid in sorted(ids):
    d = HUB/("models--"+mid.replace("/","--"))
    if not d.exists():
        rows.append(dict(model=mid, cached=False, revision=None)); continue
    revs={}
    refs=d/"refs"
    if refs.exists():
        for r in refs.iterdir():
            try: revs[r.name]=r.read_text().strip()
            except Exception: pass
    snaps=sorted(p.name for p in (d/"snapshots").iterdir()) if (d/"snapshots").exists() else []
    rows.append(dict(model=mid, cached=True, refs=revs, snapshots=snaps))
cached=[r for r in rows if r["cached"]]
out=_ROOT/"M7"/"model_revisions.json"
out.write_text(json.dumps(dict(
    note="Hugging Face repo revisions resolved from the local cache: the objects the "
         "measurements were made against. 'refs' maps a branch/revision name to its commit sha; "
         "models fetched at a pinned revision (e.g. Pythia stepN) appear as that ref name.",
    n_models=len(rows), n_cached=len(cached), models=rows), indent=1))
print(f"{len(rows)} model ids referenced; {len(cached)} resolved to a cached revision")
for r in cached[:12]:
    ref=list(r["refs"].items())[:1]
    print(f"  {r['model']:<52} {(ref[0][0]+'='+ref[0][1][:12]) if ref else (r['snapshots'][0][:12] if r['snapshots'] else '?')}")
print(f"  ... wrote {out.name}")
