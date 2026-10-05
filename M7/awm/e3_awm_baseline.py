"""E3 -- the official AWM baseline (github.com/LUMIA-Group/AWM, commit bc20ff8, ICLR 2026).

A reviewer will ask why we built our own assignment-based signal instead of testing
the actual prior method.  This runs the method as its authors ship it.

SCOPE, established by testing rather than assumed: the official implementation reads
attention weights via key patterns requiring `.layers.` / `.h.` / `.blocks.` with
LLaMA/Qwen/GPT naming.  BERT- and RoBERTa-style encoders (`encoder.layer.N.attention.
self.query.weight`) and BART's `model.encoder.layers.*` match none of them, so the
implementation cannot score most of our corpus.  We therefore evaluate on the causal
subset and report the exclusion as a property of the released method, not as a failure
of the pairs.

The question that matters is the paper's own, asked of the prior method: does AWM
separate true descent from INDEPENDENT SAME-RECIPE training, or does it confuse them
the way the studied tool does?
"""
import os, sys, json, re, subprocess, pathlib, gc, time
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Repo root is derived from this file's location so the artifact runs from any
# checkout. Set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_HERE = _pl.Path(__file__).resolve()
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _HERE.parents if (p / "MANIFEST-dataintegrity.txt").exists())))


ROOT = pathlib.Path(_ROOT)
AWM  = ROOT / "M7" / "awm" / "repo"
WORK = ROOT / "M7" / "awm" / "work"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = ROOT / "M7" / "awm" / "e3_awm_results.jsonl"

PAT_AVG = re.compile(r"Average\s+(\S+)\s+Similarity\(%\)\s*=\s*([-\d.]+)")
PAT_Z   = re.compile(r"Absolute Z-Score vs negative pairs\s*=\s*([\d.]+)")

P = "EleutherAI/pythia-"
PAIRS = [
  # label, model_a, rev_a, model_b, rev_b, class, note
  ("gpt2 -> distilgpt2",        "openai-community/gpt2",None,"distilbert/distilgpt2",None,"D-DIST","documented distillation"),
  ("SmolLM2 -> Instruct",       "HuggingFaceTB/SmolLM2-135M",None,"HuggingFaceTB/SmolLM2-135M-Instruct",None,"D-MT","instruction tuning"),
  ("Qwen2.5-0.5B -> Instruct",  "Qwen/Qwen2.5-0.5B",None,"Qwen/Qwen2.5-0.5B-Instruct",None,"D-MT","instruction tuning"),
  ("pythia-160m step1k->final", P+"160m","step1000",P+"160m","step143000","D-CP","same run, 99.3% of training after fork"),
  ("pythia-160m step16k->final",P+"160m","step16000",P+"160m","step143000","D-CP","same run, 88.8% after fork"),
  ("pythia-160m step128k->final",P+"160m","step128000",P+"160m","step143000","D-CP","same run, 10.5% after fork"),
  ("pythia-70m vs -deduped",    P+"70m",None,P+"70m-deduped",None,"N-SR","independent run, same recipe and size"),
  ("pythia-160m vs -deduped",   P+"160m",None,P+"160m-deduped",None,"N-SR","independent run, same recipe and size"),
  ("pythia-410m vs -deduped",   P+"410m",None,P+"410m-deduped",None,"N-SR","independent run, same recipe and size"),
  ("pythia-1b vs -deduped",     P+"1b",None,P+"1b-deduped",None,"N-SR","independent run, same recipe and size"),
  ("pythia-160m vs -410m",      P+"160m",None,P+"410m",None,"N-FAM","same recipe, different size"),
  ("gpt2 vs gpt2-medium",       "openai-community/gpt2",None,"openai-community/gpt2-medium",None,"N-FAM","same recipe, different size"),
  ("pythia-160m vs gpt2",       P+"160m",None,"openai-community/gpt2",None,"N-UNREL","unrelated"),
  ("SmolLM2 vs pythia-160m",    "HuggingFaceTB/SmolLM2-135M",None,P+"160m",None,"N-UNREL","unrelated"),
]

def materialise(rid, rev=None):
    """AWM needs a local dir. Save via CausalLM so the state_dict keeps the
    `transformer.`/`model.` prefix its key regex requires."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tag = rid.replace("/", "--") + (f"@{rev}" if rev else "")
    d = WORK / tag
    if (d / "config.json").exists():
        return d
    kw = {"revision": rev} if rev else {}
    m = AutoModelForCausalLM.from_pretrained(rid, **kw)
    m.save_pretrained(d)
    try: AutoTokenizer.from_pretrained(rid, **kw).save_pretrained(d)
    except Exception: pass
    del m; gc.collect()
    return d

def awm(p1, p2):
    r = subprocess.run([sys.executable, "main.py", "--model_paths", str(p1), str(p2),
                        "--device", "cpu"], cwd=AWM, capture_output=True, text=True, timeout=7200)
    avgs = {k: float(v)/100.0 for k, v in PAT_AVG.findall(r.stdout)}
    z = PAT_Z.search(r.stdout)
    return avgs, (float(z.group(1)) if z else None), r.stdout[-500:]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip(): done.add(json.loads(line)["label"])

print(f"=== E3: official AWM on {len(PAIRS)} causal pairs ===", flush=True)
for i, (lab, ma, ra, mb, rb, cls, note) in enumerate(PAIRS, 1):
    if lab in done:
        print(f"[{i}/{len(PAIRS)}] {lab}: cached", flush=True); continue
    t0 = time.time()
    try:
        pa, pb = materialise(ma, ra), materialise(mb, rb)
    except Exception as e:
        rec = dict(label=lab, cls=cls, status="LOAD_FAILED", reason=f"{type(e).__name__}: {e}"[:250])
        OUT.open("a").write(json.dumps(rec)+"\n")
        print(f"[{i}/{len(PAIRS)}] {lab}: LOAD FAILED {type(e).__name__}", flush=True); continue
    try:
        avgs, z, tail = awm(pa, pb)
    except Exception as e:
        avgs, z, tail = {}, None, f"{type(e).__name__}: {e}"
    rec = dict(label=lab, cls=cls, note=note, model_a=ma, rev_a=ra, model_b=mb, rev_b=rb,
               awm_wq=avgs.get("Wq_weights"), awm_wk=avgs.get("Wk_weights"),
               awm_wqwk=avgs.get("Wq_Wk_weights"), awm_z=z,
               status="OK" if avgs else "UNSUPPORTED",
               reason=None if avgs else tail[-250:], secs=round(time.time()-t0,1))
    OUT.open("a").write(json.dumps(rec)+"\n")
    w = rec["awm_wqwk"]
    print(f"[{i}/{len(PAIRS)}] {cls:<8} {lab:<30} AWM={('%.4f'%w) if w is not None else 'UNSUP':>8} z={z} ({rec['secs']}s)", flush=True)
    gc.collect()
print("DONE-E3", flush=True)
