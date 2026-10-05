"""C5 power fix: same-shape pairs only, both arms.

C5 (design 11.4 contrast 5) is the separability claim that motivates RESCORE/LAP.
On the SAME-SHAPE stratum -- the only stratum where WVC is even defined -- it
separates perfectly (AUC 0.000, every P~ below every P+) but n=3v3 makes the
minimum attainable two-sided exact p 0.10, so the preregistered test CANNOT
reject at alpha=0.05.  That is a power failure, not a null result.

At 7v7 the minimum exact p is 2/C(14,7) = 0.00058.  This script adds the pairs.
Every model is small, public and ungated; nothing is requested from anyone.
"""
import os, sys, json, subprocess, pathlib, glob
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT = pathlib.Path(str(_ROOT) + "/M4/power_fix.jsonl")
CACHE = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))


def dims(repo):
    d = CACHE / ("models--" + repo.replace("/", "--"))
    cs = sorted(glob.glob(str(d / "snapshots" / "*" / "config.json")))
    if not cs:
        return None
    c = json.load(open(cs[0]))
    h = c.get("hidden_size") or c.get("n_embd") or c.get("d_model") or c.get("dim")
    l = c.get("num_hidden_layers") or c.get("n_layer") or c.get("n_layers") or c.get("num_layers")
    return (h, l)


def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}


# SAME-SHAPE only.  P~ = independent runs of the same recipe at the same size.
# Pythia ships -deduped (different corpus, same recipe) and -v0 (earlier
# independent run) at each scale: a controlled ladder, not an anecdote.
PAIRS = [
    ("P~", "EleutherAI/pythia-70m",          "EleutherAI/pythia-70m-deduped"),
    ("P~", "EleutherAI/pythia-160m",         "EleutherAI/pythia-160m-deduped"),
    ("P~", "EleutherAI/pythia-410m",         "EleutherAI/pythia-410m-deduped"),
    ("P~", "EleutherAI/pythia-70m",          "EleutherAI/pythia-70m-v0"),
    ("P~", "EleutherAI/pythia-160m",         "EleutherAI/pythia-160m-v0"),
    ("P~", "EleutherAI/pythia-70m-deduped",  "EleutherAI/pythia-70m-deduped-v0"),
    ("P~", "EleutherAI/pythia-410m",         "EleutherAI/pythia-410m-v0"),
    # P+ = published derivatives that preserve dimensions (fine-tune / multitask).
    ("P+", "HuggingFaceTB/SmolLM2-135M",     "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("P+", "Qwen/Qwen2.5-0.5B",              "Qwen/Qwen2.5-0.5B-Instruct"),
    ("P+", "FacebookAI/roberta-base",        "ehsanaghaei/SecureBERT"),
    ("P+", "bigscience/bloom-560m",          "bigscience/bloomz-560m"),
    ("P+", "gpt2",                           "lvwerra/gpt2-imdb"),
    ("P+", "google-bert/bert-base-uncased",  "textattack/bert-base-uncased-SST-2"),
    ("P+", "FacebookAI/roberta-base",        "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("P+", "distilbert/distilbert-base-uncased",
           "distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
]

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip().startswith("{"):
            r = json.loads(l); done.add((r["a"], r["b"]))

f = OUT.open("a")
print(f"{'cls':<4}{'pair':<70}{'shape':>12}{'tier':>5}{'sig_id':>8}{'EAS':>8}{'WVC':>8}", flush=True)
print("-" * 116, flush=True)
for cls, a, b in PAIRS:
    if (a, b) in done:
        print(f"{cls:<4}{a.split('/')[-1]+' | '+b.split('/')[-1]:<70}{'(cached)':>12}", flush=True); continue
    r = mpk(a, b)
    da, db = dims(a), dims(b)
    same = (da == db and da and None not in da)
    sc = r.get("scores", {}); sg = r.get("signals", {})
    rec = dict(cls=cls, a=a, b=b, dims_a=da, dims_b=db, same_shape=bool(same),
               scores=sc, signals=sg, error=r.get("error"))
    f.write(json.dumps(rec) + "\n"); f.flush()
    lbl = "SAME" if same else ("DIFF" if da and db else "?")
    if r.get("error"):
        print(f"{cls:<4}{a.split('/')[-1]+' | '+b.split('/')[-1]:<70}{'ERROR':>12}  {r['error'][:60]}", flush=True)
    else:
        print(f"{cls:<4}{a.split('/')[-1]+' | '+b.split('/')[-1]:<70}{lbl:>12}"
              f"{sc.get('mfi_tier','?'):>5}{sc.get('identity_score',float('nan')):>8.4f}"
              f"{(sg.get('eas') if sg.get('eas') is not None else float('nan')):>8.4f}"
              f"{(sg.get('wvc') if sg.get('wvc') is not None else float('nan')):>8.4f}", flush=True)
f.close()
print("DONE-POWERFIX", flush=True)
