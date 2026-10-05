"""E35 -- does the identity score depend on how identical weights are serialised?

Review round 6 (R1, R2) found two pairs whose identity score differs between tables with no
weight changed:
  legal-bert vs bert-base-uncased       0.6061 Not Matched (M5/sweep_bert.jsonl, hub id)
                                        0.7391 Weak Match  (M9/e30 "unmodified copy" control)
  twitter-roberta-base vs roberta-base  0.6899             (M5/corpus_distance.jsonl, hub id)
                                        0.6997             (M9/e30 "unmodified copy" control)
In both, only NLF and WVC move (EAS and LEP are identical to four decimals). Both repos ship
pytorch_model.bin on `main`; the HF cache also holds a model.safetensors conversion in a sibling
snapshot. E30's materialise() copied every cached file, so its copy held both formats, and the
tool's create_streamer() prefers safetensors (core/signals/streamers.py). Read from the hub id,
the tool sees only the .bin. The .bin lists tensors in module order (LayerNorm.weight before
.bias, layer 2 before layer 10); a safetensors file lists them sorted by name. NLF concatenates
norm vectors in iteration order and WVC concatenates each layer's matrices in iteration order.

The test: the same weights, each format alone, run through the pinned CLI exactly as E30 ran
its arms (child directory first, parent hub id second, --no-cache). Config and tokenizer files
are identical across the two copies. If the scores split by format, serialisation order moves
the identity score.

Writes a NEW file, M9/e35_serialisation.jsonl. Copies live in M9/work_e35 and are deleted after.
"""
import os, json, shutil, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e35"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e35_serialisation.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))

PAIRS = [("nlpaueb/legal-bert-base-uncased", "google-bert/bert-base-uncased"),
         ("cardiffnlp/twitter-roberta-base", "FacebookAI/roberta-base")]
WEIGHTS = {"bin": "pytorch_model.bin", "safetensors": "model.safetensors"}

def cached(mid):
    base = HUB / ("models--" + mid.replace("/", "--")) / "snapshots"
    files = {}
    for s in sorted(base.glob("*")):
        for f in s.iterdir():
            files.setdefault(f.name, f.resolve())
    return files, (HUB / ("models--" + mid.replace("/", "--")) / "refs" / "main").read_text().strip()

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", str(a), str(b), "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return (json.loads(p.stdout[i:]), None) if i >= 0 else (None, (p.stderr or "")[-400:])

OUT.write_text("")
for child, parent in PAIRS:
    files, main = cached(child)
    for fmt, wname in WEIGHTS.items():
        d = WORK / (child.split("/")[-1] + "-" + fmt)
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for name, src in files.items():
            if name in WEIGHTS.values() and name != wname: continue
            if name.endswith((".msgpack", ".h5")) or name == "README.md": continue
            shutil.copy(src, d / name)
        r, err = mpk(d, parent)
        rec = dict(child=child, parent=parent, format=fmt, main_revision=main,
                   files=sorted(p.name for p in d.iterdir()))
        if r is None: rec.update(error=err)
        else: rec.update(scores=r["scores"], signals=r["signals"])
        with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
        s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
        print(f"{child:<36}{fmt:<12} identity {s.get('identity_score')}  nlf {g.get('nlf')}  "
              f"wvc {g.get('wvc')}  eas {g.get('eas')}  lep {g.get('lep')}  end {g.get('end')}  "
              f"-> {s.get('provenance_decision')} (tier {s.get('mfi_tier')})", flush=True)
        shutil.rmtree(d, ignore_errors=True)
print("DONE-E35")
