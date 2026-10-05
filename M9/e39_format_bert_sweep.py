"""E39 -- E36's conversion applied to the rest of the BERT sweep (Table 10).

E36 showed that re-saving legal-bert's own pytorch_model.bin as safetensors, every tensor kept,
moves its identity score from 0.6061 to 0.7391, because the tool's norm-layer and positional
signals concatenate tensors in file order. Every other child in Table 10 except squad-v1 also
ships only pytorch_model.bin on `main`, read against a parent the tool reads as safetensors. This
reruns each of them in both formats with E36's procedure, so the paper can say how far the
format moves the scores it prints. Bio_ClinicalBERT lacks `architectures`; both of its copies
get the same repair M5/corpus_distance.py wrote (["BertForMaskedLM"]).

Procedure, invocation and parent as E36 (child directory first, parent hub id second).
Writes a NEW file, M9/e39_format_bert_sweep.jsonl. Copies in M9/work_e39, deleted after.
"""
import os, json, shutil, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from safetensors.torch import save_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e39"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e39_format_bert_sweep.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
B = "google-bert/bert-base-uncased"
CHILDREN = [("textattack/bert-base-uncased-SST-2", None), ("textattack/bert-base-uncased-imdb", None),
            ("ProsusAI/finbert", None),
            ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", None),
            ("emilyalsentzer/Bio_ClinicalBERT", {"architectures": ["BertForMaskedLM"]})]

def main_snapshot(mid):
    d = HUB / ("models--" + mid.replace("/", "--"))
    rev = (d / "refs" / "main").read_text().strip()
    return d / "snapshots" / rev, rev

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", str(a), str(b), "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return (json.loads(p.stdout[i:]), None) if i >= 0 else (None, (p.stderr or "")[-400:])

OUT.write_text("")
for child, patch in CHILDREN:
    snap, rev = main_snapshot(child)
    for fmt in ("bin", "safetensors-from-bin"):
        d = WORK / (child.split("/")[-1] + "-" + fmt)
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for f in snap.iterdir():
            if not f.is_file(): continue
            if f.name.endswith((".safetensors", ".msgpack", ".h5", ".meta", ".index", ".pbtxt")) \
               or "ckpt" in f.name or f.name in ("README.md", "LICENSE"): continue
            if f.name == "pytorch_model.bin" and fmt != "bin": continue
            shutil.copy(f.resolve(), d / f.name)
        if patch:
            cfg = json.loads((d / "config.json").read_text()); cfg.update(patch)
            (d / "config.json").write_text(json.dumps(cfg, indent=1))
        if fmt != "bin":
            sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
            save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(d / "model.safetensors"))
            del sd
        r, err = mpk(d, B)
        rec = dict(child=child, parent=B, format=fmt, main_revision=rev, config_patch=patch)
        if r is None: rec.update(error=err)
        else: rec.update(scores=r["scores"], signals=r["signals"])
        with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
        s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
        print(f"{child:<55}{fmt:<22} tier {s.get('mfi_tier')} identity {s.get('identity_score')}  "
              f"nlf {g.get('nlf')}  wvc {g.get('wvc')}  -> {s.get('provenance_decision')} {err[-120:] if err else ''}",
              flush=True)
        shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E39")
