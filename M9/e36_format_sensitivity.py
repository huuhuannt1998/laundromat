"""E36 -- is E35's format effect the tensor ORDER, and does it move the three rejections?

E35 showed legal-bert scoring 0.6061 from its pytorch_model.bin and 0.7391 from the hub's
model.safetensors conversion. Two things differ between those files: tensor order (a .bin lists
tensors in module order; safetensors lists them sorted by name) and content (the .bin also holds
the tied cls.predictions.decoder weight and bias, which the conversion drops).

(a) Isolate order: convert legal-bert's own .bin to safetensors locally, keeping EVERY tensor,
    decoder included. If this scores 0.7391, order alone explains E35.
(b) The paper's three rejected derivatives (section V-D) all ship only pytorch_model.bin and are
    read against parents the tool reads as safetensors. Convert each .bin locally, all tensors
    kept, and rerun against the same parent. Reported whichever way it falls.

Conversion: torch.load(weights_only=True) -> safetensors.torch.save_file, tensors made contiguous
and de-aliased with .clone(); config and tokenizer files copied unchanged. Same CLI invocation as
E30/E35 (child directory first, parent hub id second, --no-cache).

Writes a NEW file, M9/e36_format_sensitivity.jsonl. Copies in M9/work_e36, deleted after.
"""
import os, json, shutil, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from safetensors.torch import save_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e36"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e36_format_sensitivity.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))
B = "google-bert/bert-base-uncased"
PAIRS = [("nlpaueb/legal-bert-base-uncased", B, "order isolation"),
         ("Intel/dynamic_tinybert", B, "rejected, 0.3914 as published"),
         ("microsoft/xtremedistil-l6-h256-uncased", B, "rejected, 0.4211 as published"),
         ("cardiffnlp/twitter-roberta-base-sentiment-latest", "FacebookAI/roberta-base",
          "rejected, 0.5375 as published")]

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
for child, parent, note in PAIRS:
    snap, rev = main_snapshot(child)
    for fmt in ("bin", "safetensors-from-bin"):
        d = WORK / (child.split("/")[-1] + "-" + fmt)
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        for f in snap.iterdir():
            if f.name.endswith((".safetensors", ".msgpack", ".h5")) or f.name in ("README.md",): continue
            if f.name == "pytorch_model.bin" and fmt != "bin": continue
            if f.is_file(): shutil.copy(f.resolve(), d / f.name)
        if fmt != "bin":
            sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
            save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(d / "model.safetensors"))
            n_tensors = len(sd); del sd
        else:
            n_tensors = None
        r, err = mpk(d, parent)
        rec = dict(child=child, parent=parent, note=note, format=fmt, main_revision=rev,
                   n_tensors_converted=n_tensors, files=sorted(p.name for p in d.iterdir()))
        if r is None: rec.update(error=err)
        else: rec.update(scores=r["scores"], signals=r["signals"])
        with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
        s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
        print(f"{child:<50}{fmt:<22} identity {s.get('identity_score')}  nlf {g.get('nlf')}  "
              f"wvc {g.get('wvc')}  eas {g.get('eas')}  -> {s.get('provenance_decision')} "
              f"(tier {s.get('mfi_tier')}) {err[-120:] if err else ''}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
print("DONE-E36")
