"""E44 -- one-format signal vectors for the pairs E35/E36/E39/E40 did not cover.

Purpose. The E9 recalibration (M7/e9b_recalibrate.py), the E10 threshold sweep and the E6 gate
counterfactual were computed on as-served signal vectors. E35-E40 showed that identical weights
score differently by tensor order (a .bin lists tensors in module order, a safetensors file in
sorted order; the tool prefers safetensors). E36/E39/E40 re-read 15 children after re-saving
their own pytorch_model.bin as safetensors; E35 gives twitter-roberta-base from the hub's own
conversion. The fixed canonical format used for the one-format re-check is therefore
SAFETENSORS ORDER ON BOTH SIDES, as in E41.

Every remaining pair in the E9 28-pair set and the E6/E10 31-row corpus already reads
safetensors on both sides as served (main ships model.safetensors and the HF cache holds no
.bin for it), EXCEPT three, whose child or parent ships only pytorch_model.bin on `main`:
  nreimers/BERT-Tiny_L-2_H-128_A-2  vs google-bert/bert-base-uncased   (child .bin)
  sshleifer/distilbart-cnn-6-6      vs facebook/bart-large-cnn         (child .bin)
  sshleifer/distilbart-xsum-6-6     vs facebook/bart-large-xsum        (child AND parent .bin)
This script re-reads those three in both formats with E39's procedure (re-save the model's own
main-revision pytorch_model.bin as safetensors, every tensor kept, .contiguous().clone()).

It also runs reproduction controls: three pairs that are claimed to read safetensors on both
sides as served are re-run from the hub ids, to check that the frozen as-served vectors were
indeed read that way (if they reproduce, they can be used unchanged as one-format vectors).

Invocation as E35/E39: pinned CLI (M1/oracle/model-provenance-kit, v1.1.0, a75007d),
`provenancekit compare <child> <parent> --json --no-cache`, child first. Run fully offline
(HF_HUB_OFFLINE=1, uv --offline): only files already in the local HF cache are read; nothing
is downloaded and nothing is sent anywhere.

Writes a NEW file, M9/e44_one_format_remaining.jsonl. Copies live in M9/work_e44, deleted after.
"""
import os, json, shutil, pathlib, subprocess, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UV_OFFLINE"] = "1"
import torch
from safetensors.torch import save_file
ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e44"; WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "M9" / "e44_one_format_remaining.jsonl"
HUB = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home()/".cache/huggingface/hub"))

# (child, parent, expected as-served identity from the frozen artifacts, source of that value)
CONTROLS = [
    ("EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-70m", 0.7856, "M4/power_fix.jsonl"),
    ("distilbert/distilroberta-base", "FacebookAI/roberta-base", 0.73, "M1/results/pairs2.jsonl"),
    ("sentence-transformers/all-distilroberta-v1", "FacebookAI/roberta-base", 0.7304,
     "M1/results/tier3_positives.jsonl"),
]
# (child, parent, parent_ships_bin_only)
REREAD = [
    ("nreimers/BERT-Tiny_L-2_H-128_A-2", "google-bert/bert-base-uncased", False),
    ("sshleifer/distilbart-cnn-6-6", "facebook/bart-large-cnn", False),
    ("sshleifer/distilbart-xsum-6-6", "facebook/bart-large-xsum", True),
]
SKIP_SUFFIX = (".safetensors", ".msgpack", ".h5", ".meta", ".index", ".pbtxt", ".ot")


def main_snapshot(mid):
    d = HUB / ("models--" + mid.replace("/", "--"))
    rev = (d / "refs" / "main").read_text().strip()
    return d / "snapshots" / rev, rev


def materialise(mid, fmt, tag):
    """Copy mid's main snapshot (non-weight files) into a work dir; weights as .bin or re-saved."""
    snap, rev = main_snapshot(mid)
    d = WORK / (mid.split("/")[-1] + "-" + tag + "-" + fmt)
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for f in snap.iterdir():
        if not f.is_file() and not f.is_symlink(): continue
        if f.name.endswith(SKIP_SUFFIX) or "ckpt" in f.name or f.name in ("README.md", "LICENSE"):
            continue
        if f.name == "pytorch_model.bin" and fmt != "bin": continue
        shutil.copy(f.resolve(), d / f.name)
    n = None
    if fmt == "safetensors-from-bin":
        sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous().clone() for k, v in sd.items()}, str(d / "model.safetensors"))
        n = len(sd); del sd
    return d, rev, n


def mpk(a, b):
    t0 = time.time()
    p = subprocess.run(["uv", "run", "--offline", "provenancekit", "compare", str(a), str(b),
                        "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_OFFLINE": "1",
                            "UV_OFFLINE": "1"})
    i = p.stdout.find("{")
    dt = round(time.time() - t0, 1)
    return ((json.loads(p.stdout[i:]), None, dt) if i >= 0
            else (None, (p.stderr or "")[-600:], dt))


def emit(rec, r, err, dt):
    rec["seconds"] = dt
    if r is None: rec["error"] = err
    else: rec.update(scores=r["scores"], signals=r["signals"])
    with OUT.open("a") as fh: fh.write(json.dumps(rec) + "\n")
    s = (r or {}).get("scores", {}); g = (r or {}).get("signals", {})
    print(f"{rec['child']:<45}{rec['arm']:<34} tier {s.get('mfi_tier')} identity "
          f"{s.get('identity_score')} nlf {g.get('nlf')} wvc {g.get('wvc')} -> "
          f"{s.get('provenance_decision')} ({dt}s) {err[-200:] if err else ''}", flush=True)


OUT.write_text("")
for child, parent, expect, src in CONTROLS:
    r, err, dt = mpk(child, parent)
    emit(dict(kind="reproduction_control", child=child, parent=parent, arm="as-served (hub ids)",
              child_main=main_snapshot(child)[1], parent_main=main_snapshot(parent)[1],
              expected_identity=expect, expected_source=src), r, err, dt)

for child, parent, parent_bin_only in REREAD:
    # arm 1: as served -- child .bin dir, parent hub id (reproduces the frozen value)
    d, rev, _ = materialise(child, "bin", "child")
    r, err, dt = mpk(d, parent)
    emit(dict(kind="reread", child=child, parent=parent, arm="as-served (child bin, parent hub)",
              child_main=rev, parent_main=main_snapshot(parent)[1]), r, err, dt)
    if parent_bin_only:
        # arm 1b: both as local .bin dirs -- checks that passing the parent as a dir is neutral
        pd, prev, _ = materialise(parent, "bin", "parent")
        r, err, dt = mpk(d, pd)
        emit(dict(kind="reread", child=child, parent=parent, arm="both bin dirs",
                  child_main=rev, parent_main=prev), r, err, dt)
        shutil.rmtree(pd, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    # arm 2: one format -- child re-saved as safetensors; parent safetensors (hub, or re-saved)
    d, rev, n = materialise(child, "safetensors-from-bin", "child")
    if parent_bin_only:
        pd, prev, pn = materialise(parent, "safetensors-from-bin", "parent")
        r, err, dt = mpk(d, pd)
        emit(dict(kind="reread", child=child, parent=parent, arm="one-format (both st-from-bin)",
                  child_main=rev, parent_main=prev, n_tensors_child=n, n_tensors_parent=pn),
             r, err, dt)
        shutil.rmtree(pd, ignore_errors=True)
    else:
        r, err, dt = mpk(d, parent)
        emit(dict(kind="reread", child=child, parent=parent, arm="one-format (child st-from-bin, parent hub st)",
                  child_main=rev, parent_main=main_snapshot(parent)[1], n_tensors_child=n), r, err, dt)
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)
print("DONE-E44")
