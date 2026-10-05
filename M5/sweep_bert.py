"""M5 replication on a second parent family.

The embedding-displacement finding rests on one parent (roberta-base), n=7.  That is
its weakest point.  This repeats the design on bert-base-uncased with published
children spanning continued pretraining (legal, financial, clinical) and pure
fine-tuning (SST-2, IMDB), so the two mechanisms are separable again.

Prediction registered BEFORE running, from the roberta fit:
  (a) embedding displacement predicts sigma_id better than overall displacement
  (b) children that reshape the embedding NORM distribution (END drops) lose the
      verdict; children that move it while preserving END keep it
  (c) pure fine-tunes cluster at low embedding displacement and stay Confirmed
"""
import os, sys, json, glob, shutil, subprocess, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModel

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT = pathlib.Path(str(_ROOT) + "/M5/sweep_bert.jsonl")
PARENT = "google-bert/bert-base-uncased"

CHILDREN = [
    ("CP legal",      "nlpaueb/legal-bert-base-uncased"),
    ("CP financial",  "ProsusAI/finbert"),
    ("CP clinical",   "emilyalsentzer/Bio_ClinicalBERT"),
    ("CP biomed",     "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"),
    ("FT sst2",       "textattack/bert-base-uncased-SST-2"),
    ("FT imdb",       "textattack/bert-base-uncased-imdb"),
    ("FT squad",      "csarron/bert-base-uncased-squad-v1"),
]


def repaired_path(repo):
    g = sorted(glob.glob(str(pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub")) /
               ("models--" + repo.replace("/", "--")) / "snapshots" / "*" / "config.json")))
    if g and json.load(open(g[0])).get("architectures"):
        return repo, False
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo)
    dst = pathlib.Path(str(_ROOT) + "/M5/repaired") / repo.replace("/", "--")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(local, dst, symlinks=False)
    cfg = json.load(open(dst / "config.json"))
    if not cfg.get("architectures"):
        cfg["architectures"] = ["BertForMaskedLM"]
        json.dump(cfg, open(dst / "config.json", "w"), indent=1)
        return str(dst), True
    return str(dst), False


def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", a, b, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}


def displacement(pm, cm):
    pd, cd = dict(pm.named_parameters()), dict(cm.named_parameters())
    num = den = enum = eden = 0.0
    for k, v in pd.items():
        if k not in cd or cd[k].shape != v.shape:
            continue
        d = (cd[k].float() - v.float()).pow(2).sum().item()
        n = v.float().pow(2).sum().item()
        num += d; den += n
        if "embedding" in k:
            enum += d; eden += n
    return {"overall": (num/den)**0.5 if den else float("nan"),
            "embedding": (enum/eden)**0.5 if eden else float("nan")}


pm = AutoModel.from_pretrained(PARENT).eval()
done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip().startswith("{"):
            done.add(json.loads(l)["child"])
f = OUT.open("a")
print(f"parent = {PARENT}\n", flush=True)
print(f"{'kind':<15}{'disp':>7}{'dispEmb':>9}{'tier':>5}{'sig_id':>8}{'EAS':>8}{'END':>8}{'WVC':>8}  verdict", flush=True)
print("-" * 104, flush=True)
for kind, child in CHILDREN:
    if child in done:
        print(f"{kind:<15} (cached)", flush=True); continue
    try:
        cm = AutoModel.from_pretrained(child).eval()
        disp = displacement(pm, cm); del cm
    except Exception as e:
        print(f"{kind:<15} LOAD FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True); continue
    tgt, rep = repaired_path(child)
    r = mpk(PARENT, tgt)
    if r.get("error"):
        kindtag = "MPK_DEFECT_architectures_null" if "list_type" in r["error"] else "other"
        print(f"{kind:<15} MPK ERROR [{kindtag}]", flush=True); continue
    sc, sg = r.get("scores", {}), r.get("signals", {})
    f.write(json.dumps(dict(kind=kind, parent=PARENT, child=child, displacement=disp,
                            scores=sc, signals=sg, config_repaired=rep)) + "\n"); f.flush()
    nf = lambda x: float("nan") if x is None else x
    print(f"{kind:<15}{disp['overall']:>7.4f}{disp['embedding']:>9.4f}{sc.get('mfi_tier','?'):>5}"
          f"{nf(sc.get('identity_score')):>8.4f}{nf(sg.get('eas')):>8.4f}{nf(sg.get('end')):>8.4f}"
          f"{nf(sg.get('wvc')):>8.4f}  {sc.get('provenance_decision')}"
          f"{'  [repaired]' if rep else ''}", flush=True)
f.close(); print("DONE-BERTSWEEP", flush=True)
