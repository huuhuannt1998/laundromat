"""The kappa>0 corpus-distance sweep -- pricing C1 on the axis that actually breaks the detector.

MOTIVATION.  Two children of roberta-base, same operation (continued pretraining),
opposite verdicts:
    SecureBERT     (cybersecurity)  EAS 0.9818  END 0.9966  sigma_id 0.7767  Confirmed Match
    twitter-roberta(Twitter)        EAS 0.7346  END 0.1879  sigma_id 0.5375  NOT MATCHED
The moving variable is corpus distance, not the fact of continued pretraining.

DESIGN.  AllenAI's "Don't Stop Pretraining" release is a controlled ladder: ONE
parent, ONE procedure, four domains.  Add the two endpoints above.  For each child
measure MPK's verdict AND an independent, non-circular x-axis: relative weight
displacement ||theta_c - theta_p|| / ||theta_p||.

WHY IT PRICES C1.  Constitution C1 claims evading weight signals costs as much as
training from scratch.  Scratch = displacement ~sqrt(2) (independent draws).  If the
detector already fails at displacement far below that, C1 is quantitatively false at
kappa>0 -- and the falsifying examples are benign published models, not attacks.

All models public and ungated.  CPU only.
"""
import os, sys, json, glob, shutil, subprocess, pathlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch, numpy as np
from transformers import AutoModel, AutoTokenizer

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

torch.set_grad_enabled(False)

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT = pathlib.Path(str(_ROOT) + "/M5/corpus_distance.jsonl")
PARENT = "FacebookAI/roberta-base"

CHILDREN = [
    ("DAPT cs",           "allenai/cs_roberta_base"),
    ("DAPT biomed",       "allenai/biomed_roberta_base"),
    ("DAPT news",         "allenai/news_roberta_base"),
    ("DAPT reviews",      "allenai/reviews_roberta_base"),
    ("cyber",             "ehsanaghaei/SecureBERT"),
    ("twitter",           "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("twitter (base)",    "cardiffnlp/twitter-roberta-base"),
]


def repaired_path(repo):
    """MPK crashes (MFIFingerprint validation) when config.json lacks 'architectures'.
    Six public models hit this, including AllenAI's DAPT releases -- the canonical
    controlled ladder for this very experiment.  We materialise a local copy with the
    single missing key restored.  Weights are untouched; this is metadata repair, and
    it doubles as the demonstration that ONE KEY is the difference between verifiable
    and unverifiable."""
    from huggingface_hub import snapshot_download
    cfgp = None
    g = sorted(glob.glob(str(pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub")) /
               ("models--" + repo.replace("/", "--")) / "snapshots" / "*" / "config.json")))
    if g:
        cfgp = g[0]
        if json.load(open(cfgp)).get("architectures"):
            return repo, False              # fine as published
    local = snapshot_download(repo)
    dst = pathlib.Path(str(_ROOT) + "/M5/repaired") / repo.replace("/", "--")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(local, dst, symlinks=False)
    cfg = json.load(open(dst / "config.json"))
    if not cfg.get("architectures"):
        mt = (cfg.get("model_type") or "roberta").lower()
        cfg["architectures"] = {"roberta": ["RobertaForMaskedLM"], "bert": ["BertForMaskedLM"]}.get(
            mt, ["RobertaForMaskedLM"])
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
    """Relative Frobenius displacement over shared-name, shared-shape parameters.
    Reported overall, for the embedding alone, and for the encoder alone."""
    pd = dict(pm.named_parameters()); cd = dict(cm.named_parameters())
    num = den = 0.0
    parts = {}
    for k, v in pd.items():
        if k not in cd or cd[k].shape != v.shape:
            continue
        d = (cd[k].float() - v.float()).pow(2).sum().item()
        n = v.float().pow(2).sum().item()
        num += d; den += n
        grp = "embedding" if "embeddings" in k else "encoder"
        a, b = parts.get(grp, (0.0, 0.0))
        parts[grp] = (a + d, b + n)
    out = {"overall": (num / den) ** 0.5 if den else float("nan")}
    for g, (a, b) in parts.items():
        out[g] = (a / b) ** 0.5 if b else float("nan")
    return out


print(f"parent = {PARENT}\n", flush=True)
print(f"{'domain':<16}{'disp':>7}{'dispEmb':>9}{'tier':>5}{'sig_id':>8}{'EAS':>8}"
      f"{'END':>8}{'WVC':>8}  verdict", flush=True)
print("-" * 104, flush=True)

pm = AutoModel.from_pretrained(PARENT).eval()
done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip().startswith("{"):
            done.add(json.loads(l)["child"])
f = OUT.open("a")
for dom, child in CHILDREN:
    if child in done:
        print(f"{dom:<16} (cached)", flush=True); continue
    try:
        cm = AutoModel.from_pretrained(child).eval()
        disp = displacement(pm, cm)
        del cm
    except Exception as e:
        print(f"{dom:<16} LOAD FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True); continue
    tgt, repaired = repaired_path(child)
    r = mpk(PARENT, tgt)
    if r.get("error"):
        print(f"{dom:<16} MPK ERROR: {r['error'][:70]}", flush=True); continue
    sc, sg = r.get("scores", {}), r.get("signals", {})
    rec = dict(domain=dom, parent=PARENT, child=child, displacement=disp, scores=sc,
               signals=sg, config_repaired=repaired)
    f.write(json.dumps(rec) + "\n"); f.flush()
    nf = lambda x: float("nan") if x is None else x
    print(f"{dom:<16}{disp['overall']:>7.4f}{disp.get('embedding',float('nan')):>9.4f}"
          f"{sc.get('mfi_tier','?'):>5}{nf(sc.get('identity_score')):>8.4f}"
          f"{nf(sg.get('eas')):>8.4f}{nf(sg.get('end')):>8.4f}{nf(sg.get('wvc')):>8.4f}"
          f"  {sc.get('provenance_decision')}{'  [config repaired]' if repaired else ''}", flush=True)
f.close()
print("DONE-CORPUSDIST", flush=True)
