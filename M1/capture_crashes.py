"""Capture a real, structured record of the architectures-key denial-of-verification defect.

Review finding C-10: sections/08-denial.tex cites M1/results/mpk_crashes.jsonl as the
artifact of record for the paper's most quotable claim, and that file is 0 bytes. Two of
the six table rows had no captured crash anywhere. This script produces the evidence:
for each candidate it records whether config.json carries an 'architectures' key, then
runs MPK and captures the actual exit status and stderr signature.

Writes NEW file M1/results/mpk_crashes.jsonl. Touches no existing result file.
"""
import os, sys, json, glob, pathlib, subprocess, hashlib
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

REPO = pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
OUT = pathlib.Path(str(_ROOT) + "/M1/results/mpk_crashes.jsonl")
PARENT_FOR = {
    "microsoft/MiniLM-L12-H384-uncased":        "google-bert/bert-base-uncased",
    "nreimers/TinyBERT_L-4_H-312_v2":           "google-bert/bert-base-uncased",
    "huawei-noah/TinyBERT_General_4L_312D":     "google-bert/bert-base-uncased",
    "google/bert_uncased_L-4_H-256_A-4":        "google-bert/bert-base-uncased",
    "allenai/cs_roberta_base":                  "FacebookAI/roberta-base",
    "allenai/biomed_roberta_base":              "FacebookAI/roberta-base",
    "allenai/news_roberta_base":                "FacebookAI/roberta-base",
    "allenai/reviews_roberta_base":             "FacebookAI/roberta-base",
    "emilyalsentzer/Bio_ClinicalBERT":          "google-bert/bert-base-uncased",
    "prajjwal1/bert-tiny":                      "google-bert/bert-base-uncased",
}


def config_of(repo):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, "config.json")
        return json.load(open(p)), hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except Exception as e:
        return {"__fetch_error__": f"{type(e).__name__}: {str(e)[:120]}"}, None


f = OUT.open("w")
print(f"{'model':<44}{'arch key':>12}{'exit':>6}  signature", flush=True)
print("-" * 104, flush=True)
for child, parent in PARENT_FOR.items():
    cfg, sha = config_of(child)
    if "__fetch_error__" in cfg:
        rec = dict(child=child, parent=parent, config_sha256_16=None,
                   architectures_present=None, architectures_value=None,
                   mpk_exit=None, defect=False, note=cfg["__fetch_error__"])
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{child:<44}{'FETCH-ERR':>12}{'-':>6}  {cfg['__fetch_error__'][:44]}", flush=True)
        continue
    arch = cfg.get("architectures", "__ABSENT__")
    present = arch not in (None, "__ABSENT__")
    p = subprocess.run(["uv", "run", "provenancekit", "compare", parent, child, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=3600,
                       env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    err = (p.stderr or "")
    got_json = p.stdout.find("{") >= 0
    # the defect signature is a pydantic list_type validation failure on MFIFingerprint
    defect = ("list_type" in err) or ("MFIFingerprint" in err and "validation error" in err)
    sig = ("MFIFingerprint/list_type" if defect else
           ("ok" if got_json else (err.strip().splitlines()[-1][:60] if err.strip() else "no-json")))
    rec = dict(child=child, parent=parent, config_sha256_16=sha,
               architectures_present=present,
               architectures_value=(None if arch == "__ABSENT__" else arch),
               architectures_absent=(arch == "__ABSENT__"),
               mpk_exit=p.returncode, mpk_returned_json=got_json,
               defect=bool(defect), stderr_tail=err.strip()[-400:] or None,
               mpk_version="1.1.0")
    f.write(json.dumps(rec) + "\n"); f.flush()
    shown = "ABSENT" if arch == "__ABSENT__" else ("null" if arch is None else "present")
    print(f"{child:<44}{shown:>12}{p.returncode:>6}  {sig}", flush=True)
f.close()
print("DONE-CRASHCAPTURE", flush=True)
