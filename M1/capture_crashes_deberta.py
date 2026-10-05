"""Run the pinned verifier end to end on the DeBERTa line that E14b found lacking `architectures`.

Review round 4 (Reviewer B, Q4): section VIII says the first-party Microsoft DeBERTa line "cannot
be verified", but E14b inspected configurations only (M7/e14b_hub_prevalence.py fetches
config.json and nothing else). This runs the tool on those models, with the same invocation and
the same defect signature as M1/capture_crashes.py.

Runs, all with --json --no-cache at the pinned commit:
  (a) `scan` on each of the nine microsoft/* DeBERTa repos E14c counts among the 18 (a
      database-wide lookup is the single-model question "can this model be verified at all");
  (b) `compare` of microsoft/deberta-v3-base against deepset/deberta-v3-base-squad2 in BOTH
      argument orders, to show the outcome does not depend on which slot the key-less model
      occupies (the second order downloads the squad2 child's weights before reaching the
      parent);
  (c) a control: `scan` on microsoft/deberta-base-mnli, a first-party DeBERTa release whose
      config DOES carry the key, to separate "this architecture is unsupported" from "this key
      is missing".

Writes a NEW file, M1/results/mpk_crashes_deberta.jsonl. Touches no existing result file.
"""
import os, json, pathlib, subprocess, hashlib, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))
REPO = _ROOT/"M1"/"oracle"/"model-provenance-kit"
OUT = _ROOT/"M1"/"results"/"mpk_crashes_deberta.jsonl"

MS = ["microsoft/mdeberta-v3-base", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large",
      "microsoft/deberta-v3-small", "microsoft/deberta-v3-xsmall", "microsoft/deberta-base",
      "microsoft/deberta-large", "microsoft/deberta-v2-xlarge", "microsoft/deberta-v2-xxlarge"]
RUNS = ([("scan", [m], "key absent") for m in MS] +
        [("compare", ["microsoft/deberta-v3-base", "deepset/deberta-v3-base-squad2"], "key-less model first"),
         ("compare", ["deepset/deberta-v3-base-squad2", "microsoft/deberta-v3-base"], "key-less model second"),
         ("scan", ["microsoft/deberta-base-mnli"], "control: key present")])

def config_of(repo):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "config.json")
    b = open(p, "rb").read()
    return json.loads(b), hashlib.sha256(b).hexdigest()[:16]

env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
env["HF_HUB_DISABLE_XET"] = "1"
f = OUT.open("w")
for mode, models, note in RUNS:
    cfgs = {m: config_of(m) for m in models}
    t0 = time.time()
    p = subprocess.run(["uv", "run", "provenancekit", mode, *models, "--json", "--no-cache"],
                       cwd=REPO, capture_output=True, text=True, timeout=7200, env=env)
    err = p.stderr or ""
    i = p.stdout.find("{")
    out = None
    if i >= 0:
        try: out = json.loads(p.stdout[i:])
        except Exception: out = None
    defect = ("list_type" in err) or ("MFIFingerprint" in err and "validation error" in err)
    rec = dict(mode=mode, models=models, note=note,
               config_sha256_16={m: c[1] for m, c in cfgs.items()},
               architectures_present={m: ("architectures" in c[0] and c[0]["architectures"] is not None)
                                      for m, c in cfgs.items()},
               mpk_exit=p.returncode, mpk_returned_json=out is not None, defect=bool(defect),
               stderr_tail=err.strip()[-400:] or None, secs=round(time.time() - t0, 1),
               mpk_version="1.1.0")
    if out is not None:
        sc = out.get("scores") or {}
        rec["verdict"] = sc.get("provenance_decision")
        rec["tier"] = sc.get("mfi_tier")
        rec["matches"] = [dict((k, m.get(k)) for k in ("model_id", "pipeline_score", "provenance_decision", "mfi_tier"))
                          for m in (out.get("matches") or [])][:5]
    f.write(json.dumps(rec) + "\n"); f.flush()
    print(f"{mode:<8}{' | '.join(models):<66} exit={p.returncode} defect={defect} "
          f"json={out is not None} ({rec['secs']}s)", flush=True)
f.close()
print("DONE-DEBERTA", flush=True)
