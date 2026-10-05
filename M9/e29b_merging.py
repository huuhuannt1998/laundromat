"""E29b -- merging as zero-compute laundering, with the capability instrument repaired.

What was wrong with E29 (M9/e29_merging.py -> e29_merging.jsonl). Two defects, found by the
2026-09-08 review and by re-running the instrument on an untouched model:
  1. The BERT arm's child, textattack/bert-base-uncased-SST-2, is a classification
     checkpoint with no masked-LM head, so `mlm_loss_base` (10.547 nats) is the loss of a
     randomly initialised head, i.e. chance (ln 30522 = 10.33).
  2. `write_model` materialised every merged model through `AutoModel`, which saves the
     encoder only. Reloading through `AutoModelForMaskedLM` then attaches a NEW random head.
     Reproduced here without any merge: cardiffnlp/twitter-roberta-base reads 2.46 nats loaded
     directly and 15.83 nats after an AutoModel save/load round trip. Every `mlm_loss` in
     e29_merging.jsonl, both families, is therefore the loss of a random head, and the
     "12.5 nats" cost the manuscript quoted for the RoBERTa amplification arm measures the
     instrument, not the merge.
Same failure class as CORRECTIONS.md #10 (a capability proxy that was never validated).

This run: (a) the BERT child is GroNLP/hateBERT, whose card states it was "obtained by
further training the English BERT base uncased model" and which ships a BertForMaskedLM head;
(b) merges are applied to the encoder INSIDE the child's full masked-LM model, so the child's
own head is carried through and every arm is scored with a real head; (c) a diagnostic row
reproduces defect 2 on the unmerged RoBERTa child. Transform plans, ratios, predeclared bar,
LAP comparator and MPK invocation are unchanged from E29. Output: M9/e29b_merging.jsonl.
"""
import os, sys, gc, json, time, shutil, pathlib, subprocess
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch, numpy as np
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e29b"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = ROOT / "M9" / "e29b_merging.jsonl"

BAR = 0.5298
MAX_FEAT = 1536
KEYS = ("intermediate.dense.weight", "up_proj.weight", "dense_h_to_4h.weight",
        "c_fc.weight", "fc_in.weight", "ffn.lin1.weight")

FAMILIES = {
  "bert": {
    "parent": "google-bert/bert-base-uncased",
    "child": "GroNLP/hateBERT",                       # MLM-headed continued-pretraining child
    "sibs": ["textattack/bert-base-uncased-SST-2", "textattack/bert-base-uncased-imdb"],
    "independent": "nlpaueb/legal-bert-base-uncased",
  },
  "roberta": {
    "parent": "FacebookAI/roberta-base",
    "child": "cardiffnlp/twitter-roberta-base",       # MLM-headed, as in E29
    "sibs": ["ehsanaghaei/SecureBERT", "allenai/biomed_roberta_base"],
    "independent": None,
  },
}
PROBE = ["The model was released under a permissive licence.",
         "Supply chain security depends on verifiable provenance.",
         "A derivative inherits obligations from its parent."]

def base_sd(mid):
    m = AutoModel.from_pretrained(mid, low_cpu_mem_usage=True).eval()
    d = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del m; gc.collect(); return d

def common(*dicts):
    ks = set(dicts[0])
    for d in dicts[1:]: ks &= set(d)
    return {k for k in ks if all(d[k].shape == dicts[0][k].shape for d in dicts)}

def write_mlm_model(child_id, new_base_sd, tag):
    """Materialise the merged encoder INSIDE the child's own masked-LM model, head included."""
    out = WORK / tag
    if (out / "config.json").exists(): return out
    m = AutoModelForMaskedLM.from_pretrained(child_id, low_cpu_mem_usage=True).eval()
    base = getattr(m, m.base_model_prefix)
    tgt = base.state_dict()
    # AutoModel adds a pooler the MLM model does not carry; load only what the target owns.
    res = base.load_state_dict({k: v for k, v in new_base_sd.items() if k in tgt}, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys, res.unexpected_keys)
    m.save_pretrained(out); AutoTokenizer.from_pretrained(child_id).save_pretrained(out)
    del m; gc.collect(); return out

def mlm_loss(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForMaskedLM.from_pretrained(path, low_cpu_mem_usage=True).eval()
    try:
        tot = 0.0
        for s in PROBE:
            enc = tok(s, return_tensors="pt")
            tot += float(m(**enc, labels=enc["input_ids"].clone()).loss)
        return tot / len(PROBE)
    finally:
        del m; gc.collect()

def mpk(a, b):
    p = subprocess.run(["uv", "run", "provenancekit", "compare", str(a), str(b),
                        "--json", "--no-cache"], cwd=REPO, capture_output=True,
                       text=True, timeout=7200, env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    i = p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i >= 0 else {"error": (p.stderr or "")[-300:]}

def rows(dct):
    out = []
    for k, v in dct.items():
        if v.ndim == 2 and any(x in k for x in KEYS):
            A = v.float()
            if A.shape[1] > MAX_FEAT:
                f = torch.arange(0, A.shape[1], max(1, A.shape[1] // MAX_FEAT))[:MAX_FEAT]
                A = A[:, f]
            out.append(A.numpy())
    return out

def lap(a_rows, b_rows):
    vals = []
    for A, B in zip(a_rows, b_rows):
        if A.shape != B.shape: continue
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        S = A @ B.T; r, c = linear_sum_assignment(-S)
        vals.append(float(S[r, c].mean()))
    return float(np.mean(vals)) if vals else float("nan")

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    done = set()
    if OUT.exists():
        for l in OUT.open():
            if l.strip(): done.add(json.loads(l)["tag"])
    with OUT.open("a") as fh:
        # ---- diagnostic: reproduce the E29 instrument defect on an unmerged child ----
        if "diag-roberta-headdrop" not in done and (only in (None, "roberta")):
            cid = FAMILIES["roberta"]["child"]
            direct = mlm_loss(cid)
            d = WORK / "diag_headdrop"
            if not (d / "config.json").exists():
                m = AutoModel.from_pretrained(cid).eval(); m.save_pretrained(d)
                AutoTokenizer.from_pretrained(cid).save_pretrained(d); del m; gc.collect()
            dropped = mlm_loss(d)
            rec = {"tag": "diag-roberta-headdrop", "family": "roberta", "kind": "instrument diagnostic",
                   "child": cid, "mlm_loss_direct": direct, "mlm_loss_after_automodel_roundtrip": dropped,
                   "note": "E29 saved every merged model through AutoModel; this is what that did to an unmerged child"}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  diagnostic: direct {direct:.4f} nats, after AutoModel round trip {dropped:.4f} nats", flush=True)
            shutil.rmtree(d, ignore_errors=True)
        for fam, cfg in FAMILIES.items():
            if only and only != fam: continue
            P, C, sibs = cfg["parent"], cfg["child"], cfg["sibs"]
            print(f"\n=== {fam}: parent {P}, child {C} ===", flush=True)
            p_sd = base_sd(P); p_rows = rows(p_sd)
            c_sd = base_sd(C)
            base_loss = mlm_loss(C); parent_loss = mlm_loss(P)
            print(f"  child MLM loss {base_loss:.4f} nats; parent {parent_loss:.4f} nats", flush=True)
            if f"{fam}-unmerged" not in done:
                r = mpk(C, P)
                if "error" not in r:
                    rec = {"tag": f"{fam}-unmerged", "family": fam, "kind": "unmerged child (control)",
                           "child": C, "parent": P, "scores": r["scores"], "signals": r["signals"],
                           "lap_vs_parent": lap(rows(c_sd), p_rows), "mlm_loss": base_loss,
                           "mlm_loss_base": base_loss, "mlm_loss_parent": parent_loss,
                           "capability_delta": 0.0, "evades_predeclared_bar": False}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    s = r["scores"]
                    print(f"  {'unmerged':22} pipe={s['pipeline_score']:.4f} tier={s['mfi_tier']} "
                          f"sig={s['identity_score']:.4f} LAP={rec['lap_vs_parent']:.4f}", flush=True)
            plans = []
            for a in (0.25, 0.5, 0.75):
                plans.append((f"{fam}-merge{a}", "sibling merge", lambda A=a, s2=sibs[0]: (c_sd, base_sd(s2), A, "lerp")))
            for lam in (1.5, 2.0, 3.0):
                plans.append((f"{fam}-amp{lam}", "delta amplification", lambda L=lam: (c_sd, p_sd, L, "amp")))
            plans.append((f"{fam}-task", "task arithmetic", lambda s2=sibs[1]: (c_sd, base_sd(s2), 1.0, "task")))
            if cfg["independent"]:
                for a in (0.1, 0.3):
                    plans.append((f"{fam}-indep{a}", "independent contamination",
                                  lambda A=a, I=cfg["independent"]: (c_sd, base_sd(I), A, "lerp")))
            for tag, kind, build in plans:
                if tag in done: print(f"  [skip] {tag}"); continue
                t0 = time.time()
                d1, d2, k, mode = build()
                ks = common(d1, d2)
                new = {}
                for key in d1:
                    if key not in ks: new[key] = d1[key]; continue
                    a1, a2 = d1[key].float(), d2[key].float()
                    if mode == "lerp":   v = (1 - k) * a1 + k * a2
                    elif mode == "amp":  v = a2 + k * (a1 - a2)
                    else:
                        v = a1 + (a2 - p_sd[key].float()) if (key in p_sd and p_sd[key].shape == a1.shape) else a1
                    new[key] = v.to(d1[key].dtype)
                path = write_mlm_model(C, new, tag)
                r = mpk(path, P)
                if "error" in r:
                    print(f"  {tag}: MPK ERROR {r['error'][:120]}", flush=True); continue
                s, g = r["scores"], r["signals"]
                loss = mlm_loss(path)
                cap = round(loss - base_loss, 4)
                al = lap(rows(new), p_rows)
                evades = (s["pipeline_score"] <= BAR) and (abs(cap) < 1.0)
                rec = {"tag": tag, "family": fam, "kind": kind, "child": C, "parent": P,
                       "scores": s, "signals": g, "lap_vs_parent": al,
                       "mlm_loss": loss, "mlm_loss_base": base_loss, "mlm_loss_parent": parent_loss,
                       "capability_delta": cap, "evades_predeclared_bar": bool(evades),
                       "seconds": round(time.time()-t0, 1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  {tag:22}{kind:26} pipe={s['pipeline_score']:.4f} tier={s['mfi_tier']} "
                      f"sig={s['identity_score']:.4f} LAP={al:.4f} dMLM={cap:+.3f} "
                      f"{'EVADES' if evades else ''}", flush=True)
                shutil.rmtree(path, ignore_errors=True)
                del new; gc.collect()
            del p_sd, c_sd, p_rows; gc.collect()
    print("DONE-E29B")

if __name__ == "__main__":
    main()
