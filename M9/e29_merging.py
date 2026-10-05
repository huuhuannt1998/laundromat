"""E29 -- merging and delta arithmetic as zero-compute laundering (mock review W-A2).

WHY. Section VI's catalogue is five families of algebraic identity: permutations,
compensated rescalings, vocab remap, quantisation, noise. Reviewer A's objection is
that no launderer would expect an identity to hide descent from a weight-reading
verifier, and that the operation they WOULD use is merging -- linear interpolation
with a sibling, task arithmetic, delta amplification -- which moves weights a long way,
costs seconds, and is absent from the catalogue. Until it is tested, "no zero-compute
transform evades" is a claim about SYMMETRIES, not about zero-compute laundering.

It also stress-tests section IX: if a 50/50 sibling merge defeats the LAP alignment
signal, the defence's claim shrinks to "separates recipe from single-parent descent",
which is still publishable but must be stated.

Predeclared criterion, unchanged from section III: an attempt succeeds only if the
pipeline score falls at or below the frozen null's 75th percentile, 0.5298, AND the
transform preserves capability. Capability here is masked-LM loss on a fixed probe
relative to the source child; models with no LM head are reported `unmeasured` and are
NOT counted as evasions (the E15 lesson: absence of a fidelity record is not evidence
of preserved capability).

Everything is cached and encoder-scale, so this needs ~4 GB, not 16.
Run `preflight.py --need 5` first.
"""
import os, sys, gc, json, time, shutil, pathlib, subprocess, itertools
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch, numpy as np
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False)

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
            if (p / "MANIFEST-dataintegrity.txt").exists())
REPO = ROOT / "M1/oracle/model-provenance-kit"
WORK = ROOT / "M9" / "work_e29"; WORK.mkdir(parents=True, exist_ok=True)
OUT  = ROOT / "M9" / "e29_merging.jsonl"

BAR = 0.5298                      # frozen-null p75, section III
MAX_FEAT = 1536
KEYS = ("intermediate.dense.weight", "up_proj.weight", "dense_h_to_4h.weight",
        "c_fc.weight", "fc_in.weight", "ffn.lin1.weight")

FAMILIES = {
  "bert": {
    "parent": "google-bert/bert-base-uncased",
    "sibs": ["textattack/bert-base-uncased-SST-2", "textattack/bert-base-uncased-imdb",
             "csarron/bert-base-uncased-squad-v1", "ProsusAI/finbert"],
    "independent": "nlpaueb/legal-bert-base-uncased",
  },
  "roberta": {
    "parent": "FacebookAI/roberta-base",
    "sibs": ["cardiffnlp/twitter-roberta-base", "ehsanaghaei/SecureBERT",
             "allenai/biomed_roberta_base", "allenai/cs_roberta_base"],
    "independent": None,
  },
}
PROBE = ["The model was released under a permissive licence.",
         "Supply chain security depends on verifiable provenance.",
         "A derivative inherits obligations from its parent."]

def sd(mid):
    m = AutoModel.from_pretrained(mid, low_cpu_mem_usage=True).eval()
    d = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del m; gc.collect(); return d

def common(*dicts):
    ks = set(dicts[0])
    for d in dicts[1:]: ks &= set(d)
    return {k for k in ks if all(d[k].shape == dicts[0][k].shape for d in dicts)}

def write_model(base_id, new_sd, tag):
    """Materialise a merged model MPK can read: child's config+tokenizer, merged weights."""
    out = WORK / tag
    if (out / "config.json").exists(): return out
    m = AutoModel.from_pretrained(base_id, low_cpu_mem_usage=True).eval()
    missing = m.load_state_dict(new_sd, strict=False)
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
    del m; gc.collect(); return out

def mlm_loss(path):
    """Capability probe. Returns None when the checkpoint has no LM head."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForMaskedLM.from_pretrained(path, low_cpu_mem_usage=True).eval()
    except Exception:
        return None
    try:
        tot = 0.0
        for s in PROBE:
            enc = tok(s, return_tensors="pt")
            labels = enc["input_ids"].clone()
            out = m(**enc, labels=labels)
            tot += float(out.loss)
        return tot / len(PROBE)
    except Exception:
        return None
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
        for fam, cfg in FAMILIES.items():
            if only and only != fam: continue
            P = cfg["parent"]; sibs = cfg["sibs"]
            print(f"\n=== {fam}: parent {P} ===", flush=True)
            p_sd = sd(P); p_rows = rows(p_sd)
            base_child = sibs[0]
            c_sd = sd(base_child)
            base_loss = None

            plans = []
            # 1. sibling merge at three ratios
            for a in (0.25, 0.5, 0.75):
                plans.append((f"{fam}-merge{a}", "sibling merge",
                              lambda A=a, s2=sibs[1]: (c_sd, sd(s2), A, "lerp")))
            # 2. delta amplification
            for lam in (1.5, 2.0, 3.0):
                plans.append((f"{fam}-amp{lam}", "delta amplification",
                              lambda L=lam: (c_sd, p_sd, L, "amp")))
            # 3. task arithmetic: child + (sibling - parent)
            plans.append((f"{fam}-task", "task arithmetic",
                          lambda s2=sibs[2]: (c_sd, sd(s2), 1.0, "task")))
            # 4. contamination with an independent same-architecture model
            if cfg["independent"]:
                for a in (0.1, 0.3):
                    plans.append((f"{fam}-indep{a}", "independent contamination",
                                  lambda A=a, I=cfg["independent"]: (c_sd, sd(I), A, "lerp")))

            for tag, kind, build in plans:
                if tag in done: print(f"  [skip] {tag}"); continue
                t0 = time.time()
                d1, d2, k, mode = build()
                ks = common(d1, d2)
                new = {}
                for key in d1:
                    if key not in ks: new[key] = d1[key]; continue
                    a1, a2 = d1[key].float(), d2[key].float()
                    if mode == "lerp":
                        v = (1 - k) * a1 + k * a2                       # (1-a)*child + a*other
                    elif mode == "amp":
                        v = a2 + k * (a1 - a2)                          # parent + lambda*(child-parent)
                    else:                                               # task arithmetic
                        if key in p_sd and p_sd[key].shape == a1.shape:
                            v = a1 + (a2 - p_sd[key].float())           # child + (sibling - parent)
                        else:
                            v = a1
                    new[key] = v.to(d1[key].dtype)
                path = write_model(base_child, new, tag)
                r = mpk(path, P)
                if "error" in r:
                    print(f"  {tag}: MPK ERROR {r['error'][:120]}", flush=True); continue
                s, g = r["scores"], r["signals"]
                loss = mlm_loss(path)
                if base_loss is None: base_loss = mlm_loss(base_child)
                cap = None if (loss is None or base_loss is None) else round(loss - base_loss, 4)
                al = lap(rows(new), p_rows)
                evades = (s["pipeline_score"] <= BAR) and (cap is not None and abs(cap) < 1.0)
                rec = {"tag": tag, "family": fam, "kind": kind, "child": base_child, "parent": P,
                       "scores": s, "signals": g, "lap_vs_parent": al,
                       "mlm_loss": loss, "mlm_loss_base": base_loss, "capability_delta": cap,
                       "evades_predeclared_bar": bool(evades), "seconds": round(time.time()-t0,1)}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                capstr = "unmeasured" if cap is None else f"{cap:+.3f}"
                print(f"  {tag:22}{kind:26} pipe={s['pipeline_score']:.4f} tier={s['mfi_tier']} "
                      f"sig={s['identity_score']:.4f} LAP={al:.4f} dMLM={capstr} "
                      f"{'EVADES' if evades else ''}", flush=True)
                shutil.rmtree(path, ignore_errors=True)     # disk hygiene
                del new; gc.collect()
            del p_sd, c_sd, p_rows; gc.collect()
    print("DONE-E29")

if __name__ == "__main__":
    main()
