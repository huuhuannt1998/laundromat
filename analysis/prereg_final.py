"""Final preregistered family: C4 and C5 on the powered, shape-stratified arms, Holm-corrected."""
import json, pathlib, itertools, os, glob
import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
W = {"eas": 0.36, "end": 0.19, "lep": 0.16, "nlf": 0.08}       # structural, 0.79
CONTINUED = {"ehsanaghaei/SecureBERT", "cardiffnlp/twitter-roberta-base-sentiment-latest"}


def dims(repo):
    cs = sorted(glob.glob(str(CACHE / ("models--" + repo.replace("/", "--")) / "snapshots" / "*" / "config.json")))
    if not cs:
        return None
    c = json.load(open(cs[0]))
    return ((c.get("hidden_size") or c.get("n_embed") or c.get("n_embd") or c.get("d_model") or c.get("dim")),
            (c.get("num_hidden_layers") or c.get("n_layer") or c.get("n_layers") or c.get("num_layers")))


rows = [json.loads(l) for l in (R / "M4" / "power_fix.jsonl").read_text().splitlines()
        if l.strip().startswith("{")]
rows = [r for r in rows if not r.get("error")]
for r in rows:
    da, db = dims(r["a"]), dims(r["b"])
    r["same"] = bool(da and db and None not in da and da == db)
rows = [r for r in rows if r["same"]]


def s_struct(s):
    n = sum(W[k] * s[k] for k in W if s.get(k) is not None)
    d = sum(W[k] for k in W if s.get(k) is not None)
    return n / d if d else None


def perm_p(a, b):
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a)
    cs = list(itertools.combinations(range(len(pool)), na))
    ds = np.array([pool[list(c)].mean() - pool[[j for j in range(len(pool)) if j not in set(c)]].mean() for c in cs])
    return (np.sum(np.abs(ds) >= abs(obs) - 1e-12) + 1) / (len(ds) + 1)


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0]*len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps)-r)*ps[i]); adj[i] = min(1.0, run)
    return adj


print("=" * 92); print("FINAL PREREGISTERED FAMILY -- powered, shape-stratified, Holm-corrected"); print("=" * 92)
out, ps, tags = {}, [], []
for tag, fn, sub in (("C4  s_struct, P~ vs all P+", s_struct, None),
                     ("C5  s_raw/WVC, P~ vs all P+", lambda s: s.get("wvc"), None),
                     ("C5a s_raw/WVC, P~ vs fine-tune only", lambda s: s.get("wvc"), "ft")):
    a = np.array([v for r in rows if r["cls"] == "P~" and (v := fn(r["signals"])) is not None])
    b = np.array([v for r in rows if r["cls"] == "P+" and (v := fn(r["signals"])) is not None
                  and (sub != "ft" or r["b"] not in CONTINUED)])
    if len(a) < 2 or len(b) < 2:
        print(f"\n{tag}: insufficient (n={len(a)},{len(b)})"); continue
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = 1 - U/(len(a)*len(b))
    p = perm_p(a, b)
    print(f"\n{tag}")
    print(f"    P~ {a.mean():.4f} (n={len(a)})   P+ {b.mean():.4f} (n={len(b)})   AUC(P+>P~) {auc:.3f}")
    print(f"    exact permutation p = {p:.5f}")
    out[tag] = dict(pt=a.tolist(), pp=b.tolist(), auc=auc, p=p)
    if not tag.startswith("C5a"):          # Holm family = the 2 preregistered contrasts
        ps.append(p); tags.append(tag)

adj = holm(ps)
print("\n" + "-" * 92)
print("HOLM over the preregistered inferential family {C4, C5}, m=2:")
for t, p, q in zip(tags, ps, adj):
    print(f"    {t.split('  ')[0]:<5} raw p={p:.5f}   Holm p={q:.5f}   {'REJECT H0' if q < 0.05 else 'retain H0'}")
print("\n    C5a (fine-tune only) is a STRATIFIED FOLLOW-UP, not a preregistered contrast;")
print("    it is reported uncorrected and labelled exploratory.")
json.dump(out, open(R / "analysis" / "prereg_final.json", "w"), indent=1)
print("\nwrote analysis/prereg_final.json")
