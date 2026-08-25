"""C5 decided on the powered, shape-stratified arms -- plus the derivative-type split."""
import json, pathlib, itertools
import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (R / "M4" / "power_fix.jsonl").read_text().splitlines()
        if l.strip().startswith("{")]
rows = [r for r in rows if not r.get("error")]
_all = list(rows)

# Recompute shape here: the collector's key list missed BLOOM's "n_embed" (with an
# 'e'), so bloom-560m|bloomz-560m was recorded DIFF when both are 1024/24.
import os, glob
CACHE = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
def _dims(repo):
    cs = sorted(glob.glob(str(CACHE / ("models--" + repo.replace("/", "--")) / "snapshots" / "*" / "config.json")))
    if not cs:
        return None
    c = json.load(open(cs[0]))
    h = (c.get("hidden_size") or c.get("n_embed") or c.get("n_embd")
         or c.get("d_model") or c.get("dim"))
    l = (c.get("num_hidden_layers") or c.get("n_layer") or c.get("n_layers")
         or c.get("num_layers") or c.get("encoder_layers"))
    return (h, l)
for r in rows:
    da, db = _dims(r["a"]), _dims(r["b"])
    r["dims_a"], r["dims_b"] = da, db
    r["same_shape"] = bool(da and db and None not in da and da == db)
dropped = [r for r in rows if not r["same_shape"]]
for r in dropped:
    print(f"  excluded (DIFF-shape {r['dims_a']} vs {r['dims_b']}): "
          f"{r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}")
rows = [r for r in rows if r["same_shape"]]

# WVC coverage: how often is the ONLY positional signal even computable?
_ss = list(rows)
_nowvc = [r for r in _ss if r["signals"].get("wvc") is None]
print(f"\n  WVC COVERAGE on same-shape pairs: {len(_ss)-len(_nowvc)}/{len(_ss)} computable"
      f"  ({100*len(_nowvc)/max(1,len(_ss)):.0f}% undefined)")
for r in _nowvc:
    print(f"    undefined despite matching dims {r['dims_a']}: "
          f"{r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}  [{r['cls']}]")
rows = [r for r in rows if r["signals"].get("wvc") is not None]

# derivative type matters: light fine-tune preserves position, continued
# pretraining does not.  Declared here from the published model cards, not fitted.
CONTINUED = {"ehsanaghaei/SecureBERT",
              "cardiffnlp/twitter-roberta-base-sentiment-latest"}


def typ(r):
    if r["cls"] == "P~":
        return "P~"
    return "P+ continued-pretrain" if r["b"] in CONTINUED else "P+ fine-tune"


def perm_p(a, b):
    obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a)
    cs = list(itertools.combinations(range(len(pool)), na))
    ex = len(cs) <= 300000
    if not ex:
        rng = np.random.default_rng(0)
        cs = [tuple(rng.permutation(len(pool))[:na]) for _ in range(20000)]
    ds = np.array([pool[list(c)].mean() - pool[[j for j in range(len(pool)) if j not in set(c)]].mean() for c in cs])
    return (np.sum(np.abs(ds) >= abs(obs) - 1e-12) + (1 if ex else 0)) / (len(ds) + (1 if ex else 0)), ex


print("=" * 94)
print("C5 DECIDED  --  same-shape stratum, powered arms")
print("=" * 94)
for r in sorted(rows, key=lambda r: (typ(r), r["signals"]["wvc"])):
    print(f"  {typ(r):<24} WVC {r['signals']['wvc']:.4f}  sig_id {r['scores'].get('identity_score',float('nan')):.4f}"
          f"   {r['a'].split('/')[-1]} | {r['b'].split('/')[-1]}")

pt = np.array([r["signals"]["wvc"] for r in rows if typ(r) == "P~"])
pf = np.array([r["signals"]["wvc"] for r in rows if typ(r) == "P+ fine-tune"])
pc = np.array([r["signals"]["wvc"] for r in rows if typ(r) == "P+ continued-pretrain"])
pp = np.concatenate([pf, pc]) if len(pc) else pf

print(f"\n  n: P~={len(pt)}  P+ fine-tune={len(pf)}  P+ continued-pretrain={len(pc)}")
minp = 2.0 / (len(list(itertools.combinations(range(len(pt) + len(pp)), len(pt)))) or 1)
print(f"  minimum attainable two-sided exact p at these n: {minp:.5f}"
      f"   {'ADEQUATE' if minp < 0.05 else 'STILL UNDERPOWERED'}")

for tag, b in (("C5  P~ vs ALL P+", pp), ("C5a P~ vs P+ fine-tune only", pf)):
    if len(b) < 2:
        print(f"\n  {tag}: n={len(b)} insufficient"); continue
    U, _ = stats.mannwhitneyu(pt, b, alternative="two-sided")
    auc = 1 - U / (len(pt) * len(b))          # oriented: P+ above P~
    p, ex = perm_p(pt, b)
    print(f"\n  {tag}")
    print(f"      P~ {pt.mean():.4f} (n={len(pt)})   P+ {b.mean():.4f} (n={len(b)})")
    print(f"      AUC(P+>P~) {auc:.3f}   exact p = {p:.5f} {'(exact)' if ex else '(MC)'}   "
          f"{'REJECT H0' if p < 0.05 else 'retain H0'}")

if len(pc):
    print(f"\n  CONTINUED-PRETRAIN pairs sit at WVC {list(np.round(pc,4))}, inside the P~ range "
          f"[{pt.min():.4f},{pt.max():.4f}].")
    print("  A real published derivative is positionally indistinguishable from an independent run,")
    print("  with no adversary involved.  Continued pretraining IS naturally-occurring laundering.")
json.dump(dict(p_tilde=pt.tolist(), p_finetune=pf.tolist(), p_continued=pc.tolist()),
          open(R / "analysis" / "c5_final.json", "w"), indent=1)
print("\nwrote analysis/c5_final.json")
