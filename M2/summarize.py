"""M2 consolidated tables: gate arms + the embedding exchange-rate curve."""
import json, pathlib, statistics as st, glob

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))

R=pathlib.Path(str(_ROOT) + "/M2/results")
NULL=json.loads((R/"null_frozen.json").read_text())
W={"eas":.36,"nlf":.08,"lep":.16,"end":.19,"wvc":.21}
q=NULL["quantiles"]
print("="*112); print("FROZEN NULL"); print("="*112)
print(f"|N|={NULL['n']}  " + "  ".join(f"p{k}={v}" for k,v in q.items()))
FLOOR=W['eas']+W['end']
print(f"\nANALYTIC FLOOR  w_EAS+w_END = {FLOOR:.2f}  vs  pi*=75 cut = {q['75']}  ->  "
      f"{'E=1 UNREACHABLE for embedding-preserving exact transforms' if FLOOR>q['75'] else 'reachable'}")

for f in sorted(glob.glob(str(R/"gate_*.jsonl"))):
    rows=[json.loads(l) for l in pathlib.Path(f).read_text().splitlines()]
    print("\n"+"="*112); print("GATE G1 —", pathlib.Path(f).stem.replace("gate_","")); print("="*112)
    print(f"{'case':<30}{'tier':>5}{'pipe':>8}{'sig_id':>8}{'EAS':>8}{'NLF':>8}{'LEP':>8}{'END':>8}{'WVC':>8}"
          f"{'rank':>6}{'pi':>7}{'exact':>7}{'E':>3}")
    for d in rows:
        c=d["cmp"] or {}; s=d["signals"] or {}; fi=d["fidelity"]
        print(f"{d['name']:<30}{str(c.get('mfi_tier')):>5}{c.get('pipeline_score',0):>8.4f}"
              f"{c.get('identity_score',0):>8.4f}{(s.get('eas') or 0):>8.4f}{(s.get('nlf') or 0):>8.4f}"
              f"{(s.get('lep') or 0):>8.4f}{(s.get('end') or 0):>8.4f}{(s.get('wvc') or 0):>8.4f}"
              f"{str(d['parent_rank']):>6}{(d['pi'] or 0):>7.1f}"
              f"{str(fi['exact_within_tolerance']):>7}{d['E']:>3}")

for f in sorted(glob.glob(str(R/"sweep_embed_*.jsonl"))):
    rows=[json.loads(l) for l in pathlib.Path(f).read_text().splitlines()]
    print("\n"+"="*112); print("EXIT-2 EMBEDDING EXCHANGE RATE —", pathlib.Path(f).stem.replace("sweep_embed_",""))
    print("="*112)
    print(f"{'rel_sigma':>10}{'EAS':>8}{'END':>8}{'WVC':>8}{'sig_id':>8}{'pipe':>8}{'rank':>6}{'pi':>7}"
          f"{'ppl':>9}{'rho_ppl':>9}{'kappa':>7}{'E':>3}")
    for d in rows:
        s=d["signals"] or {}; c=d["scores"] or {}; fi=d["fid"]
        print(f"{d['rel_sigma']:>10.3f}{(s.get('eas') or 0):>8.4f}{(s.get('end') or 0):>8.4f}"
              f"{(s.get('wvc') or 0):>8.4f}{c.get('identity_score',0):>8.4f}{c.get('pipeline_score',0):>8.4f}"
              f"{str(d['rank']):>6}{(d['pi'] or 0):>7.1f}{fi['ppl_new']:>9.3f}{(d['rho_ppl'] or 0):>9.4f}"
              f"{d['kappa']:>7.1f}{d['E']:>3}")
    ok=[d for d in rows if d["E"]==1]
    good=[d for d in rows if d["E"]==1 and (d["rho_ppl"] or 0)>=0.98]
    print(f"\n  E=1 at any cost: {len(ok)}/{len(rows)}   E=1 with rho>=0.98: {len(good)}/{len(rows)}"
          f"  -> {'EXIT 1/2' if good else 'EXIT 3 for this arm'}")
