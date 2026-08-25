"""Price C1 on the displacement axis.

C1: "fully evading all weight-level signals is as expensive as training from scratch."
We now have, for one parent and many benign children, both the detector's verdict and
an independent measure of how far training moved the weights.  If the detector fails
well below the scratch-training displacement, C1 is false at kappa > 0 -- and the
counterexamples are published models nobody built to evade anything.
"""
import json, pathlib
import numpy as np

R = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in (R / "corpus_distance.jsonl").read_text().splitlines()
        if l.strip().startswith("{")]
ref = json.load(open(R / "displacement_reference.json")) if (R / "displacement_reference.json").exists() else []

WEAK, HIGH, BAR = 0.65, 0.75, 0.5298
print("=" * 104)
print("CORPUS DISTANCE vs THE DETECTOR   (parent = roberta-base, all children benign & published)")
print("=" * 104)
print(f"{'domain':<18}{'disp':>7}{'dispEmb':>9}{'sig_id':>8}{'EAS':>8}{'END':>8}{'WVC':>8}  verdict")
print("-" * 104)
D, S = [], []
for r in sorted(rows, key=lambda r: r["displacement"]["overall"]):
    d = r["displacement"]["overall"]; sc = r["scores"]; sg = r["signals"]
    sid = sc.get("identity_score")
    nf = lambda x: float("nan") if x is None else x
    D.append(d); S.append(nf(sid))
    flag = "  <-- FALSE NEGATIVE" if sc.get("provenance_decision") == "Not Matched" else ""
    print(f"{r['domain']:<18}{d:>7.4f}{r['displacement'].get('embedding',float('nan')):>9.4f}"
          f"{nf(sid):>8.4f}{nf(sg.get('eas')):>8.4f}{nf(sg.get('end')):>8.4f}{nf(sg.get('wvc')):>8.4f}"
          f"  {sc.get('provenance_decision')}{flag}")

if ref:
    rd = [x["displacement"] for x in ref]
    print("\nREFERENCE POINTS (empirical 'trained from scratch'):")
    for x in ref:
        print(f"  {x['displacement']:.4f}  {x['a'].split('/')[-1]} | {x['b'].split('/')[-1]}  [{x['kind']}]")
    scratch = float(np.mean(rd))
    print(f"  mean independent-run displacement = {scratch:.4f}   (analytic uncorrelated ~1.4142)")
else:
    scratch = float("nan")

E = np.array([r["displacement"].get("embedding", np.nan) for r in
              sorted(rows, key=lambda r: r["displacement"]["overall"])])
D, S = np.array(D), np.array(S)
ok = ~np.isnan(S)


def fit(x, y):
    m, c = np.polyfit(x, y, 1)
    r2 = 1 - ((y-(m*x+c))**2).sum()/((y-y.mean())**2).sum()
    return m, c, r2


if ok.sum() >= 3:
    # which displacement predicts the detector -- overall, or the embedding alone?
    mo, co, r2o = fit(D[ok], S[ok])
    oke = ok & ~np.isnan(E)
    me, ce, r2e = (fit(E[oke], S[oke]) if oke.sum() >= 3 else (np.nan,)*3)
    print(f"\nWHICH DISPLACEMENT PREDICTS THE VERDICT?")
    print(f"  overall    sigma_id = {co:.4f} {mo:+.4f}*d   R^2 = {r2o:.4f}  (n={ok.sum()})")
    print(f"  embedding  sigma_id = {ce:.4f} {me:+.4f}*d   R^2 = {r2e:.4f}  (n={oke.sum()})")
    use_emb = (r2e == r2e) and (r2e > r2o)
    print(f"  -> {'EMBEDDING' if use_emb else 'OVERALL'} displacement is the better predictor"
          f"{'  (consistent with EAS/END reading only the embedding)' if use_emb else ''}")
    m, c, r2 = (me, ce, r2e) if use_emb else (mo, co, r2o)
    print(f"\nsigma_id vs displacement:  sigma_id = {c:.4f} {m:+.4f}*d   (R^2 = {r2:.4f})")
    for lbl, thr in (("WEAK_MATCH 0.65", WEAK), ("evasion bar 0.5298", BAR)):
        dc = (thr - c) / m if m else float("nan")
        rel = dc / scratch if scratch == scratch else float("nan")
        print(f"  crosses {lbl:<20} at displacement {dc:.4f}"
              + (f"  = {100*rel:.0f}% of a from-scratch run" if rel == rel else ""))
    print("\nC1 VERDICT ON THIS AXIS:")
    dc = (BAR - c) / m if m else float("nan")
    if scratch == scratch and dc == dc and dc < scratch:
        print(f"  FALSIFIED at kappa>0. The detector reaches the evasion bar at displacement "
              f"{dc:.3f},\n  which is {100*dc/scratch:.0f}% of an independent training run -- not 'as expensive as\n"
              f"  training from scratch'. And the models demonstrating it are benign public releases.")
    else:
        print("  not falsified on this evidence (or insufficient data)")
print("\n(displacement = ||theta_c - theta_p||_F / ||theta_p||_F over shared-shape parameters)")
