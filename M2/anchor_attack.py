"""X5 — ANCHOR-TARGETED PERTURBATION.

EAS carries 0.36 of MPK's identity weight -- the largest single share -- yet it reads
only K=64 embedding rows, chosen by a fully deterministic rule that is hardcoded in the
open-source repo (ANCHOR_PUNCT/ANCHOR_LATIN/... constants, script-proportional allocation,
then np.random.RandomState(42) backfill over ids < 50000). An O1 adversary -- one who only
reads the published source, issues no queries -- can therefore compute the exact anchor set
and perturb ONLY those rows.

Contrast with the isotropic sweep, where destroying the model (ppl 20.2 -> 53.6) moved EAS
by 0.0028: there the adversary spent its perturbation budget over all 49152 rows, of which
64 mattered. This targets the 64.
"""
import os, sys, json, shutil, subprocess, pathlib, copy, torch, statistics as st
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.path.insert(0,str(_ROOT) + "/M2/transforms")
sys.path.insert(0,str(_ROOT) + "/M1/oracle/model-provenance-kit/src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import laundry as LD
from provenancekit.core.signals.anchors import get_anchor_ids     # O1: read the source

# Repo root derived from this file's location; set LAUNDROMAT_ROOT to override.
import os as _os, pathlib as _pl
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "MANIFEST-dataintegrity.txt").exists())))


REPO=pathlib.Path(str(_ROOT) + "/M1/oracle/model-provenance-kit")
RES =pathlib.Path(str(_ROOT) + "/M2/results")
W   =pathlib.Path(str(_ROOT) + "/M2/work"); W.mkdir(exist_ok=True)
NULL=json.loads((RES/"null_frozen.json").read_text())["scores"]
def pct(x): return 100.0*sum(1 for s in NULL if s<=x)/len(NULL)
def mpk(a):
    p=subprocess.run(["uv","run","provenancekit",*a,"--json","--no-cache"],cwd=REPO,
        capture_output=True,text=True,timeout=5400,env={**os.environ,"HF_HUB_DISABLE_XET":"1"})
    i=p.stdout.find("{")
    return json.loads(p.stdout[i:]) if i>=0 else {"error":p.stderr[-200:]}

# Capability probe: longer and more varied than the 96-probe fidelity set, which was too easy
# (noise was *improving* its perplexity). Held fixed across all arms.
CORPUS=[
 "The Industrial Revolution began in Britain in the late eighteenth century and transformed manufacturing, transport, and agriculture across Europe and North America.",
 "In computer science, a hash table is a data structure that implements an associative array, mapping keys to values using a hash function to compute an index.",
 "Photosynthesis converts light energy into chemical energy stored in glucose, and it takes place in the chloroplasts of plant cells using chlorophyll.",
 "The French Revolution of 1789 abolished the monarchy, established a republic, and profoundly reshaped the political institutions of continental Europe.",
 "A neural network consists of layers of interconnected nodes, and training adjusts the weights between them to minimise a loss function over the data.",
 "Shakespeare wrote approximately thirty-nine plays and one hundred and fifty-four sonnets, and his work remains central to the English literary canon.",
 "The theory of plate tectonics explains the movement of the Earth's lithosphere, accounting for earthquakes, volcanic activity, and mountain formation.",
 "In economics, inflation is the rate at which the general level of prices for goods and services rises, eroding the purchasing power of a currency.",
 "The immune system defends the body against pathogens through innate responses and adaptive responses involving lymphocytes and antibody production.",
 "Quantum mechanics describes the behaviour of matter and energy at atomic scales, where particles exhibit both wave-like and particle-like properties.",
]*4

def anchor_ids_for(model_id, tok):
    vocab=set(tok.get_vocab().keys())
    return get_anchor_ids(tok, vocab, len(tok.get_vocab()), anchor_k=64)

def X5_anchor_perturb(model, ids, rel_sigma, seed=0):
    emb=model.get_input_embeddings().weight
    with torch.no_grad():
        e32=emb.float()
        idx=torch.tensor(sorted(set(int(i) for i in ids if 0<=int(i)<e32.shape[0])))
        rows=e32[idx]
        rn=rows.norm(dim=1,keepdim=True)
        g=torch.randn(rows.shape,generator=torch.Generator().manual_seed(seed))
        e32[idx]=rows+g*(rel_sigma*rn/(rows.shape[1]**0.5))
        emb.copy_(e32.to(emb.dtype))
    return {"transform":"X5anchor","rel_sigma":rel_sigma,"n_rows":int(idx.numel()),
            "vocab":int(emb.shape[0]),"frac_of_vocab":float(idx.numel())/emb.shape[0]}

if __name__=="__main__":
    P,C=sys.argv[1],sys.argv[2]
    tok=AutoTokenizer.from_pretrained(C)
    ids=anchor_ids_for(C,tok)
    print(f"parent={P}")
    print(f"anchor ids recovered from MPK source (O1): n={len(ids)} of vocab {len(tok.get_vocab())} "
          f"= {100*len(ids)/len(tok.get_vocab()):.3f}% of rows")
    print(f"  first 16 (punct/digit): {ids[:16]}")
    print(f"  decoded: {[tok.decode([i]) for i in ids[:24]]}")
    out=(RES/f"anchor_attack_{C.replace('/','--')}.jsonl").open("w")
    print(f"\n{'rel_sigma':>10}{'EAS':>8}{'END':>8}{'NLF':>8}{'LEP':>8}{'WVC':>8}{'sig_id':>8}{'pipe':>8}"
          f"{'rank':>6}{'pi':>7}{'ppl':>9}{'rho':>8}{'E':>3}")
    for s in [0.0,0.25,0.5,1.0,2.0,4.0,8.0]:
        ref=AutoModelForCausalLM.from_pretrained(C).eval(); m=copy.deepcopy(ref)
        applied=[LD.X1a_mlp_permute(m,1.0,11), LD.X1b_head_permute(m,11)]
        if s>0: applied.append(X5_anchor_perturb(m, ids, s, seed=11))
        fid=LD.verify_exact(ref.float(), copy.deepcopy(m).float(), tok, CORPUS)
        o=W/f"anc{str(s).replace('.','p')}"
        if o.exists(): shutil.rmtree(o)
        m.save_pretrained(o); tok.save_pretrained(o)
        applied.append(LD.M1t_metadata(o,mode="both"))
        del m,ref
        cmp_=mpk(["compare",P,str(o)]); scan=mpk(["scan",str(o),"--top-k","300","--threshold","0.0"])
        sc=cmp_.get("scores",{}) or {}; sg=cmp_.get("signals",{}) or {}
        ms=scan.get("matches",[]); ps=P.split("/")[-1]
        ix=[i for i,x in enumerate(ms,1) if x["model_id"]==ps]; R=ix[0] if ix else None
        sp=sc.get("pipeline_score"); pi=pct(sp) if sp is not None else None
        rho=fid["ppl_ref"]/fid["ppl_new"] if fid.get("ppl_new") else None
        E=1 if (R and R>5 and pi is not None and pi<=75) else 0
        out.write(json.dumps({"rel_sigma":s,"applied":applied,"scores":sc,"signals":sg,"rank":R,
            "pi":pi,"fid":fid,"rho":rho,"E":E,"kappa":0.0},default=str)+"\n"); out.flush()
        print(f"{s:>10.2f}{(sg.get('eas') or 0):>8.4f}{(sg.get('end') or 0):>8.4f}{(sg.get('nlf') or 0):>8.4f}"
              f"{(sg.get('lep') or 0):>8.4f}{(sg.get('wvc') or 0):>8.4f}{sc.get('identity_score',0):>8.4f}"
              f"{sc.get('pipeline_score',0):>8.4f}{str(R):>6}{(pi or 0):>7.1f}{fid['ppl_new']:>9.3f}"
              f"{(rho or 0):>8.4f}{E:>3}",flush=True)
        shutil.rmtree(o,ignore_errors=True)
    out.close(); print("DONE-ANCHOR")
