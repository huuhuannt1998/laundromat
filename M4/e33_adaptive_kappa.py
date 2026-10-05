"""E33 -- the compute-spending adaptive adversary against the alignment signal.

Section IX-C evaluates the defence against a ZERO-compute adaptive adversary: the QK/OV
bilinear attack, which drives alignment computed on the query--key circuit from 1.000 to
0.216 while alignment on the MLP rows -- the matrices the defence actually uses -- stays at
1.000. Section X-C turns that into a design rule: align only on parameter structures whose
residual symmetry group is discrete (permutation-only) rather than continuous. The rule was
never priced. Section IX-D carried the gap as a limitation: "The defence was not evaluated
against an adversary with training compute to spend."

This is the priced version. The adversary knows the defence exactly and spends training
compute to break it.

  arm "targeted"    minimise  L_LM(x) + lambda * A(W_child, W_parent)
                    where A is the column-normalised LAP score of section IX computed on the
                    MLP up-projection rows (dense_h_to_4h.weight) -- i.e. the defence's own
                    statistic, differentiated. The assignment is solved with the Hungarian
                    algorithm on the detached cost matrix and held fixed between refreshes;
                    the gradient at a fixed optimal assignment is a valid subgradient of the
                    assignment value (Danskin).
  arm "untargeted"  minimise  L_LM(x) only, same data, steps, optimiser and seed.
                    THE CONTROL. It separates "the attack worked" from "any training at this
                    budget moves the signal".

Both arms start from the parent's own weights, so alignment begins at exactly 1.0000 and every
subsequent movement is bought with compute. Capability is held with a domain-DISJOINT held-out
set: the attack corpus is HellaSwag narrative text, the held-out set is LAMBADA book passages,
which neither arm ever sees. We report held-out cross-entropy and LAMBADA last-word accuracy.

kappa is reported as attack tokens / parent pretraining tokens (299,892,736,000 for
pythia-160m). Because attack and pretraining use the same model, the 6ND FLOP ratio equals the
token ratio; kappa_flop additionally charges the adversary for the alignment penalty, which is
weight-space compute the parent's pretraining never paid.

BUDGET CEILING, STATED RATHER THAN IMPLIED. The review asked for kappa ~ 1-5%. That is 3-15
billion tokens. On the one laptop this project is allowed (Apple M4, MPS, no discrete GPU)
pythia-160m trains at ~2.5k tokens/s, so 1% is ~14 days of continuous compute per arm. We do
not have it and we do not fake it. We reach the budget recorded in `budget` below and report
the curve up to that ceiling; what lies above it is untested and the paper says so.

Output: M4/e33_adaptive_kappa.json
"""
import json, os as _os, pathlib as _pl, time, math, argparse, random
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_ROOT = _pl.Path(_os.environ.get("LAUNDROMAT_ROOT",
        next(p for p in _pl.Path(__file__).resolve().parents if (p / "MANIFEST-dataintegrity.txt").exists())))
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.optimize import linear_sum_assignment

torch.manual_seed(17); np.random.seed(17); random.seed(17)
OUT = _ROOT / "M4" / "e33_adaptive_kappa.json"
MODEL = "EleutherAI/pythia-160m"
PRETRAIN_TOKENS = 299.892736e9          # Pythia 143k steps x 1024 x 2048, as used by M7/e32
NONEMB_PARAMS   = 85_056_000            # 162,322,944 total - 2 x (50304 x 768) tied-shape embeddings
# The published decision geometry this attack is measured against (M7/e19, M4/defence_final_stats):
THRESHOLD   = 0.4727   # midpoint threshold fitted on the development families
HIGHEST_NEG = 0.4513   # legal-bert-base-uncased, the binding independently-pretrained negative
LOWEST_DERIV= 0.7918   # bert-base-cased -> distilbert-base-cased under recovered correspondence
MARGIN      = 0.3405   # LOWEST_DERIV - HIGHEST_NEG
SAME_RECIPE = 0.1961   # pythia-160m vs pythia-160m-deduped: what a genuinely independent same-recipe run reads

P = lambda *a: print(*a, flush=True)


# ---------------------------------------------------------------- the defence's own statistic
def ups(m):
    """MLP up-projection rows, the matrices section IX aligns on. Same selector as M7/e32."""
    return [p for n, p in m.named_parameters() if p.ndim == 2 and "dense_h_to_4h.weight" in n]

def _norm2(A):
    A = A / (A.norm(dim=0, keepdim=True) + 1e-12)     # column l2, removes the scaling freedom
    return A / (A.norm(dim=1, keepdim=True) + 1e-12)  # row l2, so the cost matrix is cosines

def lap_colnorm(A, B):
    """Exactly M7/e32's lap_colnorm (max_feat never binds at width 768). Float32, CPU."""
    An, Bn = _norm2(A.float()), _norm2(B.float())
    S = (An @ Bn.T).numpy()
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())

def full_lap(child, parent_ups):
    """The published score: mean over all layers, computed on CPU in float32."""
    with torch.no_grad():
        cu = [p.detach().to("cpu") for p in ups(child)]
        per = [lap_colnorm(a, b) for a, b in zip(cu, parent_ups)]
    return float(np.mean(per)), per


# ---------------------------------------------------------------------------------- the data
def build_corpora(tok, block, n_train_blocks, n_heldout):
    from datasets import load_dataset
    hs = load_dataset("Rowan/hellaswag")["train"]
    texts, budget_chars = [], n_train_blocks * block * 6
    tot = 0
    for r in hs:
        t = (r["ctx"] + " " + r["endings"][int(r["label"])]).strip()
        texts.append(t); tot += len(t)
        if tot > budget_chars: break
    ids = tok("\n\n".join(texts), return_tensors=None)["input_ids"]
    n = (len(ids) // block) * block
    train = torch.tensor(ids[:n], dtype=torch.long).view(-1, block)
    lam = load_dataset("EleutherAI/lambada_openai", "en")["test"]
    held = [lam[i]["text"] for i in range(min(n_heldout, len(lam)))]
    return train, held


@torch.no_grad()
def eval_heldout(model, tok, held, device, batch=8):
    """Domain-disjoint held-out cross-entropy (nats/token) and LAMBADA last-word accuracy."""
    model.eval()
    tot_nll, tot_tok, correct = 0.0, 0, 0
    for i in range(0, len(held), batch):
        chunk = held[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        ids = enc["input_ids"].to(device); att = enc["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]; m = att[:, 1:].bool()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot_nll += float((nll * m).sum()); tot_tok += int(m.sum())
    for t in held:                                    # last-word accuracy, greedy, teacher-forced
        words = t.strip().split()
        ctx, last = " ".join(words[:-1]), " " + words[-1]
        cid = tok(ctx, return_tensors="pt")["input_ids"]
        lid = tok(last, return_tensors="pt")["input_ids"]
        full = torch.cat([cid, lid], 1).to(device)
        if full.shape[1] > 300: continue
        pred = model(input_ids=full).logits[0, cid.shape[1] - 1: -1].argmax(-1).cpu()
        correct += int(torch.equal(pred, lid[0]))
    model.train()
    return tot_nll / max(tot_tok, 1), correct / max(len(held), 1)



# ------------------------------------------------------------------------------ the optimiser
class SafeAdamW:
    """AdamW written in elementwise ops with a non-finite guard.

    torch.optim.AdamW's MPS path in this build (torch 2.11.0, Apple M4) returns non-finite
    PARAMETERS from finite gradients on the first step. Traced to the embedding matrix: for the
    38.6M entries whose gradient is exactly zero the fused elementwise chain evaluates
    0 / (sqrt(0) + eps) as 0/0 rather than 0, and one non-finite entry in `embed_in` poisons the
    whole model on the next forward. The same expression on a small tensor is correct, so it is a
    fast-math path on large tensors, not the algorithm. Raising eps to 1e-4 also avoids it but
    silently changes Adam's preconditioner, so we keep eps=1e-8 and repair the exact entries whose
    mathematically-correct update is zero. Verified: 120 steps, loss flat at 2.5-2.7, zero
    non-finite parameters. Reported here because it is a reproduction hazard for anyone re-running
    this arm on Apple silicon.
    """
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, wd=0.0, warmup=100):
        self.p = [q for q in params if q.requires_grad]
        self.lr0, self.b1, self.b2, self.eps, self.wd, self.warmup = lr, betas[0], betas[1], eps, wd, warmup
        self.t = 0
        self.m = [torch.zeros_like(q) for q in self.p]
        self.v = [torch.zeros_like(q) for q in self.p]

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        lr = self.lr0 * min(1.0, self.t / max(self.warmup, 1))     # warmup: Adam's first step on a
        for q, m, v in zip(self.p, self.m, self.v):                # converged model is a sign step
            if q.grad is None: continue
            g = q.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            upd = (m / bc1) / ((v / bc2).sqrt() + self.eps)
            upd = torch.nan_to_num(upd, nan=0.0, posinf=0.0, neginf=0.0)
            if self.wd: q.add_(q, alpha=-lr * self.wd)
            q.add_(upd, alpha=-lr)

    def zero_grad(self):
        for q in self.p: q.grad = None


# ------------------------------------------------------------------------------------ the arm
def run_arm(name, lam, steps, ckpts, cfg, parent_ups, tok, train, held):
    dev = cfg["device"]
    torch.manual_seed(17); np.random.seed(17); random.seed(17)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.train()
    opt = SafeAdamW(model.parameters(), lr=cfg["lr"], warmup=cfg["warmup"])
    B, L, K, NL = cfg["batch"], cfg["block"], cfg["refresh"], cfg["pen_layers"]
    nlayer = len(parent_ups)
    p_dev = [b.to(dev) for b in parent_ups]
    assign = {}
    rows, order = [], torch.randperm(train.shape[0])
    ptr, t_start = 0, time.time()

    def snapshot(step):
        tokens = step * B * L
        lapv, per = full_lap(model, parent_ups)
        hl, acc = eval_heldout(model, tok, held, dev)
        # kappa_flop charges the adversary for the alignment penalty as well as the LM step.
        lm_flops  = 6 * NONEMB_PARAMS * tokens
        pen_flops = step * NL * 2 * (3072 * 768 * 3072) * 3   # cost matrix fwd + bwd, per sampled layer
        rec = dict(step=step, tokens=tokens,
                   kappa_tok=tokens / PRETRAIN_TOKENS,
                   kappa_flop=(lm_flops + (pen_flops if lam > 0 else 0)) / (6 * NONEMB_PARAMS * PRETRAIN_TOKENS),
                   lap=lapv, lap_per_layer=per, heldout_nll=hl, lambada_acc=acc,
                   below_threshold=lapv < THRESHOLD, below_highest_negative=lapv < HIGHEST_NEG,
                   wall_s=round(time.time() - t_start, 1))
        rows.append(rec)
        P(f"  [{name}] step {step:>5} tok {tokens:>10,} kappa {rec['kappa_tok']:.3e} "
          f"LAP {lapv:.4f} heldout {hl:.4f} acc {acc:.3f} {rec['wall_s']:.0f}s")
        return rec

    snapshot(0)
    for step in range(1, steps + 1):
        if ptr + B > train.shape[0]:
            order = torch.randperm(train.shape[0]); ptr = 0
        idx = order[ptr:ptr + B]; ptr += B
        x = train[idx].to(dev)
        out = model(input_ids=x, labels=x)
        loss = out.loss
        lm_loss = float(loss.detach())
        pen = 0.0
        if lam > 0:
            sel = np.random.choice(nlayer, size=min(NL, nlayer), replace=False)
            cu = ups(model)
            terms = []
            for li in sel:
                An, Bn = _norm2(cu[li]), _norm2(p_dev[li])
                S = An @ Bn.T
                if step % K == 1 or li not in assign:
                    with torch.no_grad():
                        r, c = linear_sum_assignment(-S.detach().float().cpu().numpy())
                    assign[li] = (torch.as_tensor(r, device=dev), torch.as_tensor(c, device=dev))
                r, c = assign[li]
                terms.append(S[r, c].mean())
            pen_t = torch.stack(terms).mean()
            pen = float(pen_t)
            loss = loss + lam * pen_t
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 50 == 0:
            P(f"    [{name}] {step}/{steps} lm {lm_loss:.3f} pen {pen:.4f} "
              f"({(time.time()-t_start)/step:.2f}s/step)")
        if step in ckpts:
            snapshot(step)
    del model, opt
    if dev == "mps": torch.mps.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    # A laundering adversary must ship a model someone still wants. At lr 3e-4 the pilot
    # drove held-out CE 3.39 -> 5.52 and LAMBADA accuracy 0.350 -> 0.150 in BOTH arms, so
    # "alignment stayed at 1.0" would have been a statement about a destroyed model rather
    # than about the defence. The rate is therefore a control, and the run of record picks
    # one that holds capability.
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default=None, help="output suffix, so probes never overwrite the run of record")
    a = ap.parse_args()

    cfg = dict(device=a.device, lr=a.lr, batch=8, block=512, refresh=10, pen_layers=3,
               warmup=100, steps=64 if a.pilot else a.steps)
    ck = [8, 16, 32, 64] if a.pilot else [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ckpts = sorted(s for s in ck if s <= cfg["steps"])

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    n_blocks = max(cfg["steps"] * cfg["batch"] // 4, 2000)
    train, held = build_corpora(tok, cfg["block"], n_blocks, 60 if a.pilot else 400)
    P(f"train blocks {tuple(train.shape)}  held-out passages {len(held)}  device {cfg['device']}")

    parent = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    parent_ups = [p.detach().clone().float() for p in ups(parent)]
    del parent

    arms = {}
    lam_t = 10.0
    only = a.only
    if only == "targeted":
        arms["targeted"] = run_arm("targeted", lam_t, cfg["steps"], ckpts, cfg, parent_ups, tok, train, held)
        P(json.dumps(arms["targeted"], indent=1)[:4000]); return
    arms["untargeted"] = run_arm("untargeted", 0.0, cfg["steps"], ckpts, cfg, parent_ups, tok, train, held)
    arms["targeted"]   = run_arm("targeted", lam_t, cfg["steps"], ckpts, cfg, parent_ups, tok, train, held)
    if not a.pilot:
        arms["targeted_aggressive"] = run_arm("targeted_aggressive", 50.0, min(512, cfg["steps"]),
                                              [s for s in ckpts if s <= 512], cfg, parent_ups, tok, train, held)

    doc = dict(
        schema="laundromat.adaptive-kappa/v1",
        model=MODEL, parent_is_start=True, device=cfg["device"], config=cfg,
        lambda_targeted=lam_t, lambda_aggressive=50.0,
        pretrain_tokens=PRETRAIN_TOKENS, nonembedding_params=NONEMB_PARAMS,
        decision_geometry=dict(threshold=THRESHOLD, highest_negative=HIGHEST_NEG,
                               lowest_derivative=LOWEST_DERIV, margin=MARGIN,
                               same_recipe_floor=SAME_RECIPE),
        budget=dict(max_steps=cfg["steps"], max_tokens=cfg["steps"] * cfg["batch"] * cfg["block"],
                    max_kappa_tok=cfg["steps"] * cfg["batch"] * cfg["block"] / PRETRAIN_TOKENS,
                    requested_kappa="0.01-0.05", reached_ceiling=True),
        arms=arms)
    out = OUT if not a.tag else OUT.with_name(f"e33_adaptive_kappa_{a.tag}.json")
    out.write_text(json.dumps(doc, indent=1))
    P(f"wrote {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
