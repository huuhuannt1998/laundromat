"""LAUNDROMAT kappa=0 transform library.

Every transform here costs ZERO training compute: pure tensor permutation/scaling
plus file writes. X1a, X1b, X2 are FUNCTION-PRESERVING BY CONSTRUCTION; M1t touches
no weights at all. Exactness is verified numerically by verify_exact() -- never assumed.
"""
import json, shutil, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- helpers
def _layers(model):
    """Locate the transformer block list across architecture families.

    Llama/Qwen/SmolLM2 expose .model.layers; GPT-NeoX (Pythia) exposes
    .gpt_neox.layers; GPT-2 exposes .transformer.h.  Added when the pythia-1.4b
    transfer arm failed with "'GPTNeoXForCausalLM' object has no attribute 'model'".
    """
    for attr, sub in (("model", "layers"), ("gpt_neox", "layers"),
                      ("transformer", "h"), ("bert", "encoder")):
        obj = getattr(model, attr, None)
        if obj is not None:
            got = getattr(obj, sub, None)
            if got is not None:
                return got.layer if sub == "encoder" else got
    raise AttributeError(f"cannot locate layer list on {type(model).__name__}")

def _gen(seed):
    g = torch.Generator().manual_seed(seed); return g

# ---------------------------------------------------------------- X1a
def X1a_mlp_permute(model, frac=1.0, seed=0):
    """Permute the MLP intermediate dimension, per layer.

    y = down @ (silu(gate@x) * (up@x)).  Permuting the ROWS of gate and up permutes
    the elementwise product identically; permuting the COLUMNS of down undoes it.
    Exact for any permutation. The intermediate axis carries no positional structure
    (no RoPE, no head grouping), so this is the safest exact symmetry in the family.

    Non-gated MLPs (GPT-NeoX: dense_h_to_4h -> GELU -> dense_4h_to_h) admit the same
    symmetry: permuting the rows of the input projection AND ITS BIAS, then the columns
    of the output projection, is exact because GELU is elementwise. Forgetting the bias
    would silently break exactness, so it is permuted explicitly.
    """
    n = 0
    for li, layer in enumerate(_layers(model)):
        mlp = layer.mlp
        gated = hasattr(mlp, "gate_proj")
        w_in = mlp.gate_proj if gated else getattr(mlp, "dense_h_to_4h", None)
        if w_in is None:
            continue
        I = w_in.weight.shape[0]
        k = max(1, int(round(frac * I)))
        perm = torch.arange(I)
        sub = torch.randperm(I, generator=_gen(seed * 1000 + li))[:k]
        perm[sub] = sub[torch.randperm(k, generator=_gen(seed * 1000 + li + 7))]
        with torch.no_grad():
            if gated:
                mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[perm, :])
                mlp.up_proj.weight.copy_(mlp.up_proj.weight[perm, :])
                mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, perm])
            else:
                mlp.dense_h_to_4h.weight.copy_(mlp.dense_h_to_4h.weight[perm, :])
                if getattr(mlp.dense_h_to_4h, "bias", None) is not None:
                    mlp.dense_h_to_4h.bias.copy_(mlp.dense_h_to_4h.bias[perm])
                mlp.dense_4h_to_h.weight.copy_(mlp.dense_4h_to_h.weight[:, perm])
        n += 1
    return {"transform": "X1a", "frac": frac, "seed": seed, "layers": n}

# ---------------------------------------------------------------- X1b
def X1b_head_permute(model, seed=0):
    """Permute attention heads as whole units, GQA-aware.

    Permutes KV GROUPS as units: each kv head plus the q heads that share it move
    together, and o_proj's matching column blocks follow. Exact because attention is
    equivariant to a consistent permutation of head slots, and RoPE is untouched
    (we never permute WITHIN a head, which would break the rotary pairing).
    """
    cfg = model.config
    H  = cfg.num_attention_heads
    KV = getattr(cfg, "num_key_value_heads", H) or H
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
    grp = H // KV
    n = 0
    for li, layer in enumerate(_layers(model)):
        a = layer.self_attn
        gperm = torch.randperm(KV, generator=_gen(seed * 2000 + li))
        q_idx, kv_idx = [], []
        for g in gperm.tolist():
            for j in range(grp):
                h = g * grp + j
                q_idx.extend(range(h * hd, (h + 1) * hd))
            kv_idx.extend(range(g * hd, (g + 1) * hd))
        q_idx  = torch.tensor(q_idx);  kv_idx = torch.tensor(kv_idx)
        with torch.no_grad():
            a.q_proj.weight.copy_(a.q_proj.weight[q_idx, :])
            if a.q_proj.bias is not None: a.q_proj.bias.copy_(a.q_proj.bias[q_idx])
            a.k_proj.weight.copy_(a.k_proj.weight[kv_idx, :])
            if a.k_proj.bias is not None: a.k_proj.bias.copy_(a.k_proj.bias[kv_idx])
            a.v_proj.weight.copy_(a.v_proj.weight[kv_idx, :])
            if a.v_proj.bias is not None: a.v_proj.bias.copy_(a.v_proj.bias[kv_idx])
            a.o_proj.weight.copy_(a.o_proj.weight[:, q_idx])
        n += 1
    return {"transform": "X1b", "seed": seed, "layers": n, "kv_groups": KV, "grp": grp}

# ---------------------------------------------------------------- X2
def X2_norm_scale(model, log_sigma=0.1, seed=0, scope="all"):
    """RMSNorm <-> linear scaling invariance.

    RMSNorm emits (x/rms(x)) * g, consumed by a linear W. Scaling g elementwise by
    alpha and dividing W's INPUT COLUMNS by the same alpha leaves the product exactly
    unchanged. Applied at input_layernorm -> {q,k,v}_proj, post_attention_layernorm ->
    {gate,up}_proj, and final norm -> lm_head.

    This is the only exact transform that moves NORM-LAYER weights, so it is the one
    that can touch MPK's NLF signal -- and, by rescaling individual tensors, LEP.
    """
    d = model.config.hidden_size
    touched = []
    for li, layer in enumerate(_layers(model)):
        if scope in ("all", "attn"):
            a = torch.exp(torch.randn(d, generator=_gen(seed*3000+li)) * log_sigma)
            with torch.no_grad():
                layer.input_layernorm.weight.mul_(a)
                for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                    p.weight.div_(a.unsqueeze(0))
            touched.append(f"L{li}.attn")
        if scope in ("all", "mlp"):
            b = torch.exp(torch.randn(d, generator=_gen(seed*3000+li+11)) * log_sigma)
            with torch.no_grad():
                layer.post_attention_layernorm.weight.mul_(b)
                for p in (layer.mlp.gate_proj, layer.mlp.up_proj):
                    p.weight.div_(b.unsqueeze(0))
            touched.append(f"L{li}.mlp")
    if scope == "all" and not model.config.tie_word_embeddings:
        c = torch.exp(torch.randn(d, generator=_gen(seed*3000+999)) * log_sigma)
        with torch.no_grad():
            model.model.norm.weight.mul_(c)
            model.lm_head.weight.div_(c.unsqueeze(0))
        touched.append("final")
    return {"transform": "X2", "log_sigma": log_sigma, "seed": seed, "scope": scope,
            "sites": len(touched), "final_norm_scaled": scope=="all" and not model.config.tie_word_embeddings}

# ---------------------------------------------------------------- M1t
def M1t_metadata(out_dir, mode="max_pos"):
    """Edit config.json to break MPK's arch_hash. Touches NO weights.

    arch_hash is a hash over dimension fields + model_type + architectures, all of
    which the publisher writes. This is the zero-cost, bidirectional surface.
    """
    cp = pathlib.Path(out_dir) / "config.json"
    cfg = json.loads(cp.read_text()); before = dict(cfg)
    if mode == "max_pos":
        cfg["max_position_embeddings"] = int(cfg.get("max_position_embeddings", 2048)) + 1
    elif mode == "arch_name":
        cfg["architectures"] = [a + "Ext" for a in cfg.get("architectures", ["Model"])]
    elif mode == "both":
        cfg["max_position_embeddings"] = int(cfg.get("max_position_embeddings", 2048)) + 1
        cfg["architectures"] = [a + "Ext" for a in cfg.get("architectures", ["Model"])]
    cp.write_text(json.dumps(cfg, indent=2))
    return {"transform": "M1t", "mode": mode,
            "changed": {k: (before.get(k), cfg.get(k)) for k in cfg if before.get(k) != cfg.get(k)}}

# ---------------------------------------------------------------- exactness
@torch.no_grad()
def verify_exact(ref_model, new_model, tok, probes, max_len=128):
    """F1 verification: these transforms are exact BY THEOREM; this measures float error.

    Predeclared tolerance (fp32): max|delta logit| <= 1e-2 and relative ppl delta <= 1e-3.
    """
    ref_model.eval(); new_model.eval()
    md, nll_r, nll_n, ntok = 0.0, 0.0, 0.0, 0
    for t in probes:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        if ids.shape[1] < 2: continue
        lr = ref_model(**enc).logits.float()
        ln = new_model(**enc).logits.float()
        md = max(md, (lr - ln).abs().max().item())
        tgt = ids[:, 1:].reshape(-1)
        for lg, acc in ((lr, "r"), (ln, "n")):
            v = torch.nn.functional.cross_entropy(
                    lg[:, :-1].reshape(-1, lg.shape[-1]), tgt, reduction="sum").item()
            if acc == "r": nll_r += v
            else:          nll_n += v
        ntok += tgt.numel()
    ppl_r = torch.exp(torch.tensor(nll_r / ntok)).item()
    ppl_n = torch.exp(torch.tensor(nll_n / ntok)).item()
    rel = abs(ppl_n - ppl_r) / ppl_r
    return {"max_abs_logit_delta": md, "ppl_ref": ppl_r, "ppl_new": ppl_n,
            "rel_ppl_delta": rel, "n_tokens": ntok,
            "exact_within_tolerance": bool(md <= 1e-2 and rel <= 1e-3)}

# ---------------------------------------------------------------- N1emb  (INEXACT, kappa=0)
def N1_embed_noise(model, rel_sigma=0.01, seed=0):
    """Perturb the EMBEDDING MATRIX ONLY with relative Gaussian noise. kappa = 0 (no training).

    Motivated by the G1-B floor result: EAS and END read the embedding matrix and are
    invariant to every embedding-PRESERVING exact transform, so sigma_id >= w_EAS + w_END
    ~ 0.55 no matter what is done elsewhere. The only route to those two signals is to
    change embedding values -- which is necessarily INEXACT and therefore costs capability.
    This transform traces that exchange rate.
    """
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        # Accumulate in fp32 then cast back. Adding directly in bf16 silently rounds away
        # any perturbation below the bf16 relative resolution (~2^-8 = 4e-3), which made
        # rel_sigma 0.003 and 0.010 produce bit-identical artifacts in the first sweep.
        e32 = emb.float()
        rn  = e32.norm(dim=1, keepdim=True)
        g   = torch.randn(emb.shape, generator=_gen(seed))
        e32 = e32 + g * (rel_sigma * rn / (emb.shape[1] ** 0.5))
        emb.copy_(e32.to(emb.dtype))
    return {"transform": "N1emb", "rel_sigma": rel_sigma, "seed": seed,
            "tied": bool(model.config.tie_word_embeddings), "vocab": int(emb.shape[0])}

# ---------------------------------------------------------------- M3t (kappa=0, tokenizer config)
def M3t_tokenizer_config(out_dir, max_length=128):
    """Write a `truncation` block into tokenizer.json. kappa = 0, weights untouched.

    Discovered while resolving anomaly A1: MPK folds a hash of the tokenizer BACKEND into
    family_hash, and a fast tokenizer's truncation configuration is part of that serialised
    backend. The block is pure runtime inference config -- it does not change behaviour on
    untruncated input and says nothing about lineage -- yet it changes family_hash and demotes
    the MFI gate from tier 2 to tier 3, forcing the verdict onto the weight signals.
    Subtler than editing architecture fields: an innocuous-looking artifact difference.
    """
    import json as _json
    tp = pathlib.Path(out_dir) / "tokenizer.json"
    if not tp.exists():
        return {"transform": "M3t", "applied": False, "reason": "no tokenizer.json"}
    d = _json.loads(tp.read_text())
    before = d.get("truncation")
    d["truncation"] = {"direction": "Right", "max_length": int(max_length),
                       "strategy": "LongestFirst", "stride": 0}
    tp.write_text(_json.dumps(d))
    return {"transform": "M3t", "applied": True, "before": before, "after": d["truncation"]}

# ---------------------------------------------------------------- X7  QK-invariance (kappa=0)
def X7_qk_invariance(model, log_sigma=0.3, seed=0):
    """Attention QK invariance: W_q -> W_q M, W_k -> W_k M^-T, for ANY invertible M per head.

    scores = (x W_q)(x W_k)^T = x W_q W_k^T x^T, so W_q M M^-1 W_k^T = W_q W_k^T. Exact.

    This is a RICHER symmetry than permutation or diagonal scaling: M is an arbitrary invertible
    matrix, so it MIXES coordinates rather than merely relabelling or rescaling them. That matters
    for the defence: LAP alignment undoes a PERMUTATION, but no assignment problem can undo a
    general linear mixing. Fernandez et al. note invertible-matrix invariants exist; nobody has
    tested them against a deployed provenance detector.

    Requires NO RoPE (RoPE does not commute with arbitrary M). GPT-2 / BERT-style learned or
    absolute positional models qualify; Llama/Qwen/Pythia do not.
    """
    cfg = model.config
    H  = cfg.num_attention_heads
    d  = cfg.hidden_size
    hd = d // H
    n = 0
    blocks = getattr(getattr(model, "transformer", model), "h", None)
    if blocks is None:
        raise ValueError("X7 expects a GPT-2-style .transformer.h block list")
    for li, blk in enumerate(blocks):
        W = blk.attn.c_attn.weight          # Conv1D: [d, 3d] laid out q|k|v
        B = blk.attn.c_attn.bias            # [3d]
        with torch.no_grad():
            for h in range(H):
                g = _gen(seed * 5000 + li * 100 + h)
                M = torch.randn(hd, hd, generator=g) * log_sigma
                M = torch.matrix_exp(M)                      # guaranteed invertible
                Minv_T = torch.linalg.inv(M).T
                qs, ks = h * hd, d + h * hd
                W[:, qs:qs + hd] = W[:, qs:qs + hd].float() @ M
                W[:, ks:ks + hd] = W[:, ks:ks + hd].float() @ Minv_T
                B[qs:qs + hd] = B[qs:qs + hd].float() @ M
                B[ks:ks + hd] = B[ks:ks + hd].float() @ Minv_T
        n += 1
    return {"transform": "X7qk", "log_sigma": log_sigma, "seed": seed, "layers": n,
            "heads": H, "head_dim": hd, "note": "exact only without RoPE"}

# ---------------------------------------------------------------- Q1  quantize round-trip (kappa=0)
def Q1_quant_roundtrip(model, bits=8, per_channel=True):
    """Quantize -> dequantize every 2D weight. kappa = 0, INEXACT by construction.

    Baseline B-1 from the design: "the standard benign transform; anything weaker is noise."
    A publisher does this routinely (int8/int4 release). It is the noise floor the transform arm
    needs -- any laundering effect smaller than a plain quantization round-trip is not an effect.
    """
    n = 0
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim != 2:
                continue
            w = p.float()
            if per_channel:
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
            else:
                s = w.abs().max().clamp(min=1e-12) / qmax
            p.copy_(((w / s).round().clamp(-qmax - 1, qmax) * s).to(p.dtype))
            n += 1
    return {"transform": "Q1", "bits": bits, "per_channel": per_channel, "tensors": n}

# ---------------------------------------------------------------- X3  OV-circuit invariance (kappa=0)
def X3_ov_invariance(model, log_sigma=0.3, seed=0):
    """OV-circuit invariance: W_v -> W_v M, W_o -> M^-1 W_o, per head. Exact for any invertible M.

    Attention output is softmax(QK^T) (x W_v) W_o. Inserting M M^-1 between W_v and W_o leaves the
    product unchanged, so this is exact for ANY invertible M -- a general linear MIXING, like X7.

    CRUCIALLY, UNLIKE X7 THIS WORKS WITH RoPE. RoPE rotates only Q and K; it never touches V or O.
    So the general-linear symmetry that X7 could only reach on GPT-2/BERT-style models is available
    on Llama / Qwen / Pythia too, via the OV circuit. That materially widens the class of models on
    which alignment-based defences reading attention matrices are defeatable.
    """
    cfg = model.config
    H   = cfg.num_attention_heads
    KV  = getattr(cfg, "num_key_value_heads", H) or H
    hd  = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
    grp = H // KV
    n = 0
    for li, layer in enumerate(_layers(model)):
        a = layer.self_attn
        with torch.no_grad():
            for kv in range(KV):
                g = _gen(seed * 7000 + li * 100 + kv)
                M = torch.matrix_exp(torch.randn(hd, hd, generator=g) * log_sigma)
                Minv = torch.linalg.inv(M)
                vs = kv * hd
                a.v_proj.weight[vs:vs + hd, :] = (M.T @ a.v_proj.weight[vs:vs + hd, :].float()).to(a.v_proj.weight.dtype)
                if a.v_proj.bias is not None:
                    a.v_proj.bias[vs:vs + hd] = (a.v_proj.bias[vs:vs + hd].float() @ M).to(a.v_proj.bias.dtype)
                # every q-head sharing this kv-head reads the same V, so o_proj's matching
                # column blocks all take M^-1
                for j in range(grp):
                    h = kv * grp + j
                    os_ = h * hd
                    # out = attn @ W_o.T, so head block h contributes attn_h @ W_o[:,blk].T.
                    # If attn_h -> attn_h @ M then we need W_o_new[:,blk].T = M^-1 @ W_o_old[:,blk].T,
                    # i.e. W_o_new[:,blk] = W_o_old[:,blk] @ M^-T  (NOT @ M^-1 -- that was the bug).
                    a.o_proj.weight[:, os_:os_ + hd] = (a.o_proj.weight[:, os_:os_ + hd].float() @ Minv.T).to(a.o_proj.weight.dtype)
        n += 1
    return {"transform": "X3ov", "log_sigma": log_sigma, "seed": seed, "layers": n,
            "kv_heads": KV, "head_dim": hd, "note": "exact WITH RoPE (RoPE touches only Q,K)"}
