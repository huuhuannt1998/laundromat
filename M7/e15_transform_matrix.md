# E15 — consolidated transform matrix

49 arms over 8 experiment files. `capability` is the predeclared criterion: max|dlogit| <= 1e-2 in float32 on a fixed probe set.

| arm | transform | child | sig_id | pipeline | tier | verdict | capability |
|---|---|---|---|---|---|---|---|
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.8454 | 0.8454 | 3 | High-Confidence Match | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.8441 | 0.8441 | 3 | High-Confidence Match | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.837 | 0.837 | 3 | High-Confidence Match | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.7672 | 0.7672 | 3 | High-Confidence Match | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.5793 | 0.5793 | 3 | Not Matched | unmeasured |
| anchor_attack_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X5anchor+M1t |  | 0.4985 | 0.4985 | 3 | Not Matched | unmeasured |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | M1t | SmolLM2-135M-Instruct | 0.9992 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9936 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.988 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9767 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9559 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9337 | 0.9 | 2 | Confirmed Match | exact |
| frac_sweep_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9116 | 0.9 | 2 | Confirmed Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | S0-identity | SmolLM2-135M-Instruct | 0.9992 | 1.0 | 1 | Confirmed Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | M1t | SmolLM2-135M-Instruct | 0.9992 | 0.9 | 2 | Confirmed Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X2 | SmolLM2-135M-Instruct | 0.9518 | 1.0 | 1 | Confirmed Match | BROKEN (0.247) |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X2+M1t | SmolLM2-135M-Instruct | 0.9518 | 0.9518 | 3 | High-Confidence Match | BROKEN (0.247) |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X1b | SmolLM2-135M-Instruct | 0.9334 | 1.0 | 1 | Confirmed Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X1a | SmolLM2-135M-Instruct | 0.9116 | 1.0 | 1 | Confirmed Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+M1t | SmolLM2-135M-Instruct | 0.9116 | 0.9116 | 3 | High-Confidence Match | exact |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X2 | SmolLM2-135M-Instruct | 0.8137 | 1.0 | 1 | Confirmed Match | BROKEN (0.247) |
| gate_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+X2+M1t | SmolLM2-135M-Instruct | 0.8137 | 0.8137 | 3 | High-Confidence Match | BROKEN (0.247) |
| gate_Qwen--Qwen2.5-0.5B-Instruct | S0-identity | Qwen2.5-0.5B-Instruct | 0.9994 | 1.0 | 1 | Confirmed Match | exact |
| gate_Qwen--Qwen2.5-0.5B-Instruct | M1t | Qwen2.5-0.5B-Instruct | 0.9994 | 0.9 | 2 | Confirmed Match | exact |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X2 | Qwen2.5-0.5B-Instruct | 0.9688 | 1.0 | 1 | Confirmed Match | BROKEN (0.335) |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X2+M1t | Qwen2.5-0.5B-Instruct | 0.9688 | 0.9688 | 3 | High-Confidence Match | BROKEN (0.335) |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X1b | Qwen2.5-0.5B-Instruct | 0.9269 | 1.0 | 1 | Confirmed Match | exact |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X1a | Qwen2.5-0.5B-Instruct | 0.9238 | 1.0 | 1 | Confirmed Match | exact |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X1a+M1t | Qwen2.5-0.5B-Instruct | 0.9238 | 0.9238 | 3 | High-Confidence Match | exact |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X1a+X1b+X2 | Qwen2.5-0.5B-Instruct | 0.8375 | 1.0 | 1 | Confirmed Match | BROKEN (0.336) |
| gate_Qwen--Qwen2.5-0.5B-Instruct | X1a+X1b+X2+M1t | Qwen2.5-0.5B-Instruct | 0.8375 | 0.8375 | 3 | High-Confidence Match | BROKEN (0.336) |
| m2t_attack | M2t-overlap0.90 |  | 0.9889 | 1.0 | 1 | Confirmed Match | unmeasured |
| m2t_composed | M2t(overlap=0.0) |  | 0.974 | 1.0 | 1 | Confirmed Match | unmeasured |
| m2t_composed | X1a |  | 0.9116 | 1.0 | 1 | Confirmed Match | unmeasured |
| m2t_composed | M2t(overlap=0.0)+X1a |  | 0.8864 | 1.0 | 1 | Confirmed Match | unmeasured |
| m2t_composed | M2t(overlap=0.0)+X1a+M1t+M3t |  | 0.8864 | 0.8864 | 3 | High-Confidence Match | unmeasured |
| q1_baseline | q1_baseline |  | 0.9992 | 1.0 | 1 | Confirmed Match | unmeasured |
| q1_baseline | q1_baseline |  | 0.9992 | 1.0 | 1 | Confirmed Match | unmeasured |
| q1_baseline | q1_baseline |  | 0.9991 | 1.0 | 1 | Confirmed Match | unmeasured |
| q1_baseline | q1_baseline |  | 0.9966 | 1.0 | 1 | Confirmed Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct |  | 0.9992 | 1.0 | 1 | Confirmed Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8458 | 0.8458 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8456 | 0.8456 | 3 | High-Confidence Match | unmeasured |
| sweep_embed_HuggingFaceTB--SmolLM2-135M-Instruct | X1a+X1b+N1emb+M1t |  | 0.8448 | 0.8448 | 3 | High-Confidence Match | unmeasured |

- arms breaking the capability criterion: **8**
- arms with identity score below the 0.5298 evasion bar: **1**
- of those, demonstrably capability-preserving: **0**
- of those, capability unmeasured in this arm's file: **1**
