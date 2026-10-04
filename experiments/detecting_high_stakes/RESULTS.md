# High-Stakes Detection with Activation Probes

Replication and extension of ["Detecting High-Stakes Interactions with Activation Probes"](https://arxiv.org/abs/2506.10805) (Cambridge, Harvard, UCL, LASR Labs -- ICML 2025 Workshop).

## Paper Summary

The paper trains lightweight classifiers ("probes") on a model's internal activations to detect whether a user prompt is high-stakes (medical, legal, financial, safety-critical) or low-stakes (casual, routine). The probe reads the residual stream during the model's normal forward pass at zero additional cost, replacing expensive LLM-as-judge classifiers with a tiny classifier that achieves comparable accuracy at six orders of magnitude less compute.

**Paper setup:** Llama 3.3 70B, layer 31 (of 80), 6 probe architectures, 10K synthetic training samples, evaluated on 6 external OOD datasets.

**Paper result:** Mean AUROC >0.91 across OOD datasets, matching 8B-12B LLM classifiers.

## Our Setup

| Parameter | Paper | Run v1 (last-token) | Run v6 (full-sequence) |
|-----------|-------|---------------------|------------------------|
| Model | Llama 3.3 70B | Llama 3.1 8B | Llama 3.1 8B |
| Layers probed | All 80 | 8-24 (17 layers) | 10, 13, 15, 17, 20 |
| Best layer | 31 (~39% depth) | 15 (~47% depth) | 10 (~31% depth) |
| Probe architecture | Attention (best) | Attention (4 heads) | Attention (4 heads, einops) |
| Token selection | All tokens | Last token only | All tokens |
| Training samples | 10,000 | 8,000 | 1,000 |
| Dataset | Arrrlex/models-under-pressure | Same | Same (stratified subsample) |
| Cluster | -- | Lambda K8s (H100 SXM) | Lambda K8s (H100 SXM) |

## Results

### Run v1: Last-Token Probing (8K samples, 17 layers)

| Layer | val_auroc | val_acc |
|-------|-----------|---------|
| 8 | 0.9870 | 0.939 |
| 12 | 0.9887 | 0.950 |
| **15** | **0.9917** | **0.960** |
| 18 | 0.9907 | 0.952 |
| 24 | 0.9884 | 0.943 |

**Best layer:** 15, test AUROC = 0.992, accuracy = 95.4%

**Controls:** Shuffled labels AUROC = 0.528, random features AUROC = 0.498 (chance level, confirming real signal).

### Run v6: Full-Sequence Probing (1K samples, 5 layers)

| Layer | val_auroc | val_acc |
|-------|-----------|---------|
| **10** | **0.9988** | **0.987** |
| 13 | 0.9979 | 0.980 |
| 15 | 0.9977 | 0.974 |
| 17 | 0.9979 | 0.967 |
| 20 | 0.9951 | 0.960 |

**Best layer:** 10, test AUROC = 0.988, accuracy = 95.4%

Full-sequence attention probing achieves higher validation AUROC (0.999 vs 0.992) with 8x fewer samples, because the attention mechanism can weight informative tokens across the full sequence rather than relying solely on the last token.

### OOD Evaluation (Run v6)

Probe trained on 1K synthetic samples, evaluated on 5 external datasets it has never seen:

| Dataset | Domain | Samples | AUROC | Accuracy | F1 |
|---------|--------|---------|-------|----------|-----|
| **In-distribution (test)** | Synthetic | 153 | **0.988** | **0.954** | **0.954** |
| Mental Health | Doctor-patient transcripts | 540 | **0.964** | **0.852** | 0.827 |
| MT-Bench | Multi-turn conversations | 604 | **0.955** | **0.851** | 0.826 |
| ToolACE | Tool-use API calls | 734 | 0.807 | 0.510 | 0.037 |
| Anthropic HH | RLHF dialogue pairs | 2,984 | 0.683 | 0.527 | 0.158 |
| Aya Red-teaming | Multilingual adversarial | 1,242 | 0.613 | 0.504 | 0.016 |

### Interpretation

**Strong generalization (AUROC >0.95):** Mental Health and MT-Bench share similar conversational structure with training data. The probe transfers well to these domains.

**Partial generalization (AUROC ~0.8):** ToolACE has moderate AUROC (0.81) but near-zero F1 -- the probe can rank inputs by risk but the decision threshold from synthetic training doesn't transfer. Threshold recalibration on a few domain examples would likely fix this.

**Poor generalization (AUROC ~0.6):** Anthropic HH (Reddit-sourced dialogue) and Aya (multilingual red-teaming) are fundamentally different distributions. This confirms the "False Sense of Security" finding (Wang et al., 2509.03888) that probes learn surface patterns specific to their training distribution.

## How to Run

```bash
sonde run high_stakes   # .venv-hf; the vllm backend is experimental
```

The recipe (`sonde/recipes/high_stakes.yaml`):
1. Loads a seeded 1,000-row subsample of `Arrrlex/models-under-pressure` (`training`).
2. Extracts every token of the raw (untemplated) input at blocks 10, 13, 15, 17 and 20 of Llama-3.1-8B.
3. Fits one attention probe per block and selects by validation recall at 1% FPR (validation AUROC when val has too few negatives for 1% FPR).
4. Records a shuffled-label control.
5. Scores the selected probe on the five `*_balanced` OOD test splits.
6. Writes `runs/high_stakes/probes/high_stakes.npz` plus `metrics.json` and `score/*/metrics.json`.

The numbers above were produced by sonde v0.1 (tag `v0.1-legacy`) at batch
size 1. They have not yet been reproduced with the current pipeline.

## Key Takeaways

1. **Probes work on smaller models.** Llama 3.1 8B achieves 0.988 AUROC, comparable to the paper's 70B results on in-distribution data.
2. **Full-sequence > last-token.** Attention over all tokens yields 0.999 val AUROC with only 1K samples vs 0.992 with 8K samples and last-token only.
3. **OOD is the real test.** In-distribution AUROC (0.988) drops to 0.61-0.96 on external datasets, with severe failure on distributions far from synthetic training data.
4. **1K samples suffice.** Consistent with the paper's finding that even 32 deployment-domain samples improve beyond synthetic-only training.

## References

- [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) (2506.10805)
- [Dataset: Arrrlex/models-under-pressure](https://huggingface.co/datasets/Arrrlex/models-under-pressure)
- Open extensions: [spec §17](../../docs/superpowers/specs/2026-10-02-sonde-redesign-design.md#17-deferred-add-when-a-config-needs-it)
