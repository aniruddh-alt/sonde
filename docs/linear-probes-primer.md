# Linear Probes

A presentation-ready primer on linear probes for LLM interpretability: what they are, how they are trained, where they are used, and what the key papers have shown.

---

## 1. What Is a Linear Probe?

A **linear probe** is a lightweight supervised classifier (or regressor) trained to predict a property $y$ from a frozen internal activation $h \in \mathbb{R}^d$ of a neural network. It is "linear" in the strict sense that the decision function is an *affine* map of activations — no non-linearity between the representation and the logit.

$$\hat{y} = \sigma(w^\top h + b), \qquad w \in \mathbb{R}^d,\; b \in \mathbb{R}$$

The linearity is load-bearing: a probe's success tells you the property is **linearly decodable**, which is a much stronger claim than "encoded somewhere." This tracks the **linear representation hypothesis** — that models represent features as *directions* in activation space.

**Origins.** Crystallized by [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644), who attached cheap linear classifiers to every intermediate layer of ConvNets as "diagnostic thermometers" for representation quality. Gradients do not flow back into the host network — probes only measure what is already there.

---

## 2. Architecture: What a Probe Looks Like

A probe is just a tensor $W \in \mathbb{R}^{C \times d}$ (plus optional bias). The architectural decisions that matter are **where you read from** and **how you fit $w$**.

### 2.1 Input → Output

```
           LLM (frozen weights)
   ┌─────────────────────────────────────────┐
   │  ... → Layer L-1 → Layer L → Layer L+1  │
   │                       │                 │
   │                       │  h_{L, t}       │
   │                       ▼                 │
   │                 ┌───────────┐           │
   │                 │ d-dim     │           │
   │                 │ activation│           │
   │                 └─────┬─────┘           │
   └───────────────────────┼─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │  Linear Probe:  ŷ = σ(wᵀh + b)   │
          │     w ∈ ℝᵈ    (learned)          │
          └────────────────┬─────────────────┘
                           │
                           ▼
                p(y = 1 | h)   ∈  [0, 1]
```

- **Input:** activation $h_{L,t} \in \mathbb{R}^d$ at (layer $L$, token position $t$) for a single forward pass. For Qwen2.5-0.5B, $d = 896$; for Llama-3-70B, $d = 8192$.
- **Output:** scalar probability (binary), softmax over $C$ classes, or a continuous target (e.g., structural-probe tree distance, latitude).
- **Implicit geometric claim:** there exists a hyperplane $\{h : w^\top h + b = 0\}$ that separates the concept. The direction $w/\|w\|$ doubles as a **concept vector** for steering and causal interventions.

### 2.2 Probe Variants

All are linear in $h$; they differ in objective and assumptions.

| Variant | Objective | Notes |
|---|---|---|
| **Logistic regression** | cross-entropy on $\sigma(w^\top h + b)$ | default in interp; calibrated only on the training distribution and only without class weighting (sonde trains with `pos_weight`, so read scores against the val-chosen threshold) |
| **Linear SVM** | hinge loss | max-margin; robust to outliers |
| **Ridge regression** | $\|y - Wh\|_2^2 + \lambda\|W\|_F^2$ | closed form; used for continuous targets |
| **Difference of means (DoM)** | $w = \mu_+ - \mu_-$, threshold at midpoint | no optimization; preferred for steering |
| **LDA / Fisher** | $w \propto \Sigma^{-1}(\mu_+ - \mu_-)$ | Bayes-optimal for Gaussian classes with shared $\Sigma$ |
| **Structural probe** | PSD metric $A = B^\top B$ over tree-distance | recovers parse trees (Hewitt & Manning 2019) |

### 2.3 Design Choices

| Knob | Options | Why it matters |
|---|---|---|
| **Layer $L$** | sweep across all layers | semantic features usually peak in middle/late layers |
| **Token position $t$** | last prompt token, last response token, mean-pool, specific tokens | changes the *construct* being measured (see §4) |
| **Read site** | residual stream, MLP out, attention head out | residual stream is the default "canvas" |
| **Normalization** | raw, final RMSNorm/LayerNorm applied | putting $h$ in unembedding geometry |
| **Capacity** | strict linear, $k$-sparse, low-rank, MLP | trade decodability for interpretability |

### 2.4 Training

- **Base model frozen.** Activations are extracted once and cached (often as safetensors). Probes measure what is *already* present, not what gradient descent can cram in.
- **Small data.** Typical $N \in [10^3, 10^5]$. Because the probe is linear and $N \ll$ pretraining, overfitting dominates — strong $\ell_2$ regularization is standard.
- **Selection via held-out set.** Sweeping (layer × position × weight-decay) inflates effective capacity; always keep a final test split untouched.
- **Better metrics than accuracy.** [MDL probing](https://arxiv.org/abs/2003.12298) (Voita & Titov 2020) and conditional $\mathcal{V}$-information ([Hewitt et al. 2021](https://arxiv.org/abs/2109.09234)) account for probe capacity; **selectivity** against random-label control tasks ([Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)) isolates representation content from probe memorization.

---

## 3. Where Linear Probes Are Used

Three broad roles, each producing a different kind of insight:

1. **Diagnostic readout.** "Is property $P$ linearly decodable at layer $L$?" — probes used to *characterize* the model's internal representations (Alain & Bengio; Tenney et al. on the BERT pipeline; Gurnee & Nanda's sparse-probing case studies).
2. **Concept discovery.** "Which features does this model represent?" — probes as a scientific instrument, often swept across layers and token positions to find when/where a concept crystallizes (Gurnee & Tegmark on space/time; Li et al. on Othello board state).
3. **Runtime monitor.** "Is concept $C$ active *right now* during generation?" — cheap per-token classifiers threshold-able into safety guardrails (Anthropic sleeper-agent probes; Apollo deception probes; hallucination detectors).

The same probe mathematics underwrites all three; only the deployment surface differs.

---

## 4. Input-Token vs Output-Token Probing

Where along the token axis you probe fundamentally changes the construct being measured. Conflating the two is the single most common source of misinterpreted probe results.

### 4.1 Probes on input (prompt) token representations tell us about...

- **Properties of the input itself**: task identity, entity types, syntactic structure, language, topic.
- **The model's comprehension state**: whether it *understands* the prompt as harmful, as asking for X, as containing a trigger phrase.
- **Factual truth-values of statements the model has read**: even when the model is about to *assert* the opposite ([Mallen & Belrose, Quirky LMs](https://arxiv.org/abs/2312.01037)).
- **Pre-decisional latent state**: what the model knows *before* committing to an action.

A probe trained on the last prompt token is an aggregated summary vector — cheap to read, but temporally collapsed.

### 4.2 Probes on output (response) token representations tell us about...

- **The unfolding trajectory**: what the model is about to say next.
- **Behavioral commitment**: whether it is about to refuse, fabricate, comply, or defect.
- **Real-time monitoring signals**: token-by-token hallucination or deception scores that can stream alongside generation.

### 4.3 The Refusal Confound (a worked example)

A probe trained at the last prompt token on refusal-labeled data is almost certainly a **harmful-prompt detector**, not a **refusal-behavior detector**. The label (did the model refuse?) is strongly correlated with a prompt feature (is the prompt harmful?), and the prompt feature is *fully determined before generation begins*.

Disambiguation requires:
1. Probing at response tokens, or at minimum the first generated token after the decision gate.
2. Including **jailbreaks** (harmful prompts complied with) and **over-refusals** (harmless prompts refused).
3. Testing transfer: a behavior probe should fire on compliant-then-refusing trajectories and *not* on harmful-but-complied ones.

The same logic applies to deception, sycophancy, and scheming probes: if the probe site precedes the behavior, you are measuring antecedents, not the behavior itself.

### 4.4 Layerwise Probing

A sweep over layers is cheap insurance. [Tenney et al. (2019)](https://arxiv.org/abs/1905.05950) showed BERT "rediscovers the classical NLP pipeline": in BERT-large (24 layers) the regions that matter for each task appear in the order POS tagging, parsing, NER, semantic roles, then coreference. The ordering is the finding; the regions overlap, and the paper makes no claim about decoder-only LLMs. *Flat* probe-accuracy curves suggest a feature is present at the input and merely propagated; sharp *late-layer peaks* suggest genuine in-network computation.

---

## 5. Key Papers (Problem / Method / Findings)

### 5.1 [Marks & Tegmark (2023) — The Geometry of Truth](https://arxiv.org/abs/2310.06824)

- **Problem.** Are "truth" probe directions *causally* load-bearing, or merely correlational? Do different probe methods find the same direction?
- **Method.** Curate clean true/false factual datasets; compare logistic regression, mass-mean (DoM) and CCS probes on LLaMA-2-7B/13B/70B residual streams (PCA is used only to visualise); perform causal interventions by adding/subtracting the direction.
- **Findings.** Mass-mean probes generalise about as well as LR and CCS, and their directions are the most causally implicated (MM beats LR and CCS in 7 of 8 intervention settings). Probes transfer across factual domains → shared truth representation. Intervention flips the model's downstream judgments, establishing causation. Middle-to-late layers separate true vs false cleanly.

### 5.2 [Li et al. (2023) — Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341)

- **Problem.** LLMs often "know" the right answer but output falsehoods (hallucination, sycophancy). Can we steer truthfulness without fine-tuning?
- **Method.** Train per-head linear probes on TruthfulQA-style contrasts. Rank heads by probe accuracy. At inference, shift the top-$K$ heads by $\alpha$ along the mass-mean shift (the class-mean difference), which beat the probe-weight direction in their ablation (42.3% vs 34.8% true*informative).
- **Findings.** TruthfulQA truthfulness on LLaMA-Alpaca from 32.5% → 65.1%. Truth signal is sparsely localized in a few heads. Hundreds of labeled examples suffice. Popularized activation steering as a cheap alignment tool.

### 5.3 [Burns et al. (2022) — Contrast-Consistent Search (CCS)](https://arxiv.org/abs/2212.03827)

- **Problem.** All prior probing needs ground-truth labels — unavailable exactly when we most need them (superhuman or deceptive models).
- **Method.** Feed model "X is true" and "X is false"; extract hidden states for both. Train a probe $p$ to satisfy (a) *consistency*: $p(x^+) + p(x^-) \approx 1$, and (b) *confidence*: outputs pushed from 0.5. No labels, no generation.
- **Findings.** ~4% above zero-shot on 10 QA datasets across 6 models. Maintains accuracy even when the model is prompted to lie — evidence it reads internal *belief* rather than output behavior. Foundational for unsupervised ELK-style oversight.

### 5.4 [Arditi et al. (2024) — Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

- **Problem.** Is refusal implemented by complex distributed circuitry, or by something simpler — and therefore surgically removable?
- **Method.** Per-layer DoM on harmful vs harmless instructions. Two interventions: (i) **directional ablation** — project out the direction from the residual stream at inference; (ii) **directional addition** — inject into benign prompts. Tested on 13 open-weight chat models up to 72B.
- **Findings.** A single 1-D subspace mediates refusal across all 13 models. Ablation causes compliance with harmful instructions while preserving MMLU-level capabilities. Addition induces refusal on benign prompts. Adversarial-suffix jailbreaks work by suppressing propagation of this same direction — unifying white-box and black-box attacks.

### 5.5 [Zou et al. (2023) — Representation Engineering (RepE)](https://arxiv.org/abs/2310.01405)

- **Problem.** Bottom-up mech-interp (circuits, neurons) scales poorly. Can we build a transparency toolkit at the level of population-level representations for high-level concepts (honesty, power-seeking, emotion)?
- **Method.** **Linear Artificial Tomography (LAT)**: elicit activations with paired stimuli (honest vs dishonest responses to same prompt); extract a *reading vector* per concept via PCA / DoM over contrastive differences; use the same vectors for *control* by residual-stream addition.
- **Findings.** Reading vectors detect concepts like honesty with high accuracy. Control vectors steer model behavior — dramatic TruthfulQA gains, reduced harmful outputs, modulated emotional tone — using only contrastive pairs. Canonical reference for "top-down" interpretability.

### 5.6 [Gurnee et al. (2023) — Finding Neurons in a Haystack](https://arxiv.org/abs/2305.01610)

- **Problem.** Full-rank probes can memorize or exploit weak distributed signals. Where exactly do interpretable features live?
- **Method.** $k$-sparse linear probes (constrained to use at most $k$ neurons) on Pythia 70M–6.9B. Sweep $k$ from 1 upward across 100+ features (language identity, programming, compound words, base-64).
- **Findings.** Many high-level features are carried by **monosemantic single neurons** in middle layers (a French-language neuron, an "is-python" neuron). Early layers use sparse superposition. Representation sparsity generally increases with scale. Strongest pre-SAE evidence for superposition.

### 5.7 [Park, Choe & Veitch (2023) — The Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658)

- **Problem.** "Linear representation" is used loosely: probe-decodability vs steering-effectiveness vs concept-arithmetic (king − man + woman). These need not coincide.
- **Method.** Formalize linear representation via counterfactual pairs. Distinguish output/unembedding space from input/embedding space. Prove they align under a particular non-Euclidean **causal inner product** derived from the unembedding matrix.
- **Findings.** Under the causal inner product, (i) DoM on counterfactual pairs provably recovers the causal direction; (ii) probe and intervention directions align; (iii) unrelated concepts become orthogonal, enabling compositional steering. Theoretical scaffolding for why Marks & Tegmark, Arditi, and RepE work.

---

## 6. Steering Vectors: How Linear Probes Learn Feature Directions

A trained probe weight $w \in \mathbb{R}^d$ is the normal vector to the decision hyperplane. If the feature is *linearly represented*, $w$ approximates the direction along which that feature varies — so adding $\alpha \cdot \tilde{w}$ to the residual stream at inference should push the model's state further into the "positive" half-space and produce behavior consistent with the feature being present.

### 6.1 The Steering Operation

$$h' = h + \alpha \cdot \tilde{w}, \qquad \tilde{w} = w / \|w\|$$

The mirror operation is **directional ablation** — project $h$ onto the hyperplane orthogonal to $\tilde{w}$ to *erase* the feature:

$$h' = h - (\tilde{w}^\top h)\, \tilde{w}$$

[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) even bake this into the weights via **weight orthogonalization** — projecting every matrix that writes to the residual stream to have no component along the refusal direction.

### 6.2 Why Difference-of-Means Steers Well

For Gaussian classes with shared covariance $\Sigma$, the Bayes-optimal linear classifier is Fisher's direction $w^* = \Sigma^{-1}(\mu_+ - \mu_-)$. DoM equals it only when $\Sigma \propto I$, so DoM is usually the weaker classifier. Logistic regression is pulled toward any axis that gives low-noise separability, including axes merely *correlated* with the feature. DoM ignores $\Sigma$ and points where the class means actually differ, which is what an intervention moves. Empirically, [Marks & Tegmark](https://arxiv.org/abs/2310.06824) find mass-mean directions more causally implicated than LR directions at similar accuracy, and [ITI](https://arxiv.org/abs/2306.03341) steers best along the mass-mean shift.

### 6.3 Probe Direction ≠ Feature Direction

Two senses of "linear" often conflated:

- **Linear in classifier weight space.** The decoder is $\sigma(w^\top h + b)$. Trivial — says nothing about the model's geometry.
- **Linearly *represented* in activation space.** There exists a direction $v$ such that varying the feature moves $h$ along $v$. Empirical claim about the model.

LR probes satisfy the first always, the second approximately. A high-AUROC probe can still fail to steer because $w_{LR} \neq v_{\text{feature}}$ whenever confounds exist.

### 6.4 Probes Prove Correlation; Steering Proves Causation

A probe's success is *observational* evidence: the feature is linearly decodable. It does not prove the model *uses* the feature downstream. Steering is a **causal test**: if $+\alpha \tilde{w}$ produces the predicted behavioral change and $-\alpha \tilde{w}$ produces its opposite, the direction is causally implicated.

### 6.5 Relevant Steering Methods

- [**ActAdd** (Turner et al. 2023)](https://arxiv.org/abs/2308.10248) — steer from contrasting *prompt pairs* alone; no optimization.
- [**CAA** (Panickssery et al. 2023)](https://arxiv.org/abs/2312.06681) — Contrastive Activation Addition, averaging DoM over many (positive, negative) behavior pairs.
- [**LEACE** (Belrose et al. 2023)](https://arxiv.org/abs/2306.03819) — closed-form erasure that provably stops every linear classifier from detecting a concept while changing the representation as little as possible.

---

## 7. Examples

### 7.1 LLM Monitors (Safety Probes)

Linear probes have become the leading runtime monitoring technique: a single dot product per cached hidden state is cheap enough to run on every forward pass, and the scalar score is easy to threshold and audit.

**Sleeper-agent / deception detection.**
- [MacDiarmid et al. (Anthropic, 2024) — *Simple probes can catch sleeper agents*](https://www.anthropic.com/research/probes-catch-sleeper-agents). Probes trained on a crude "Are you helpful? Yes/No" contrast pair catch behavioral defection in sleeper-agent models at AUROC > 99%, even without knowing the trigger. Defection is linearly legible across wide bands of middle layers.
- [Goldowsky-Dill et al. (Apollo Research, 2025)](https://arxiv.org/abs/2502.03407). Deception probes on Llama-3.3-70B-Instruct: AUROC 0.96-0.999 on held-out scenarios (insider-trading concealment, sandbagging). Catch 95-99% of deceptive responses at a threshold set for 1% FPR on unrelated control chat data — still insufficient as a robust defense, but a strong baseline.

**Hallucination / truthfulness.**
- [Azaria & Mitchell (2023) — SAPLMA](https://arxiv.org/abs/2304.13734). Classifier over hidden activations predicts statement truth at 71-83% accuracy, above token-probability baselines confounded by length and frequency.
- [Kadavath et al. (2022) — *Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221). P(True) is the model's own probability that a proposed answer is true; P(IK) comes from a head fine-tuned with the model, so neither is a probe on frozen activations. Large models are well calibrated on multiple-choice and true/false questions in the right format.

**Harmfulness / jailbreak.**
- [RepE (Zou et al. 2023)](https://arxiv.org/abs/2310.01405) harmfulness probes from single contrast pairs.

### 7.2 Feature Discovery

Used as a scientific instrument, probes answer *what does this model know?*

**Sparse neuron-level features.** [Gurnee et al.](https://arxiv.org/abs/2305.01610) found dedicated monosemantic neurons for French text, code, base64, all-caps, and compound words in Pythia middle layers.

**World models.** [Li et al. *Emergent World Representations* (ICLR 2023)](https://arxiv.org/abs/2210.13382) recovered OthelloGPT's board state with *nonlinear* (MLP) probes, and intervening on the probed representation flips predicted legal moves. [Nanda, Lee & Wattenberg (2023)](https://arxiv.org/abs/2309.00941) then showed the state is *linear* once squares are read as "mine" vs "theirs" instead of black vs white. Probes don't just *read* a world model — they identify the one the network *uses*.

**Spatial and temporal structure.** [Gurnee & Tegmark (2023) — *Language Models Represent Space and Time*](https://arxiv.org/abs/2310.02207). Ridge probes recover latitude/longitude and dates from Llama-2 activations linearly, with $R^2$ rising through the first half of the layers, plateauing near the middle, and higher in larger models. Individual "space neurons" and "time neurons" encode coordinates directly.

**Layerwise pipeline discovery.** [Tenney et al. (2019)](https://arxiv.org/abs/1905.05950) used edge-probing scalar-mixing weights to show BERT rediscovers the classical NLP pipeline across depth.

### 7.3 Probes vs Sparse Autoencoders (SAEs)

Probes and SAEs are **complementary**, not competitive.

| | Linear probe | SAE |
|---|---|---|
| Supervision | Supervised (need labels) | Unsupervised |
| Scope | Targeted (one concept at a time) | Exhaustive (thousands of features) |
| Cost | One matrix-vector product | Full autoencoder per layer |
| Use when... | You know what to look for | You want to discover what's there |

An SAE feature that *underperforms* a linear probe on its own purported concept is suspect. Probes remain the gold-standard baseline against which SAE features are judged.

---

## 8. Limitations and Pitfalls

1. **Probe capacity confound.** A high-accuracy MLP probe may have learned the task itself from random features. Report **selectivity** against word-type-randomized control tasks ([Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)); prefer linear probes; use MDL.
2. **Correlation ≠ causation.** Linear separability does not prove the model uses the direction. Complement with activation patching, nullspace removal, or steering.
3. **Distributional artifacts.** Surface cues (length, topic, tokens) can drive probe accuracy. Matched-control datasets and balanced sampling are essential.
4. **Prompt vs response site.** Probes at different token positions measure different constructs (§4).
5. **Adversarial manipulation.** Probes can be defeated by adversarial activation perturbations that flip the probe score while preserving behavior.
6. **Selection inflation.** Sweeping (layer × position × weight-decay) inflates effective capacity. Hold out a final test split never used for selection.
7. **Linear vs nonlinear debate.** The field has largely converged on linear (or rank-constrained) probes plus controls as the defensible default, with MLPs reserved for upper-bound "is the information present at all?" questions.

---

## 9. One-Sentence Summary

> A linear probe is a frozen-model readout that tests whether a concept is linearly decodable from internal activations — cheap enough to run on every token, theoretically grounded by the linear representation hypothesis, and causally validated by the same weight vector doubling as a steering direction.

---

## References

Foundational:
- Alain & Bengio (2016). *Understanding intermediate layers using linear classifier probes*. [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
- Conneau et al. (2018). *What you can cram into a single vector*. [arXiv:1805.01070](https://arxiv.org/abs/1805.01070)
- Hewitt & Manning (2019). *A Structural Probe for Finding Syntax in Word Representations*. [NAACL 2019](https://aclanthology.org/N19-1419/)
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks*. [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)
- Tenney et al. (2019). *BERT Rediscovers the Classical NLP Pipeline*. [arXiv:1905.05950](https://arxiv.org/abs/1905.05950)
- Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances*. [arXiv:2102.12452](https://arxiv.org/abs/2102.12452)
- Hewitt et al. (2021). *Conditional probing: measuring usable information beyond a baseline*. [arXiv:2109.09234](https://arxiv.org/abs/2109.09234)

Modern LLM probing:
- Burns et al. (2022). *Discovering Latent Knowledge Without Supervision (CCS)*. [arXiv:2212.03827](https://arxiv.org/abs/2212.03827)
- Li et al. (2023). *Inference-Time Intervention (ITI)*. [arXiv:2306.03341](https://arxiv.org/abs/2306.03341)
- Gurnee et al. (2023). *Finding Neurons in a Haystack*. [arXiv:2305.01610](https://arxiv.org/abs/2305.01610)
- Marks & Tegmark (2023). *The Geometry of Truth*. [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
- Zou et al. (2023). *Representation Engineering*. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Park, Choe & Veitch (2023). *The Linear Representation Hypothesis*. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)
- Arditi et al. (2024). *Refusal Is Mediated by a Single Direction*. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

Steering and erasure:
- Subramani, Suresh & Peters (2022). *Extracting Latent Steering Vectors*. [arXiv:2205.05124](https://arxiv.org/abs/2205.05124)
- Turner et al. (2023). *Activation Addition (ActAdd)*. [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Panickssery et al. (2023). *Contrastive Activation Addition (CAA)*. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
- Belrose et al. (2023). *LEACE: Perfect linear concept erasure in closed form*. [arXiv:2306.03819](https://arxiv.org/abs/2306.03819)

Applications:
- Kadavath et al. (2022). *Language Models (Mostly) Know What They Know*. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
- Azaria & Mitchell (2023). *The Internal State of an LLM Knows When It's Lying (SAPLMA)*. [arXiv:2304.13734](https://arxiv.org/abs/2304.13734)
- Li et al. (2023). *Emergent World Representations (OthelloGPT)*. [arXiv:2210.13382](https://arxiv.org/abs/2210.13382)
- Nanda, Lee & Wattenberg (2023). *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*. [arXiv:2309.00941](https://arxiv.org/abs/2309.00941)
- Gurnee & Tegmark (2023). *Language Models Represent Space and Time*. [arXiv:2310.02207](https://arxiv.org/abs/2310.02207)
- MacDiarmid et al. (Anthropic, 2024). *Simple probes can catch sleeper agents*. [Anthropic blog](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- Mallen & Belrose (2023). *Eliciting Latent Knowledge from Quirky Language Models*. [arXiv:2312.01037](https://arxiv.org/abs/2312.01037)
- Goldowsky-Dill et al. (Apollo Research, 2025). *Detecting Strategic Deception Using Linear Probes*. [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)
