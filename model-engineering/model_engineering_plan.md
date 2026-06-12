# Model Engineering Upskilling Plan — SLMs in the Life Sciences Domain

*A sequenced learning + project plan to move from "consumer of pretrained models" to "engineer who can build, fine-tune, and reason about the scaling and bottlenecks of small language models."*

---

## 0. Framing: what this plan is closing

Your existing strengths are real and you should not re-learn them:

- Statistical / classical ML and the math underneath it
- Domain fluency in genomics, NGS, variant calling, protein–DNA binding, multi-omics
- Using pretrained models as components (RAG, document decomposition, BioBERT/BioNeMo tooling)
- Production systems thinking (incident handling, pipelines)

The gap is **everything that happens *inside* the model boundary** — the part that's currently a black box:

1. **Mechanics** — what a training step actually does (forward, loss, backward, optimizer step), how a tokenizer + model + data loader fit together, what the artifacts of training are.
2. **Adaptation methods** — full fine-tuning vs. PEFT (LoRA/QLoRA), when each is appropriate, and what they cost.
3. **The physics of scale** — where memory goes, where compute goes, what the bottleneck actually is at a given model size, and how performance responds to more parameters / more data / more compute.
4. **Experiment hygiene at the model level** — eval design, tracking, reproducibility (this is where your stats background gives you a head start).

The plan front-loads the learning, then runs a single project that forces you to *use* all of it on a problem in your own domain.

**Time shape:** roughly 4 learning phases (~3–4 weeks part-time) then a 5-milestone project (~3–4 weeks part-time). Compress or expand freely — the dependency order matters more than the calendar.

**Hardware reality check:** everything here is doable on a single consumer GPU (8–16 GB), free Colab/Kaggle T4, or even CPU for the smallest model variants. You do *not* need a cluster to get the intuition. The project is specifically designed around model sizes that fit modest hardware.

---

## Part A — Learning Phase

### Phase 0 — Transformer internals from scratch (≈4–6 hours)

**Goal:** never again treat the model as opaque. Build a tiny transformer end to end once, by hand.

- **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (video + the `nanoGPT` repo). Type it out; don't just watch. By the end you should be able to point at attention, the residual stream, layernorm, the MLP block, and the LM head and say what each does.
- **"The Annotated Transformer"** (Harvard NLP) as a reference for the original architecture with code alongside the paper.
- Optional but worth it: **Karpathy's `makemore` series** for the optimizer/training-loop fundamentals before the GPT video.

**Checkpoint:** you can explain, in one paragraph each, (a) what the optimizer state is and why Adam needs ~2 extra copies of the params, (b) why activations dominate memory at long sequence length, (c) what the KV cache is and when it matters. (You'll *verify* these empirically in the project — here you just need the mental model.)

### Phase 1 — The fine-tuning stack (≈6–8 hours)

**Goal:** fluency with the Hugging Face ecosystem as a working toolkit.

- **`transformers`** — `AutoModel*`, `AutoTokenizer`, `Trainer` / `TrainingArguments`. Run one tiny supervised fine-tune of a small model end to end on a toy dataset just to see the loop work.
- **`datasets`** — streaming, mapping, tokenization, batching/collation.
- **`peft`** — LoRA and QLoRA. Read the LoRA paper (Hu et al., 2021) and the QLoRA paper (Dettmers et al., 2023); they're short and high-leverage. Understand *rank*, *alpha*, *target modules*, and why LoRA trades a tiny number of trainable params for most of the quality.
- **`trl`** — skim SFTTrainer; you'll likely use it. (DPO/PPO are out of scope for now — note they exist, move on.)
- Tracking: set up **Weights & Biases** (or TensorBoard) now and use it for everything after. Plotting loss curves and resource metrics is half the learning.

**Checkpoint:** you can articulate the decision tree "full fine-tune vs. LoRA vs. QLoRA vs. frozen-features-+-linear-probe" and what each costs in trainable parameters and memory.

### Phase 2 — The physics of scale (≈8–10 hours)

**Goal:** be able to predict and explain *where the wall is* before you hit it. This is the heart of what you asked for.

- **Memory accounting.** Learn the four consumers: (1) model parameters, (2) gradients, (3) optimizer states, (4) activations — plus KV cache at inference. Be able to estimate each for a given model in fp16/bf16 vs. 4-bit. The rule of thumb "full fp16 Adam fine-tune needs ~16 bytes/param" should feel obvious by the end.
- **Throughput levers.** Mixed precision (bf16/fp16), gradient accumulation, gradient checkpointing (trades compute for activation memory), **FlashAttention** (memory-linear attention), **sequence packing**, and **4-bit quantization**. Recent protein-LM work is a clean real-world demonstration: FlashAttention + sequence packing gave large inference speedups and memory reductions, and 4-bit quantization of billion-parameter models cut memory further while preserving accuracy on variant-effect prediction — you'll reproduce a small version of this.
- **Distributed training — conceptually.** Data parallel vs. model/tensor parallel vs. pipeline parallel; **DeepSpeed ZeRO** stages and **FSDP** (what each shards: ZeRO-1 optimizer states, ZeRO-2 + gradients, ZeRO-3 + parameters). You won't run multi-node, but you must be able to explain the tradeoffs. Read the ZeRO paper's intro/figures.
- **Scaling laws.** Read **Chinchilla** (Hoffmann et al., 2022) for the compute-optimal data/params tradeoff, and skim Kaplan et al. (2020). The key transferable idea: more parameters is not free quality — it must be matched with data and compute, and with *limited task data the bigger model can lose*. Hold this lightly; you'll test it directly in the project.

**Checkpoint:** given "fine-tune a 650M model on a 16 GB GPU," you can sketch a feasible recipe (precision, batch size, grad accumulation, LoRA vs. full, checkpointing) and justify each choice.

### Phase 3 — Evaluation & experiment hygiene (≈3–4 hours)

**Goal:** lean on your statistics strength to do this part *better* than most ML engineers.

- Train/val/test discipline for fine-tuning; data leakage traps specific to biological sequences (homology leakage between splits is the classic one — random splits overstate performance).
- Metric choice per task type (classification vs. regression vs. ranking/zero-shot), confidence intervals, seeds and variance across runs.
- Benchmark literacy for the project domain: **ProteinGym** (variant-effect prediction) and **FLIP** (fitness landscape tasks: stability, fluorescence, AAV).

**Checkpoint:** you can design an eval that wouldn't embarrass you in front of a reviewer, including how you'd report variance.

---

## Part B — The Project: a protein-LM fine-tuning + scaling study

**Why this project.** It's in your domain (proteins/variants — adjacent to your dissertation and Helix work), it spans a real model-size range on modest hardware, and it forces you through every concept above. It also produces something you can *talk about in interviews* — directly filling the "fine-tuning / foundation-model / generative-bio" gap that's shown up repeatedly in your applications.

**Model family:** ESM-2, available at `8M`, `35M`, `150M`, `650M`, `3B`, `15B` parameters (HuggingFace: `facebook/esm2_t6_8M_UR50D`, `..._t12_35M...`, `..._t30_150M...`, `..._t33_650M...`). Optionally **ESM-C (600M)** as an efficiency-optimized comparison point. The 8M–650M range is the sweet spot for a single-GPU scaling study.

**Task:** pick one supervised downstream task with a real benchmark. Good first choices (small, clean, well-documented):
- **Thermostability** (regression) — FLIP.
- **Subcellular localization** or **binary peptide property** (classification) — e.g., blood-brain-barrier peptide prediction.
- **Variant effect prediction** (zero-shot + supervised) — ProteinGym; closest to your Helix experience and the most "you" story to tell.

Start with the simplest (a single property task) and only move to variant effects once the pipeline works.

### Milestone 0 — Baseline: frozen embeddings + linear probe
Extract embeddings from a frozen ESM-2 (start with `35M`), train a simple classifier/regressor (logistic regression or a 1-layer MLP) on top. This is your "no fine-tuning" floor and reuses your classical-ML comfort zone. Log the metric and the wall-clock/memory.

### Milestone 1 — Full fine-tuning, smallest model
Full fine-tune `esm2_8M` or `35M` on the same task with the `Trainer`. Watch the loss curve in W&B. Compare against the M0 baseline. **Deliverable:** a short note on how much task-specific fine-tuning bought you over frozen features.

### Milestone 2 — PEFT: LoRA / QLoRA, and the comparison
Re-run M1 with LoRA, then QLoRA (4-bit base). For each run record: trainable parameters, peak GPU memory, time per epoch, and final metric. **Deliverable:** a table — full FT vs. LoRA vs. QLoRA — with the explicit tradeoff (you'll typically see >99% fewer trainable params and large memory savings for a small or negligible metric drop).

### Milestone 3 — The scaling study (the centerpiece)
Run the *same task and protocol* across `8M → 35M → 150M → 650M`. Plot:
- metric vs. parameter count
- peak memory vs. parameter count
- training time vs. parameter count
Then interpret. You are specifically looking to *experience* the published finding that bigger isn't monotonically better when task data is limited — and to find where on this curve the returns flatten for *your* dataset. **Deliverable:** the three plots + a paragraph on where you'd stop and why.

### Milestone 4 — Bottleneck profiling & mitigation
Take the largest model you can run (likely `650M`) and deliberately hit the wall. Profile the memory breakdown (params / grads / optimizer / activations). Then apply, one at a time, and measure the effect of: bf16, gradient accumulation, gradient checkpointing, FlashAttention, 4-bit quantization. **Deliverable:** a before/after table showing which lever bought what, and a written explanation of *why* each worked in terms of the Phase-2 memory accounting. This is the single most valuable artifact for demonstrating model-engineering intuition.

### Optional Milestone 5 — Stretch toward generative
If energized: instead of a property head, do a small **masked-LM continued-pretraining (domain adaptation)** run on a focused protein family, then re-measure downstream performance. This is your bridge toward the "foundation model training" and generative-bio territory that several of your target roles emphasize.

---

## Deliverables to keep (for interviews + your own record)

1. A clean repo with the pipeline, configs, and W&B run links.
2. The scaling plots (M3) and the bottleneck table (M4) — these two are the strongest evidence of model-engineering intuition.
3. A 1-page README written as if explaining to a hiring manager: the question, the setup, what you found about scaling and bottlenecks, and what you'd do with more compute.

---

## Resource quick-list

- **Code-along:** Karpathy nanoGPT / "Let's build GPT"; The Annotated Transformer.
- **Stack docs:** Hugging Face `transformers`, `datasets`, `peft`, `trl`; bitsandbytes (quantization); Weights & Biases.
- **Papers (short, high-leverage):** LoRA (Hu 2021), QLoRA (Dettmers 2023), Chinchilla (Hoffmann 2022), ZeRO (Rajbhandari 2020).
- **Domain:** ESM-2 model cards on HuggingFace; ProteinGym and FLIP benchmark sites/papers; recent efficient-PLM fine-tuning work (FlashAttention + sequence packing + 4-bit) as a concrete engineering reference.

---

*Sequencing principle: don't start the project until Phases 0–2 are done, but don't over-polish the learning either — the project is where the intuition actually forms. The reading exists to make the project legible, not to be completed for its own sake.*
