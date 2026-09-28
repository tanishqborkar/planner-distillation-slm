# planner-distillation-slm

Code, data and per-problem results for **"Collapsing a Guided Reasoning
Pipeline into a Single Small Language Model"**.

Two findings. Solution guidance helps general-purpose executors at every scale
we tested and does nothing measurable for a mathematically specialised executor
of the same family and size. And where it does help, the planner model can be
distilled away entirely: one 1.5B model matches the two-model pipeline exactly,
in half the time.

---

## Results

**Does a plan help?** Problems solved out of 1319 on the GSM8K test split, fp16,
one shared answer parser. Wins and losses count items resolved by exactly one of
the two conditions; *p* from McNemar's exact test.

| Executor | specialised | no plan | with plan | net | wins:losses | *p* |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | no | 298 | 421 | +123 | 263:140 | 9e-10 |
| Qwen2.5-1.5B | no | 600 | 738 | +138 | 310:172 | 3e-10 |
| Qwen2.5-3B | no | 706 | 910 | +204 | 327:123 | 2e-22 |
| Qwen2.5-Math-1.5B | **yes** | 916 | 945 | +29 | 196:167 | 0.14 (n.s.) |

The benefit does not track capability — the 3B model is the stronger of the two
mid-sized general models and gains more, not less. It tracks specialisation.

**What does a correct answer cost?** Qwen2.5-1.5B base throughout. The last
column divides elapsed time by the fraction solved, pricing a useful answer
rather than an attempt.

| Configuration | solved | tokens | s/item | s/correct |
|---|---|---|---|---|
| one model, equations only | 698 | 41 | 4.1 | **7.8** |
| one model, planner-taught | 738 | 116 | 11.8 | **21.0** |
| one model, teacher-taught | 718 | 118 | 12.1 | 22.3 |
| executor alone, no plan | 600 | 130 | 15.6 | 34.2 |
| planner + executor | 738 | 230 | 23.2 | 41.5 |

The distilled single model ties the pipeline's count exactly at half the
latency, a third of the resident parameters and half the output. No axis we
measured favours the pipeline.

---

## Layout

```
code_files/
  generate_sg.py                        teacher prompt + guidance generation
  trainings.ipynb                       planner and student fine-tuning
  2slm-pipeline.ipynb                   two-model evaluation
  1slm-guidance-internalisation.ipynb   single-model distillation
  final-benchmarking.ipynb              frontier and latency measurement

data/
  sg_dataset_improv.json                7,473 operator-tagged guidance samples
  oneslm_data_plannerdistill.jsonl      distillation targets (plan + equations)

results/
  gsm8k/    per-problem predictions on the full test split, n=1319
  ood/      SVAMP (n=300) and MultiArith (n=180)
  plans/    raw planner output, for auditing plan quality independently
  summary/  the JSON files the paper's tables are computed from
```

Every prediction CSV carries one row per problem: the raw generation, the
parsed answer, which parse strategy fired, and whether it was correct.

---

## Re-scoring with your own parser

This is the part worth your attention. Our first parser took the last number in
each output and scored **16 points low**, because these models routinely state
the correct answer and then talk themselves out of it. The raw generations are
included so you can check that rather than take our word for it:

```python
import pandas as pd
d = pd.read_csv("results/gsm8k/grad_qwen2.5-3B.csv")
print(d.guided_correct.mean(), d.baseline_correct.mean())
print(d.guided_strategy.value_counts())   # which extraction rule fired
print(d.guided_raw.iloc[0])               # the model's actual output
```

The shipped parser tries, in order: a `####` marker, a `\boxed{}` wrapper, then
the last equation to resolve before any self-correction phrase.

---

## Method in brief

1. **Guidance data.** Gemini 2.5 Flash is shown each GSM8K training problem
   *together with its reference solution as hidden context*, and asked to strip
   out the computation while preserving the decomposition. Every step is tagged
   with one of eight operators (ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE, COUNT,
   CONVERT, RATIO), so plan validity is decidable at inference with no gold
   answer in hand.
2. **Planner.** Qwen3-4B, 4-bit QLoRA, rank 16, loss masked to the plan.
3. **Pipeline.** The executor's weights are never touched; removing the plan
   block from the prompt gives the comparison condition.
4. **Distillation.** The planner runs once, offline, across GSM8K train. The
   student is then supervised on `question → plan + equations + #### answer`,
   so at inference it writes its own plan and works through it in one pass. No
   4B model is loaded.

---

## Four things that cost us real time

- **Prompts must end `Plan:\n` and `Solution:\n`.** Qwen's tokeniser merges `:`
  with a following newline into a single token, so ending at the colon is a
  character-level prefix of the training string but not a token-level one. Our
  first planner scored 0% on a format check while its loss curve looked
  perfectly healthy.
- **fp16 throughout.** An earlier 4-bit pass scored 5–7 points lower than the
  same configurations in fp16 — wider than several of the effects being
  measured.
- **Latency is sequential at batch 1**, warmed up, with CUDA synchronisation.
  Batched throughput answers a different question; mixing the two once inflated
  a speed-up of ours roughly fivefold.
- **Mask the loss by tokenising prompt and target separately.** A
  string-matching masker silently dropped 7,098 of 7,100 training rows without
  raising anything.

---


