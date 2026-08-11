# QDev crew adapter

### 📊 [Read the full report →](https://rohitkumarmanne-442.github.io/qdev-crew-adapter/)

The report is one self-contained page: the charts, the scorer, both pages of
arithmetic, and the notebook, all embedded and downloadable.
[Carousel PDF](slides/qdev-finetune-carousel.pdf) ·
[Calculations](CALCULATIONS.txt) ·
[Notebook](colab/QDev_Crew_Adapter.ipynb)

---

One LoRA adapter that teaches an open 7B the house language of every agent in
QDev Orchestration AI, across all seven SDLC tasks: plan, dev, review, qe, bug,
devops and custom.

**Crew score 0.528 → 0.907.** On the slice made of real production artifacts,
0.617 → 0.954, with 18 of 18 responses parsing.

This repo holds the parts you need to check that claim: the scorer, the
notebook, and the arithmetic. The training corpus is not here, because it is
derived from private repositories.

---

## The numbers

| task | agent | before | after |
|---|---|---|---|
| review | Brain | 0.000 | **1.000** |
| bug | Scooby | 0.250 | **1.000** |
| devops | Perry | 0.777 | **1.000** |
| plan | Phineas | 0.824 | **1.000** |
| custom | user-built | 1.000 | 1.000 |
| qe | Velma | 0.272 | **0.844** |
| dev | Ferb | 0.570 | 0.506 |
| **overall** | | **0.528** | **0.907** |
| **real production slice** | | **0.617** | **0.954** |

`dev` regressed, and the cause is known: its training targets emit the
file-contents array before four cheap required fields, so any truncation costs
all four at once. That is a field-ordering bug in the data, not a model
failure. See "Known issues" below.

Base model: `HuggingFaceH4/zephyr-7b-beta` (Mistral-7B architecture).
Adapter: LoRA r=16, alpha=32, 41,943,040 trainable parameters (0.58%), 80 MB.

---

## Check the scoring yourself

`evaluate/scorer.py` is deterministic and offline. No judge model, no API call,
no randomness. A response earns credit for four things:

| weight | check |
|---|---|
| 0.30 | schema: every required field present, right type, arrays meet their minimum |
| 0.20 | enums: values drawn from the platform's real enum sets |
| 0.35 | rigor: concrete file paths, executable repro steps, a rollback that is a procedure |
| 0.15 | clean: no `TODO`, no `<insert name>`, no lorem ipsum |

```python
from evaluate.scorer import score_one
score_one("review", '{"verdict":"APPROVE","summary":"...","findings":[]}')
```

The same scorer rates the platform's own production output **0.999**, which is
what makes it a ceiling rather than a curve. If it could not rate known-good
work near 1.0, the scorer would be the thing that is wrong.

---

## Reproduce the arithmetic

```bash
python -m calc.report --model mistral-7b   # the full LoRA / QLoRA breakdown
python -m calc.handout                     # 15 worked sections, step by step
```

`CALCULATIONS.txt` is the same content as plain text, laid out for copying by
hand. It covers int8 quantization worked on real values, blockwise
quantization, NF4 vs int4 on the same eight weights, double quantization, the
LoRA parameter count, and the memory ladder.

Predicted from the model dimensions: **0.5787%** trainable.
Reported by the GPU: **0.5758%**. The trainable count matched exactly.

---

## Run the training

`colab/QDev_Crew_Adapter.ipynb` on a Colab **L4**. Not a T4: Turing has no
flash-attention kernel, so attention memory is `heads × seq²` and the sequence
window caps at 4096. At 4096 only 1 of 120 `qe` training examples still fits
its target, which deletes that task from training entirely.

```
python run_finetune.py --stage baseline
python run_finetune.py --stage qlora --rank 16 --alpha 32
python run_finetune.py --stage eval --adapter out/qlora --name qlora
```

The script sizes its own sequence window from measured free VRAM, checkpoints
every 10 steps, and resumes evals after a disconnect. Both long stages are
restartable, which matters because Colab Pro has no background execution.

GPU time, one rented L4:

| | |
|---|---|
| baseline eval | 2.0 h |
| QLoRA training | 5.7 h |
| QLoRA eval | 2.4 h |
| LoRA training | 4.4 h |
| LoRA eval | 2.6 h |
| **total** | **17.1 h** |

Evaluation cost more than training. Three arms on 74 rows at real generation
ceilings is the expensive part, and it is the part usually skipped.

---

## LoRA vs QLoRA

Same rank, same data, same optimiser, same schedule. The only variable was
whether the frozen base is 4-bit NF4 or bf16.

| | QLoRA | LoRA |
|---|---|---|
| resident VRAM | **4.9 GiB** | 13.7 GiB |
| window it could afford | **12,288** | 8,192 |
| overall | **0.907** | 0.846 |

On the three tasks where both arms saw identical training data they tie at
exactly 1.000. The whole gap sits in `qe` and `dev`, the two long-target tasks
where the shorter window cost LoRA training examples.

So quantizing the frozen base cost nothing measurable. What it bought was
context. **This is not a controlled quantization A/B** — the windows differ,
and that difference is the finding rather than a footnote.

---

## Known issues

**`dev` truncates.** 6 of 8 `dev` rows ran to the 12,000-token generation
ceiling. Its targets emit `changes` first, carrying whole file bodies (8,294
tokens at p90), and `commit_message`, `pr_title`, `pr_body` and `summary` all
come after it. Exhaust the budget inside `changes` and four required fields
vanish at once, which scores as a schema failure rather than a truncated
response. The fix is a key reorder in the training targets, then a retrain.

**The scorer saturates.** Five tasks land on exactly 1.000. It grades contract
compliance and rigor thresholds, not writing quality. "Reliably shippable" is
the honest claim, not "perfect".

---

## What is not here

The training corpus, the seed extraction and the teacher-generation pipeline.
They read from private repositories and a production database, so publishing
them would leak private source. What is here is everything needed to audit the
scoring and re-derive the arithmetic.

## Licence

MIT
