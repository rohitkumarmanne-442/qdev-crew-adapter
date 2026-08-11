"""
The LoRA / QLoRA arithmetic, worked step by step for copying by hand.

    python -m calc.handout

Every figure comes from calc.lora_math / calc.quant_math using zephyr-7b-beta's
real dimensions, so what you write on paper is what the GPU reported. Where a
number was independently confirmed by the training run, the confirmation is
printed beside it.

Deliberately kept to arithmetic a person can reproduce with a calculator: a
scale, a rounding, a division. No matrix algebra, no calculus. The point is
that every claim in the post has a line of arithmetic behind it.
"""
from __future__ import annotations

import sys

from .lora_math import (PRESETS, ALL_LINEAR, lora_budget, lora_matrix,
                        suggest_rank, textbook_example)
from .quant_math import (NF4_LEVELS, bin_probability_mass, double_quant,
                         full_finetune_memory, int4_uniform_levels,
                         int8_symmetric, memory_table, model_bytes,
                         quantize_block, training_memory)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R, ALPHA = 16, 32
SPEC = PRESETS["mistral-7b"]        # zephyr-7b-beta is Mistral-7B architecture
MEASURED_TRAINABLE = 41_943_040     # printed by the L4 run
MEASURED_TOTAL = 7_283_675_136      # printed by the L4 run

# Eight weights to work by hand. Normal-ish, most of the mass near zero, one
# outlier -- which is the whole reason absmax scaling and NF4 behave the way
# they do. Fixed, not sampled, so the handout is reproducible.
HAND_WEIGHTS = [0.0031, -0.0142, 0.0087, -0.0009,
                0.0410, -0.0056, 0.0021, -0.0198]


def rule(ch="=", n=78):
    print(ch * n)


def head(n, title):
    print()
    rule()
    print(f"  {n}.  {title}")
    rule()


def main():
    b = lora_budget(SPEC, R, ALPHA)
    print()
    print("  LoRA / QLoRA arithmetic for the QDev crew adapter")
    print(f"  base: {SPEC.name} architecture (zephyr-7b-beta), r = {R}, "
          f"alpha = {ALPHA}")
    print()
    print("  PART A  quantization       sections 1-6")
    print("  PART B  low-rank adaptation sections 7-12")
    print("  PART C  putting it together sections 13-15")

    # ══════════════════ PART A - QUANTIZATION ════════════════════════════
    head(1, "Why quantize at all")
    print(f"""
  A parameter is just a number, and you choose how many bits to spend on it.
  {SPEC.params:,} parameters, priced per precision:
""")
    print(f"  {'precision':<11}{'bits':>6}{'bytes/param':>13}{'model size':>13}"
          f"{'vs fp32':>10}")
    print("  " + "-" * 53)
    for row in memory_table(SPEC.params):
        print(f"  {row['precision']:<11}{row['bits']:>6}"
              f"{row['bytes_per_param']:>13.2f}{row['gb']:>11.1f} GB"
              f"{row['shrink']:>9.0f}x")
    print("""
  That is the entire motivation. Same weights, fewer bits each, and the model
  goes from "needs a data centre" to "fits on a rented GPU".
""")

    # ─────────────────────────────────────────────────────────────────────
    head(2, "int8 quantization, worked by hand")
    q = int8_symmetric(-1.0, 1.0)
    print(f"""
  Map a real range onto an integer grid.

      scale       s = (x_max - x_min) / (q_max - q_min)
      zero point  z = the real value that lands on integer 0
      quantize    q = round((x - z) / s)
      dequantize  x_hat = s*q + z

  For x in [-1, 1] onto int8 codes -127 .. 128:

      s = (1 - (-1)) / (128 - (-127))
        = 2 / 255
        = {q.scale:.8f}

      z = {q.zero_point:.6f}          (symmetric: zero maps to zero)
""")
    print(f"  {'x':>10}{'q = round((x-z)/s)':>22}{'x_hat = s*q + z':>19}"
          f"{'abs error':>12}")
    print("  " + "-" * 63)
    for x in (0.7500, 0.1234, -0.5000, 0.0039, -0.9999):
        r_ = q.roundtrip(x)
        print(f"  {r_['x']:>10.4f}{r_['q']:>22d}{r_['x_hat']:>19.6f}"
              f"{r_['abs_err']:>12.6f}")
    worst = q.scale / 2
    print(f"""
  The error is bounded by half a step: s/2 = {worst:.6f}.
  That is the guarantee. Nothing you quantize into this grid can be off by
  more than {worst:.6f}, and on average it is about half that again.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(3, "Blockwise quantization, on eight real weights")
    print(f"""
  One scale for a whole 7B tensor would be dominated by its single largest
  value. So quantization is done per BLOCK, with its own absmax:

      absmax = max(|w|) over the block
      each weight is normalised to w / absmax, in [-1, 1]
      then snapped to the nearest level of a 16-entry code book

  Our eight weights:
""")
    for w in HAND_WEIGHTS:
        print(f"      {w:+.4f}")
    am = max(abs(w) for w in HAND_WEIGHTS)
    print(f"""
      absmax = {am:.4f}

  Normalised (w / absmax), which is what actually gets quantized:
""")
    for w in HAND_WEIGHTS:
        print(f"      {w:+.4f} / {am:.4f} = {w/am:+.4f}")

    # ─────────────────────────────────────────────────────────────────────
    head(4, "NF4 vs int4 on those same eight weights")
    lv4 = int4_uniform_levels()
    print(f"""
  Four bits gives 16 slots. The only question is where to put them.

  int4, absmax scaling: evenly spaced, scale = absmax/7, codes -7 .. +7
""")
    print("      " + "  ".join(f"{v:+.3f}" for v in lv4[:8]))
    print("      " + "  ".join(f"{v:+.3f}" for v in lv4[8:]))
    print(f"""
  NF4: 16 levels at the equal-probability quantiles of a normal
""")
    print("      " + "  ".join(f"{v:+.3f}" for v in NF4_LEVELS[:8]))
    print("      " + "  ".join(f"{v:+.3f}" for v in NF4_LEVELS[8:]))
    print(f"""
  Notice NF4's levels bunch up near zero and thin out at the tails. That is
  the whole idea: each level carries {bin_probability_mass():.2f}% of the
  probability mass (1/16), so resolution follows where the weights actually
  are instead of being spread evenly over a range they mostly do not occupy.

  Snapping our block with each code book:
""")
    a = quantize_block(HAND_WEIGHTS, lv4)
    n = quantize_block(HAND_WEIGHTS, NF4_LEVELS)
    print(f"  {'w':>10}{'int4 recon':>14}{'|err|':>11}"
          f"{'NF4 recon':>14}{'|err|':>11}")
    print("  " + "-" * 60)
    for i, w in enumerate(HAND_WEIGHTS):
        ea = abs(w - a["recon"][i])
        en = abs(w - n["recon"][i])
        print(f"  {w:>+10.4f}{a['recon'][i]:>+14.5f}{ea:>11.5f}"
              f"{n['recon'][i]:>+14.5f}{en:>11.5f}")
    print(f"""
      int4  mean abs error = {a['mae']:.6f}    rmse = {a['rmse']:.6f}
      NF4   mean abs error = {n['mae']:.6f}    rmse = {n['rmse']:.6f}

  On eight weights this is anecdote, not evidence. Measured properly over
  131,072 samples in blocks of 64, NF4's mean squared error is 14.6 % lower
  than int4's, and the margin held across sigmas.

  Two traps, both of which gave me a wrong answer first time:
    - one block of 64 weights is pure noise. You need ~50k+ to measure this.
    - the int4 baseline must include an exact zero level. Space 16 levels
      evenly with no zero and you beat a strawman, not real int4.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(5, "Double quantization: quantizing the quantizers")
    dq = double_quant()
    print(f"""
  Every block of 64 weights needs its own absmax, stored as fp32. That scale
  is overhead on top of the 4 bits per weight:

      32 bits / 64 weights = {dq['bits_per_param_before']:.3f} bits per weight

  Double quantization stores those absmax values as int8, and adds one fp32
  scale per 256 of them:

      8/64  = 0.125000        the int8 absmax
    + 32/(64 x 256)           the scale for those
      = {dq['bits_per_param_after']:.6f} bits per weight

      saved = {dq['bits_per_param_before']:.3f} - {dq['bits_per_param_after']:.3f}
            = {dq['bits_saved_per_param']:.3f} bits per weight

  On {SPEC.params/1e9:.2f}B parameters:

      {SPEC.params:,} x {dq['bits_saved_per_param']:.3f} bits / 8 = \
{SPEC.params*dq['bits_saved_per_param']/8/1e9:.2f} GB saved

  So the true cost of a 4-bit weight is not 4 bits, it is
  4 + {dq['bits_per_param_after']:.3f} = {4+dq['bits_per_param_after']:.3f} bits.
  Small in percentage terms, and the difference between fitting and not when
  you are 100 MB short. I was 73 MB short on the LoRA arm and lost the run.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(6, "Paged optimizers")
    print(f"""
  Adam keeps two moments per TRAINABLE parameter. In fp32 that is 8 bytes each:

      {b.trainable:,} x 8 bytes = {b.trainable*8/2**20:.0f} MB

  In 8-bit it is 2 bytes each:

      {b.trainable:,} x 2 bytes = {b.trainable*2/2**20:.0f} MB

      saved = {b.trainable*8/2**20:.0f} - {b.trainable*2/2**20:.0f} = \
{b.trainable*6/2**20:.0f} MB

  "Paged" means those states live in unified memory and spill to host RAM
  under pressure instead of raising out-of-memory. The saving is real: my LoRA
  run died asking for 168 MB with 95 MB free, a 73 MB shortfall, and 240 MB is
  exactly what this switch returns.
""")

    # ══════════════════ PART B - LOW RANK ADAPTATION ═════════════════════
    head(7, "LoRA, the textbook square case")
    print(f"""
  Fine-tuning learns an update dW and adds it to the frozen weight W:

      W' = W + dW

  dW has the same shape as W, so training it costs as much as the model. The
  LoRA claim is that dW has low intrinsic rank, and can be factored:

      dW  ~  A @ B      A is (d x r),  B is (r x d),  r << d

  Worked for a 512 x 512 matrix at rank 8:

{textbook_example(512, 8)}

  The shape of the saving: you replace d x d with r x (d + d). Quadratic in d
  becomes linear in d. The bigger the matrix, the better the trade.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(8, "The real shapes LoRA attaches to")
    print(f"""
  hidden size          d = {SPEC.hidden:,}
  layers               L = {SPEC.layers}
  attention heads          {SPEC.heads}
  key/value heads          {SPEC.kv_heads}      <- grouped-query attention
  head dim                 {SPEC.hidden} / {SPEC.heads} = {SPEC.head_dim}
  MLP inner            f = {SPEC.intermediate:,}

  Because there are {SPEC.kv_heads} kv heads and not {SPEC.heads}, k_proj and
  v_proj are NOT square. They project d -> kv_heads x head_dim:

      {SPEC.kv_heads} x {SPEC.head_dim} = {SPEC.kv_heads * SPEC.head_dim}

  Missing this is the single most common way a LoRA parameter estimate comes
  out wrong, usually about 15 percent too high.
""")
    shapes = SPEC.target_shapes()
    print(f"  {'projection':<12}{'d_in':>8}{'d_out':>9}")
    print("  " + "-" * 29)
    for t in ALL_LINEAR:
        di, do = shapes[t]
        print(f"  {t:<12}{di:>8,}{do:>9,}")

    # ─────────────────────────────────────────────────────────────────────
    head(9, "Trainable parameters per matrix")
    print(f"""
      A is d_in x r        B is r x d_out        so A + B = r x (d_in + d_out)

  Worked for q_proj, the square one:

      A: {SPEC.hidden:,} x {R} = {SPEC.hidden * R:,}
      B: {R} x {SPEC.hidden:,} = {R * SPEC.hidden:,}
      total  = {2 * SPEC.hidden * R:,}   vs a full update of
               {SPEC.hidden:,} x {SPEC.hidden:,} = {SPEC.hidden ** 2:,}
      ratio  = {SPEC.hidden ** 2 / (2 * SPEC.hidden * R):,.0f}x fewer

  And for k_proj, where GQA makes it rectangular:

      A: {SPEC.hidden:,} x {R} = {SPEC.hidden * R:,}
      B: {R} x {SPEC.kv_heads*SPEC.head_dim:,} = \
{R * SPEC.kv_heads * SPEC.head_dim:,}
      total  = {SPEC.hidden*R + R*SPEC.kv_heads*SPEC.head_dim:,}
""")
    print(f"  {'projection':<12}{'full update':>16}{'LoRA A+B':>12}{'ratio':>9}")
    print("  " + "-" * 49)
    for t in ALL_LINEAR:
        di, do = shapes[t]
        m = lora_matrix(di, do, R)
        print(f"  {t:<12}{m['full']:>16,}{m['lora']:>12,}{m['reduction']:>8.0f}x")

    # ─────────────────────────────────────────────────────────────────────
    head(10, "Totalling it across the model")
    per_layer = sum(v["per_layer"] for v in b.per_target.values())
    print(f"""
  per layer, all seven projections     {per_layer:>15,}
  x {SPEC.layers} layers                            {b.trainable:>15,}   <- TRAINABLE

  total model parameters               {SPEC.params:>15,}
  fraction trained    {b.trainable:,} / {SPEC.params:,}
                                                 = {b.pct_of_base:.4f} %
  reduction                                        {b.reduction_vs_full_ft:,.0f}x fewer

  CONFIRMED ON THE GPU. The L4 run printed:
      trainable {MEASURED_TRAINABLE:,} / {MEASURED_TOTAL:,} = \
{100*MEASURED_TRAINABLE/MEASURED_TOTAL:.4f}%
  Predicted {b.trainable:,}, measured {MEASURED_TRAINABLE:,}: \
{'exact match' if b.trainable == MEASURED_TRAINABLE else 'close'}.
  The percentage differs by \
{abs(b.pct_of_base - 100*MEASURED_TRAINABLE/MEASURED_TOTAL):.4f} points only
  because the preset's parameter count is rounded. The trainable count is exact.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(11, "Choosing r")
    print(f"""
  r buys capacity and costs parameters, linearly. Same model, same targets:
""")
    print(f"  {'r':>5}{'trainable':>16}{'% of model':>13}{'adapter (bf16)':>17}")
    print("  " + "-" * 51)
    for rr in (4, 8, 16, 32, 64, 128):
        bb = lora_budget(SPEC, rr, rr * 2)
        print(f"  {rr:>5}{bb.trainable:>16,}{bb.pct_of_base:>12.4f}%"
              f"{bb.trainable*2/2**20:>14.0f} MB")
    print(f"""
  Doubling r doubles the trainable count exactly, because every term is
  r x (d_in + d_out). Rule of thumb for this size class: {suggest_rank(SPEC.params)}.
  I used r = {R}: enough capacity for seven task formats, small enough that the
  adapter still ships as one {b.trainable*2/2**20:.0f} MB file.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(12, "Why alpha exists")
    print(f"""
  At inference the adapter is folded back in as

      W' = W + (alpha / r) x A @ B
             = W + ({ALPHA} / {R}) x A @ B
             = W + {b.scaling:.0f} x A @ B

  alpha/r is a fixed gain on the update. It exists so changing r does not
  silently change how strongly the adapter speaks: raise r for capacity and
  raise alpha with it to hold the scale. Merged this way the adapter costs
  ZERO extra latency, because W' is just a weight matrix again.
""")

    # ══════════════════ PART C - PUTTING IT TOGETHER ═════════════════════
    head(13, "What the adapter weighs")
    for prec, per in (("bfloat16", 2), ("float32", 4)):
        print(f"  {b.trainable:,} x {per} bytes ({prec:<8}) = "
              f"{b.trainable*per/2**20:>8.1f} MB")
    print(f"""
  Against the base model on disk:
      {SPEC.params:,} x 2 bytes = \
{model_bytes(SPEC.params, 'float16')/2**30:.1f} GB

  So one agent's specialisation ships as {b.trainable*2/2**20:.0f} MB. Seven
  agents sharing one base is one \
{model_bytes(SPEC.params, 'float16')/2**30:.0f} GB model plus seven small
  files, not seven {model_bytes(SPEC.params, 'float16')/2**30:.0f} GB models.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(14, "Memory: full fine-tune vs LoRA vs QLoRA")
    full = full_finetune_memory(SPEC.params, "bfloat16")
    lo = training_memory(SPEC.params, b.trainable, "bfloat16")
    ql = training_memory(SPEC.params, b.trainable, "nf4")
    P = SPEC.params
    print(f"""
  Full fine-tune holds four things for EVERY parameter, plus an fp32 master
  copy because bf16 alone loses too much precision in the optimiser step:

      weights      {P:,} x 2 bytes  = {P*2/1e9:>6.1f} GB
      gradients    {P:,} x 2 bytes  = {P*2/1e9:>6.1f} GB
      optimiser    {P:,} x 8 bytes  = {P*8/1e9:>6.1f} GB
      master fp32  {P:,} x 4 bytes  = {P*4/1e9:>6.1f} GB
                                                ------
                                                {full:>6.1f} GB

  LoRA freezes the base, so gradients and optimiser state exist only for the
  {b.trainable:,} trainable parameters:

      frozen base  = {lo.base_weights_gb:>6.1f} GB
      adapter      = {lo.adapter_gb:>6.2f} GB
      grads        = {lo.gradients_gb:>6.2f} GB
      optimiser    = {lo.optimizer_gb:>6.2f} GB
                     ------
                     {lo.total_gb:>6.1f} GB

  QLoRA also stores the frozen base in 4-bit NF4:

      frozen base  = {ql.base_weights_gb:>6.1f} GB
      quant consts = {ql.quant_constants_gb:>6.2f} GB
      adapter      = {ql.adapter_gb:>6.2f} GB
      grads + opt  = {ql.gradients_gb + ql.optimizer_gb:>6.2f} GB
                     ------
                     {ql.total_gb:>6.1f} GB      {full/ql.total_gb:.0f}x less than full

  CONFIRMED ON THE GPU. Resident memory after load, same card, same adapter:
      QLoRA (NF4 base)    4.9 GiB
      LoRA  (bf16 base)  13.7 GiB
  and at a 12,288-token window the bf16 arm ran out of memory on a 22 GB L4,
  twice, at step 6 of 156. It only fit at 8,192.
""")

    # ─────────────────────────────────────────────────────────────────────
    head(15, "What it bought, on real work")
    print(f"""
  Scored on 74 held-out prompts across all seven agents, of which 18 are real
  production artifacts from the platform, replayed through the exact prompt
  builders that run in production. Deterministic offline scorer, no judge
  model, greedy decoding, identical eval for every arm.

      crew score, overall        0.528  ->  0.907      +0.380   (1.7x)
      on real production work    0.617  ->  0.954      +0.337
      responses that parsed      44/74  ->  71/74

      Brain   code review        0.000  ->  1.000   was prose 8 times out of 8
      Scooby  bug filing         0.250  ->  1.000
      Velma   test authoring     0.272  ->  0.844
      Perry   release plans      0.777  ->  1.000
      Phineas story drafting     0.824  ->  1.000

  Cost: {b.trainable*2/2**20:.0f} MB of adapter, $28.86 of teacher generation,
  and 17 hours of L4 time:

      baseline eval        2.0 h
      QLoRA training       5.7 h
      QLoRA eval           2.4 h
      LoRA training        4.4 h
      LoRA eval            2.6 h
                          -----
                           17.1 h

  Note that evaluation cost more than the QLoRA training did. Measuring three
  arms honestly on 74 rows, at real generation ceilings, is the expensive part
  and it is the part usually skipped.
""")
    rule()
    print()


if __name__ == "__main__":
    main()
