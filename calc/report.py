"""
Print (and export) the LoRA / QLoRA arithmetic for a given model + config.

    python -m calc.report --model mistral-7b --rank 16 --alpha 32
    python -m calc.report --model smollm2-360m --rank 8 --alpha 16 --json out.json

Every figure is computed from the model's real dimensions, so whatever config
actually gets trained is the config that gets published.
"""
import argparse
import json
import random
import sys

# Windows consoles default to cp1252 and choke on the box-drawing characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .lora_math import (PRESETS, ALL_LINEAR, ATTN_ONLY, lora_budget,
                        suggest_rank, human, RANK_TABLE)
from .quant_math import (memory_table, int8_symmetric, NF4_LEVELS, nf4_vs_int4,
                         bin_probability_mass, double_quant, training_memory,
                         full_finetune_memory, model_bytes)

LINE = "─" * 78


def h(title: str) -> str:
    return f"\n{LINE}\n  {title}\n{LINE}"


def build(model_key: str, rank: int, alpha: int, targets: list,
          seq_len: int = 1024, batch: int = 1) -> dict:
    spec = PRESETS[model_key]
    bud = lora_budget(spec, rank, alpha, targets)

    qlora = training_memory(spec.params, bud.trainable, "nf4",
                            optimizer="adamw8bit", use_double_quant=True, paged=True)
    lora16 = training_memory(spec.params, bud.trainable, "bfloat16",
                             optimizer="adamw", use_double_quant=False)
    full_gb = full_finetune_memory(spec.params, "bfloat16", "adamw")

    return {"spec": spec, "budget": bud, "qlora": qlora, "lora16": lora16,
            "full_gb": full_gb, "seq_len": seq_len, "batch": batch}


def render(ctx: dict) -> str:
    spec, bud = ctx["spec"], ctx["budget"]
    qlora, lora16, full_gb = ctx["qlora"], ctx["lora16"], ctx["full_gb"]
    out = []

    # ── 1. the model ───────────────────────────────────────────────────
    out.append(h(f"1.  MODEL — {spec.name}"))
    out.append(f"  parameters      {spec.params:,}  ({human(spec.params)})")
    out.append(f"  hidden size     {spec.hidden}          layers  {spec.layers}")
    out.append(f"  attn heads      {spec.heads}  (kv {spec.kv_heads}, head_dim {spec.head_dim})")
    out.append(f"  mlp inner       {spec.intermediate}")

    # ── 2. why quantize ────────────────────────────────────────────────
    out.append(h(f"2.  WHAT {human(spec.params)} PARAMETERS COST, PER PRECISION"))
    out.append(f"  {'precision':<12}{'bits':>6}{'bytes/param':>14}{'memory':>12}{'shrink':>10}")
    for r in memory_table(spec.params):
        out.append(f"  {r['precision']:<12}{r['bits']:>6}{r['bytes_per_param']:>14.2f}"
                   f"{r['gib']:>10.2f}GB{r['shrink']:>9.0f}x")
    out.append(f"\n  A {human(spec.params)} model in float32 does not fit on a 16GB card."
               f"\n  In NF4 it takes {model_bytes(spec.params,'nf4')/1024**3:.2f}GB — that is the whole point of QLoRA.")

    # ── 3. LoRA parameter math ─────────────────────────────────────────
    out.append(h(f"3.  LoRA DECOMPOSITION  (r={bud.r}, alpha={bud.alpha}, scale a/r={bud.scaling:g})"))
    out.append("  dW (d_in x d_out)  ->  A (d_in x r) @ B (r x d_out)")
    out.append(f"  trainable per matrix = r * (d_in + d_out)\n")
    out.append(f"  {'module':<12}{'d_in':>7}{'d_out':>7}{'full dW':>14}{'LoRA A+B':>12}{'shrink':>9}")
    for name, m in bud.per_target.items():
        out.append(f"  {name:<12}{m['d_in']:>7}{m['d_out']:>7}{m['full']:>14,}"
                   f"{m['lora']:>12,}{m['reduction']:>8.0f}x")
    out.append(f"\n  x {spec.layers} layers:")
    out.append(f"    full  dW  for these modules   {bud.full_delta:>16,}  ({human(bud.full_delta)})")
    out.append(f"    LoRA  A+B for these modules   {bud.trainable:>16,}  ({human(bud.trainable)})")
    out.append(f"    reduction on the update       {bud.full_delta/bud.trainable:>16.0f}x")
    out.append(f"\n  Against the FULL model ({human(spec.params)} params):")
    out.append(f"    trainable                     {bud.trainable:>16,}")
    out.append(f"    = {bud.pct_of_base:.4f}% of the model     ({bud.reduction_vs_full_ft:.0f}x fewer trained params)")
    out.append(f"    adapter on disk (bf16)        {bud.trainable*2/1e6:>16.1f} MB")

    # ── 4. rank guidance ───────────────────────────────────────────────
    out.append(h("4.  RANK — WHAT r ACTUALLY BUYS"))
    out.append(f"  {'model size':<16}{'typical r':<18}{'trainable @ each r'}")
    for label, ranks in RANK_TABLE:
        out.append(f"  {label:<16}{str(ranks):<18}")
    out.append(f"\n  For {spec.name} the usual band is r = {suggest_rank(spec.params)}.")
    out.append(f"  {'r':>5}{'trainable':>16}{'% of model':>14}{'adapter MB':>14}")
    for r in sorted(set(suggest_rank(spec.params) + [bud.r])):
        b = lora_budget(spec, r, r * 2, bud.targets)
        mark = "  <- chosen" if r == bud.r else ""
        out.append(f"  {r:>5}{b.trainable:>16,}{b.pct_of_base:>13.4f}%{b.trainable*2/1e6:>13.1f}M{mark}")

    # ── 5. int8 walkthrough ────────────────────────────────────────────
    out.append(h("5.  QUANTIZATION — ONE WEIGHT, END TO END (float32 -> int8)"))
    q = int8_symmetric(-1.0, 1.0)
    out.append(f"  grid            q in [{q.q_min}, {q.q_max}]   ->  {q.q_max-q.q_min} steps")
    out.append(f"  scale           (x_max - x_min) / (q_max - q_min)")
    out.append(f"                  ({q.x_max} - ({q.x_min})) / ({q.q_max} - ({q.q_min}))"
               f" = {q.scale:.8f}")
    out.append(f"  zero_point      {q.zero_point:+.8f}\n")
    out.append(f"  {'x (float32)':>14}{'q (int8)':>12}{'x_hat':>14}{'abs error':>14}")
    for x in (-0.91, -0.78, -0.39, 0.22, 0.28, 0.87):
        r = q.roundtrip(x)
        out.append(f"  {r['x']:>14.4f}{r['q']:>12d}{r['x_hat']:>14.6f}{r['abs_err']:>14.6f}")
    errs = [q.roundtrip(x)["abs_err"] for x in (-0.91, -0.78, -0.39, 0.22, 0.28, 0.87)]
    out.append(f"\n  max round-trip error {max(errs):.6f}  —  4x less memory, ~0.4% distortion.")

    # ── 6. NF4 ─────────────────────────────────────────────────────────
    out.append(h("6.  WHY NF4 BEATS PLAIN int4"))
    out.append(f"  int4 = 16 evenly spaced levels. Weights are not evenly spaced —")
    out.append(f"  they are roughly normal, so most codes land where no weights live.")
    out.append(f"  NF4 places its 16 levels at equal-probability quantiles of a normal:")
    out.append(f"  every bin carries {bin_probability_mass(16):.2f}% of the mass.\n")
    out.append("  NF4 code book:")
    for i in range(0, 16, 8):
        out.append("    " + "  ".join(f"{v:+.4f}" for v in NF4_LEVELS[i:i+8]))
    rng = random.Random(7)
    n_w = 131_072
    sample = [rng.gauss(0, 0.02) for _ in range(n_w)]     # realistic LLM weight scale
    cmp = nf4_vs_int4(sample, block=64)
    out.append(f"\n  {n_w:,} weights ~ N(0, 0.02), quantized in blocks of 64")
    out.append(f"  (int4 baseline = symmetric absmax, scale=absmax/7, includes exact zero):")
    out.append(f"    {'':<10}{'RMSE':>12}{'MAE':>12}{'max err':>12}")
    out.append(f"    {'int4':<10}{cmp['int4']['rmse']:>12.6f}{cmp['int4']['mae']:>12.6f}{cmp['int4']['max_err']:>12.6f}")
    out.append(f"    {'NF4':<10}{cmp['nf4']['rmse']:>12.6f}{cmp['nf4']['mae']:>12.6f}{cmp['nf4']['max_err']:>12.6f}")
    d = cmp["rmse_improvement_pct"]
    verdict = (f"NF4 cuts reconstruction RMSE by {d:.1f}% at identical 4-bit cost."
               if d > 0 else
               f"int4 wins by {-d:.1f}% RMSE here — NF4's advantage is distribution-"
               f"dependent, not automatic.")
    out.append(f"    {verdict}")

    # ── 7. double quantization ─────────────────────────────────────────
    dq = double_quant()
    out.append(h("7.  DOUBLE QUANTIZATION — QUANTIZING THE QUANTIZATION CONSTANTS"))
    out.append(f"  Weights are quantized in blocks of {dq['block']}; each block keeps one")
    out.append(f"  absmax constant. Stored as fp32 that is 32/{dq['block']} = "
               f"{dq['bits_per_param_before']:.4f} bits per weight.")
    out.append(f"  Double quant stores those constants as int8, plus one fp32 scale")
    out.append(f"  per {dq['second_block']} of them:")
    out.append(f"      8/{dq['block']} + 32/({dq['block']}x{dq['second_block']}) = "
               f"{dq['bits_per_param_after']:.4f} bits per weight")
    out.append(f"  saving  {dq['bits_saved_per_param']:.4f} bits/param  ->  "
               f"{dq['bits_saved_per_param']*spec.params/8/1e9:.2f} GB on {human(spec.params)} params")

    # ── 8. the bottom line ─────────────────────────────────────────────
    out.append(h("8.  TRAINING MEMORY — THE NUMBER THAT DECIDES IF IT RUNS"))
    out.append(f"  {'strategy':<26}{'weights':>10}{'consts':>9}{'adapter':>10}"
               f"{'grads':>9}{'optim':>9}{'TOTAL':>11}")
    out.append(f"  {'full fine-tune (bf16)':<26}{'':>10}{'':>9}{'':>10}{'':>9}{'':>9}{full_gb:>9.1f}GB")
    out.append(f"  {'LoRA (bf16 base)':<26}{lora16.base_weights_gb:>10.2f}{lora16.quant_constants_gb:>9.2f}"
               f"{lora16.adapter_gb:>10.3f}{lora16.gradients_gb:>9.3f}{lora16.optimizer_gb:>9.3f}"
               f"{lora16.total_gb:>9.2f}GB")
    out.append(f"  {'QLoRA (NF4+DQ+paged)':<26}{qlora.base_weights_gb:>10.2f}{qlora.quant_constants_gb:>9.2f}"
               f"{qlora.adapter_gb:>10.3f}{qlora.gradients_gb:>9.3f}{qlora.optimizer_gb:>9.3f}"
               f"{qlora.total_gb:>9.2f}GB")
    out.append(f"\n  full fine-tune -> QLoRA  =  {full_gb/qlora.total_gb:.0f}x less memory")
    out.append(f"  (activations excluded — gradient checkpointing keeps them ~1-2GB")
    out.append(f"   at seq_len {ctx['seq_len']}, batch {ctx['batch']}.)")
    fits = [(n, v) for n, v in (("8GB", 8), ("16GB T4", 16), ("24GB", 24)) if qlora.total_gb + 2 < v]
    out.append(f"  fits on: {', '.join(n for n, _ in fits) if fits else 'needs >24GB'}")
    out.append("")
    return "\n".join(out)


def as_json(ctx: dict) -> dict:
    spec, bud = ctx["spec"], ctx["budget"]
    return {
        "model": spec.name,
        "params": spec.params,
        "rank": bud.r, "alpha": bud.alpha, "scaling": bud.scaling,
        "targets": bud.targets,
        "trainable": bud.trainable,
        "pct_of_base": bud.pct_of_base,
        "reduction_vs_full_ft": bud.reduction_vs_full_ft,
        "full_delta_params": bud.full_delta,
        "adapter_mb_bf16": bud.trainable * 2 / 1e6,
        "memory_gb": {
            "full_finetune_bf16": ctx["full_gb"],
            "lora_bf16": ctx["lora16"].total_gb,
            "qlora_nf4": ctx["qlora"].total_gb,
        },
        "precision_table": memory_table(spec.params),
        "double_quant": double_quant(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-7b", choices=list(PRESETS))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--attn-only", action="store_true",
                    help="adapt q/k/v/o only instead of every linear layer")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--json", help="also write the figures to this path")
    a = ap.parse_args()

    ctx = build(a.model, a.rank, a.alpha,
                ATTN_ONLY if a.attn_only else ALL_LINEAR, a.seq_len, a.batch)
    print(render(ctx))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(as_json(ctx), f, indent=2)
        print(f"  wrote {a.json}\n")


if __name__ == "__main__":
    main()
