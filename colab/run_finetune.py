"""
QDev crew adapter — baseline eval, LoRA, QLoRA, after-eval.

Runs on a Colab T4 (16 GB). Everything is one script so the before/after
comparison uses the SAME model, the SAME eval set and the SAME scorer; a
baseline measured any other way is not a baseline.

    python run_finetune.py --stage qlora  --smoke    # 4 steps, proves wiring
    python run_finetune.py --stage baseline          # the BEFORE number
    python run_finetune.py --stage qlora             # train
    python run_finetune.py --stage eval --adapter out/qlora --name qlora
    python run_finetune.py --stage lora
    python run_finetune.py --stage eval --adapter out/lora  --name lora

Results append to out/results.json, keyed by run name.

Memory on a T4 is governed by the vocabulary, not the parameter count — see
predict_peak_gib(). The script sizes its own sequence window by predicting peak
VRAM from the model config before anything is downloaded, and picks the largest
window that fits. Overrides, if ever needed:

    --max-length 3072            force a window instead of auto-sizing
    --no-liger                   if training crashes inside Triton
    --base <model id>            pass it to EVERY stage including baseline, or
                                 the before/after compares two different models
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

# The OOM traceback reported 1.17 GB "reserved but unallocated" — pure
# fragmentation from allocating and freeing logits tensors of varying sequence
# length. expandable_segments lets the allocator grow a segment instead of
# hunting for a contiguous block, and it must be set before torch initialises
# CUDA, so it goes above the import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch                                                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate.scorer import score_one, aggregate            # noqa: E402

# A 32k-vocab 7B, chosen because of the vocabulary rather than despite it.
#
# Qwen2.5-7B is the better base model in the abstract, but it carries a 152,064
# token vocabulary, and on a 14.56 GB T4 the logits tensor (vocab x seq, held
# roughly three times over during a backward pass) is what decides whether
# training runs at all:
#
#     Qwen2.5-7B   152,064 vocab -> 5.0 GB of logits at seq 4096   OOM
#     zephyr-7b    32,000  vocab -> 1.6 GB of logits at seq 6144   fits
#
# Same parameter count, 4.75x less logits memory, and the longer window keeps
# 97% of the corpus instead of 82% — the dropped rows are concentrated in qe and
# dev, so the short window costs exactly the two hardest tasks. Zephyr is an
# ungated Mistral-7B derivative, so no HF token is needed, and its dimensions
# are the ones calc/report.py already models.
BASE = os.environ.get("QDEV_BASE_MODEL", "HuggingFaceH4/zephyr-7b-beta")
OUT = pathlib.Path("out")
OUT.mkdir(exist_ok=True)

# Generation ceilings per task: 1.15x the LONGEST target in the corpus, measured
# with this tokenizer — not estimated.
#
# The previous values were guesses and they were badly wrong: they would have cut
# off 94% of devops answers, 86% of custom and 40% of dev. A truncated answer
# scores as broken JSON, and the damage is asymmetric — a tuned model emitting a
# complete 2,000-token object gets truncated and scored 0, while a base model
# rambling for 300 tokens fits under the cap. That does not just add noise, it
# systematically understates the improvement the whole exercise is measuring.
GEN_TOKENS = {"plan": 1700, "qe": 6700, "dev": 12000, "review": 2400,
              "bug": 1300, "devops": 2900, "custom": 2700}

# The binding constraint on a T4 is not the 7B of weights — it is the vocabulary.
# Qwen2.5 has 152,064 tokens, so the logits tensor is vocab x seq, and training
# holds roughly three at once (fp16 forward, fp32 upcast for cross-entropy, the
# gradient):
#
#     seq 6144 -> 152064 x 6144 x (2+4+2) B  ~= 7.5 GB   OOM on a 14.56 GB T4
#     seq 4096 -> 152064 x 4096 x (2+4+2) B  ~= 5.0 GB   fits, but see below
#
# Dropping to 4096 is not free: 147 of 827 training examples have a *target*
# longer than the budget, concentrated in qe and dev, so shrinking the window
# silently deletes the two hardest tasks from the corpus.
#
# Fused linear cross-entropy (Liger) is the actual fix. It computes the loss in
# chunks and never materialises full logits, so peak memory stops scaling with
# vocabulary and 6144 fits with room to spare. When it is unavailable we fall
# back to 4096 and eat the drops, because an OOM is worse than a smaller corpus.
# Tried largest-first; the first one that fits measured free VRAM is used. Every
# step down drops training examples whose target no longer has room, and the
# losses are not spread evenly — measured against this corpus:
#
#     seq    overall     dev      qe
#   12288       100%    100%    100%
#    8192        98%     88%    100%
#    6144        96%     71%     98%
#    4096        79%     50%      1%     <- deletes the QE task outright
#
# 12288 is the top of the ladder because it is the first window that costs
# nothing at all, and on a 24 GB card with flash attention it is affordable.
WINDOWS = [12288, 10240, 8192, 6144, 4096, 3072, 2048]
MAX_LEN_PLAIN = 8192          # fallback when there is no GPU to measure


def dtype_for_gpu():
    """T4 is Turing (sm_75) and has no bfloat16. Ampere+ does."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def predict_peak_gib(four_bit: bool, seq: int) -> tuple[float, dict, int]:
    """Estimate peak training VRAM from the model config, before downloading it.

    Calibrated against a real failure rather than derived from first principles:
    Qwen2.5-7B at seq 4096 on a 14.56 GiB T4 reported 12.94 GiB in use while
    asking for another 1.75 GiB, so true peak was ~14.7 GiB. Back-solving that
    against known weight and logits terms puts activations near 0.65 GiB per
    1024 tokens under gradient checkpointing, which is the constant used here.

    Three terms, in the order they matter on a small card:
      weights  quantized body at 0.5 B/param, but embeddings and lm_head stay
               in half precision at 2 B/param — for a 152k vocab that is 2 GB
               of the total and the reason a naive nparam*0.5 underestimates.
      logits   vocab x seq, held ~3 times over (fp16, fp32 upcast, gradient).
      acts     checkpointed activations, linear in sequence length.
    """
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(BASE)
    vocab = c.vocab_size
    hidden = c.hidden_size
    layers = c.num_hidden_layers
    inter = getattr(c, "intermediate_size", 4 * hidden)
    kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    head = hidden // c.num_attention_heads

    per_layer = (hidden * hidden * 2                      # q_proj, o_proj
                 + hidden * kv * head * 2                 # k_proj, v_proj (GQA)
                 + hidden * inter * 3)                    # gate, up, down
    body = per_layer * layers
    embed = vocab * hidden * (1 if getattr(c, "tie_word_embeddings", False) else 2)

    weights = (body * (0.5 if four_bit else 2) + embed * 2) / 2**30
    logits = vocab * seq * (2 + 4 + 2) / 2**30
    # Attention is QUADRATIC in sequence length, not linear. Modelling it
    # linearly is what predicted 9.1 GiB for a run that actually needed >14.6.
    # The memory-efficient SDPA kernel never builds the score matrix; the math
    # fallback builds it twice (scores, then softmax). We pin the efficient
    # kernel above, so assume it — but the math figure is what to reach for
    # when a run OOMs inside scaled_dot_product_attention anyway.
    # Which attention kernel we get is a property of the GPU, not a choice.
    # Flash needs sm_80+ (Ampere). On sm_75 (Turing T4) flash does not exist and
    # the memory-efficient kernel rejects this shape — verified the hard way:
    # disabling math there produced "RuntimeError: Invalid backend". So Turing
    # gets the math kernel, which builds heads x seq x seq twice.
    sm = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 8
    if sm >= 8:
        attn = c.num_attention_heads * seq * head * 4 / 2**30
    else:
        attn = c.num_attention_heads * seq * seq * 2 * 2 / 2**30
    acts = layers * seq * hidden * 2 / 2**30      # checkpointed layer inputs
    # Measured residual, not a fudge factor. At seq 6144 the terms above sum to
    # 12.4 GiB but the run died holding 14.24 GiB: LoRA fp32 gradients, paged
    # optimizer staging, autocast copies, the CUDA context and allocator
    # fragmentation together account for roughly 2 GiB that no per-tensor
    # formula captures.
    overhead = 2.0
    total = weights + logits + attn + acts + overhead
    return (total,
            {"weights": weights, "logits": logits, "attn": attn,
             "acts": acts, "overhead": overhead},
            vocab)


def liger_available() -> bool:
    """Whether fused linear cross-entropy can actually be used here.

    Import success is not enough — Liger is Triton-based and parts of it assume
    Ampere, so it can install cleanly on a Turing T4 and then fail at the first
    backward pass. Probe the kernel for real; a false positive here costs a
    crash 40 minutes into training.

    Every failure path prints why. An earlier version returned a bare False on
    import error, which made "not installed" and "class moved between releases"
    look identical from the log — and the second one is a bug in this function,
    not a fact about the GPU.
    """
    if not torch.cuda.is_available():
        print("  liger: no CUDA device")
        return False
    try:
        import liger_kernel
    except Exception as exc:
        print(f"  liger: not importable ({type(exc).__name__}: {exc}). "
              f"pip install liger-kernel")
        return False

    # The class has lived at both of these paths across releases; try the
    # package root last since it re-exports.
    loss_cls = None
    tried = []
    for mod, attr in (
        ("liger_kernel.transformers.fused_linear_cross_entropy",
         "LigerFusedLinearCrossEntropyLoss"),
        ("liger_kernel.transformers", "LigerFusedLinearCrossEntropyLoss"),
    ):
        try:
            loss_cls = getattr(__import__(mod, fromlist=[attr]), attr)
            break
        except Exception as exc:
            tried.append(f"{mod}: {type(exc).__name__}")
    if loss_cls is None:
        ver = getattr(liger_kernel, "__version__", "?")
        print(f"  liger {ver} installed but the fused-CE class was not found "
              f"({'; '.join(tried)})")
        return False

    try:
        h, v = 64, 512
        x = torch.randn(8, h, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        w = torch.randn(v, h, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        t = torch.randint(0, v, (8,), device="cuda")
        loss_cls()(w, x, t).backward()
        return True
    except Exception as exc:
        print(f"  liger present but the kernel will not run on this GPU: "
              f"{type(exc).__name__}: {str(exc)[:160]}")
        return False


def load_split(name: str) -> list[dict]:
    p = pathlib.Path("dataset") / f"{name}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def build_eval_set(per_task: int, include_real: bool) -> list[dict]:
    """Every real production row, plus a per-task sample of generated rows.

    The real slice is the headline; the generated slice gives coverage for the
    five tasks that have no production data.
    """
    rows: list[dict] = []
    if include_real:
        for r in load_split("test_real"):
            r = dict(r); r["slice"] = "real"
            rows.append(r)
    seen: dict[str, int] = {}
    for r in load_split("test_gen"):
        t = r["task"]
        if seen.get(t, 0) >= per_task:
            continue
        seen[t] = seen.get(t, 0) + 1
        r = dict(r); r["slice"] = "gen"
        rows.append(r)
    return rows


# ─────────────────────────── model loading ───────────────────────────────
def load_model(adapter: str | None, four_bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dt = dtype_for_gpu()
    kw = dict(dtype=dt, device_map="auto", attn_implementation="sdpa")
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # quantile code book, not uniform int4
            bnb_4bit_use_double_quant=True,      # quantize the quantization constants
            bnb_4bit_compute_dtype=dt,
        )
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                    # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    if adapter:
        _defuse_torchao_check()
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload() if not four_bit else model
    model.eval()
    return model, tok


def _defuse_torchao_check() -> None:
    """Stop PEFT's torchao probe from raising on Colab's preinstalled 0.10.0.

    Attaching an adapter walks PEFT's dispatcher list. `dispatch_torchao` calls
    `is_torchao_available()`, which — despite the name — RAISES ImportError when
    it finds a torchao older than 0.16 rather than returning False. Colab ships
    0.10.0, so the probe aborts adapter loading before any dispatcher that could
    have handled the layer gets a turn.

    It only bites in bf16: with 4-bit loading, the bitsandbytes dispatcher
    matches first and PEFT never reaches the torchao one. Nothing here uses
    torchao — quantization is bitsandbytes throughout — so report it absent
    instead of upgrading it and risking a torch version change mid-run.
    """
    try:
        import torchao
    except Exception:
        return                                   # absent: PEFT handles that fine
    # Hand-rolled compare rather than packaging.version — an exception here used
    # to be swallowed by the outer guard, which silently skipped the whole fix
    # on any environment without `packaging` installed.
    raw = str(getattr(torchao, "__version__", "0")).split("+")[0]
    try:
        ver = tuple(int(x) for x in raw.split(".")[:3])
    except ValueError:
        ver = (0,)
    if ver >= (0, 16, 0):
        return
    # Import peft first so its submodules — including the torchao dispatcher —
    # are in sys.modules, then rebind the name everywhere it is bound. Patching
    # only peft.import_utils is not enough: the dispatcher does a from-import,
    # so it holds its own reference that a source-module rebind never reaches.
    try:
        import peft  # noqa: F401
        import peft.tuners.lora  # noqa: F401
    except Exception:
        return
    patched = 0
    for name, m in list(sys.modules.items()):
        if not name.startswith("peft"):
            continue
        if getattr(m, "is_torchao_available", None) is not None:
            m.is_torchao_available = lambda: False
            patched += 1
    if patched:
        print(f"  torchao {torchao.__version__} is older than PEFT requires and "
              f"its availability check raises instead of returning False; "
              f"reported absent in {patched} module(s) — unused here.")


# ─────────────────────────────── eval ────────────────────────────────────
def kv_batch_limit(model, task: str, prompt_tokens: int, want: int) -> int:
    """Largest batch whose KV cache still fits, for this task's token budget.

    A fixed batch size is wrong here because the generation ceiling varies 9x
    across tasks. The KV cache is 2 (k+v) x kv_heads x head_dim x layers x
    dtype_bytes per token per sequence — 128 KiB/token for this model — and it
    grows with (prompt + max_new_tokens), so `dev` at 12,000 new tokens costs
    6.5x what `plan` does at 1,700.

    Concretely, on a 22 GiB L4 in bf16: weights are 13.5 GiB, and dev at batch 4
    needs 7.3 GiB of cache. 21.8 of 22.0 — it OOMs, roughly 40 minutes into an
    eval, after the cheap tasks have already passed.
    """
    if not torch.cuda.is_available():
        return want
    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    per_tok = 2 * kv_heads * head_dim * cfg.num_hidden_layers * 2   # bf16/fp16
    span = prompt_tokens + GEN_TOKENS.get(task, 1500)
    free = torch.cuda.mem_get_info()[0]
    usable = free * 0.75 - 1.0 * 2**30        # margin for logits + fragmentation
    if usable <= 0:
        return 1
    return max(1, min(want, int(usable // (span * per_tok))))


@torch.no_grad()
def run_eval(model, tok, rows: list[dict], batch: int, run_name: str) -> dict:
    scored, raws = [], []
    # Group by task so a whole batch shares one generation ceiling.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["task"])

    # Checkpoint after every batch so a Colab timeout costs minutes, not the
    # whole eval. Greedy decoding makes this sound: re-running a row would
    # reproduce it exactly, so a resumed run and an uninterrupted one produce
    # identical numbers. Keyed by (task, index) because the row order is fixed
    # by the sorted `order` above.
    part_path = OUT / f"partial_{run_name}.json"
    resume: dict[str, dict] = {}
    if part_path.exists():
        try:
            saved = json.loads(part_path.read_text(encoding="utf-8"))
            if saved.get("n_rows") == len(rows):
                resume = saved.get("rows", {})
                print(f"  resuming: {len(resume)}/{len(rows)} rows already "
                      f"scored in {part_path.name}")
            else:
                print(f"  ignoring {part_path.name}: it holds "
                      f"{saved.get('n_rows')} rows, this eval has {len(rows)}")
        except Exception as exc:
            print(f"  could not read {part_path.name} ({type(exc).__name__})")

    def flush():
        part_path.write_text(json.dumps(
            {"n_rows": len(rows), "run": run_name,
             "rows": {k: v for k, v in resume.items()}},
            ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    done = 0
    announced: set[str] = set()
    learned: dict[str, int] = {}
    s = 0
    while s < len(order):
        task = rows[order[s]]["task"]
        same = [i for i in order[s:] if rows[i]["task"] == task]
        prompts_all = [
            tok.apply_chat_template(rows[i]["messages"][:2],
                                    tokenize=False, add_generation_prompt=True)
            for i in same
        ]
        longest = max(len(tok(p)["input_ids"]) for p in prompts_all)
        # `learned` carries an OOM lesson across batches of the same task.
        # Without it the retry re-discovers the same limit on every group and
        # pays a failed full-width generation each time.
        b = min(kv_batch_limit(model, task, longest, batch),
                learned.get(task, batch))
        if task not in announced:
            announced.add(task)
            if b < batch:
                print(f"    [{task}] batch {batch} -> {b}: "
                      f"{GEN_TOKENS.get(task)} new tokens on a {longest}-token "
                      f"prompt would not fit the KV cache")
        # Retry narrower on OOM rather than losing the run. Every memory
        # estimate in this script has been wrong at least once, and an eval
        # that dies at row 60 of 74 costs an hour and produces nothing.
        #
        # `s` advances only once the batch width is settled — halving the batch
        # after advancing would silently skip the rows that were dropped.
        # Rows already scored in a previous attempt: replay from the checkpoint
        # instead of regenerating. Skipped before any model work happens.
        while same and str(same[0]) in resume:
            i = same[0]
            rec = resume[str(i)]
            sc = dict(rec["score"]); sc["slice"] = rows[i]["slice"]
            scored.append(sc)
            raws.append({"task": rows[i]["task"], "slice": rows[i]["slice"],
                         "raw": rec["raw"],
                         "new_tokens": rec.get("new_tokens"),
                         "hit_cap": rec.get("hit_cap"),
                         "cap": rec.get("cap")})
            same.pop(0)
            prompts_all.pop(0)
            s += 1
            done += 1
        if not same:
            continue

        texts, idx = None, []
        while texts is None:
            idx = same[:b]
            prompts = prompts_all[:b]
            try:
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=12288).to(model.device)
                out = model.generate(
                    **enc,
                    max_new_tokens=GEN_TOKENS.get(task, 1500),
                    do_sample=False,             # greedy: before/after must match
                    pad_token_id=tok.pad_token_id,
                )
                gen = [out[j][enc["input_ids"].shape[1]:]
                       for j in range(len(prompts))]
                texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
                # How many tokens it actually spent, and whether it stopped
                # because it was finished or because it ran out. Without this,
                # "truncated at the ceiling" and "genuinely wrong" look
                # identical in the log.
                ntok = [int((g != tok.pad_token_id).sum()) for g in gen]
            except torch.cuda.OutOfMemoryError:
                enc = out = None
                torch.cuda.empty_cache()
                if b == 1:
                    print(f"    [{task}] OOM even at batch 1 — recording an "
                          f"empty response for this row rather than aborting")
                    texts, ntok = [""], [0]
                else:
                    b = max(1, b // 2)
                    learned[task] = b
                    print(f"    [{task}] OOM, retrying at batch {b} "
                          f"(and holding there for the rest of {task})")
        s += len(idx)

        cap = GEN_TOKENS.get(task, 1500)
        for j, i in enumerate(idx):
            sc = score_one(rows[i]["task"], texts[j])
            # 4000 chars was too tight to diagnose anything: a dev response runs
            # to tens of thousands, so every sample looked identically
            # "truncated" because the LOG truncated it, not the model.
            rec = {"score": sc, "raw": texts[j][:24000],
                   "new_tokens": ntok[j], "cap": cap,
                   "hit_cap": ntok[j] >= cap - 1,
                   "chars": len(texts[j])}
            resume[str(i)] = rec
            sc = dict(sc); sc["slice"] = rows[i]["slice"]
            scored.append(sc)
            raws.append({"task": rows[i]["task"], "slice": rows[i]["slice"],
                         "raw": rec["raw"], "new_tokens": ntok[j],
                         "hit_cap": rec["hit_cap"], "cap": cap})
        done += len(idx)
        flush()
        # Free the cache between task groups: the batch size changes with the
        # generation ceiling, so leftover blocks from a wide cheap task
        # fragment the arena a narrow expensive one then needs contiguously.
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"    {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    agg = aggregate(scored)
    agg["by_slice"] = {}
    for sl in ("real", "gen"):
        sub = [s for s in scored if s["slice"] == sl]
        if sub:
            agg["by_slice"][sl] = aggregate(sub)["overall"]
    agg["run"] = run_name
    agg["base_model"] = BASE
    agg["elapsed_s"] = round(time.time() - t0)
    # Record the conditions with the number. A before/after is only meaningful
    # if both runs were produced the same way, and six months from now the only
    # way to know that is if each run says so itself.
    agg["eval_dtype"] = str(next(model.parameters()).dtype).split(".")[-1]
    agg["four_bit"] = any(type(m).__name__ == "Linear4bit"
                          for m in model.modules())
    agg["n_rows"] = len(rows)
    agg["gen_tokens"] = dict(GEN_TOKENS)
    agg["greedy"] = True
    # Computed here, BEFORE results.json is written. It used to be derived down
    # with the printing, which meant the warning appeared on screen and the
    # field never reached disk — so the one fact that explains a low score was
    # missing from the only file anyone reads later.
    capped: dict[str, list] = {}
    for r in raws:
        capped.setdefault(r["task"], []).append(bool(r.get("hit_cap")))
    hits = {t: (sum(v), len(v)) for t, v in capped.items() if any(v)}
    agg["hit_cap"] = {t: {"n": n, "of": m} for t, (n, m) in hits.items()}

    (OUT / f"raw_{run_name}.json").write_text(
        json.dumps(raws, ensure_ascii=False, indent=2), encoding="utf-8")
    res_path = OUT / "results.json"
    allres = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    allres[run_name] = agg
    res_path.write_text(json.dumps(allres, indent=2), encoding="utf-8")
    # The run is in results.json now, so the checkpoint has done its job.
    # Leaving it would make a deliberate re-run silently replay stale rows.
    part_path.unlink(missing_ok=True)

    o = agg["overall"]
    print(f"\n  {run_name}: score={o['score']:.3f}  emit={o['emit_rate']:.3f}  "
          f"schema={o['schema']:.3f}  enums={o['enums']:.3f}  rigor={o['rigor']:.3f}")
    for sl, v in agg["by_slice"].items():
        print(f"    {sl:<5} score={v['score']:.3f}  emit={v['emit_rate']:.3f}")
    print(f"  extract modes: {agg['extract_modes']}")
    # A task that runs to its ceiling is being cut off, and a cut-off object
    # loses every required field it had not reached yet — which reads as a bad
    # model rather than a small budget. Name it.
    if hits:
        print("  ! ran to the generation ceiling (output was cut off): "
              + ", ".join(f"{t} {n}/{m}" for t, (n, m) in sorted(hits.items())))
    print(f"  {'task':<9}{'n':>4}{'emit':>7}{'schema':>8}{'enums':>7}"
          f"{'rigor':>7}{'score':>8}")
    for t, v in agg["tasks"].items():
        print(f"  {t:<9}{v['n']:>4}{v['emit_rate']:>7.2f}{v['schema']:>8.3f}"
              f"{v['enums']:>7.3f}{v['rigor']:>7.3f}{v['score']:>8.3f}")
    # A run where NOTHING parsed is nearly always a harness fault — wrong chat
    # template, a generation cap below the shortest valid object, an adapter
    # that failed to attach — not a model that is merely bad. Say so, because
    # the alternative is reading 0.000 as a legitimate baseline.
    if agg["overall"]["emit_rate"] == 0:
        print("\n  ! nothing parsed on ANY row. Before treating this as a"
              "\n    baseline, read out/raw_%s.json — if the text is prose or"
              "\n    empty rather than truncated JSON, the harness is wrong."
              % run_name)
    print()
    return agg


# ─────────────────────────────── train ───────────────────────────────────
def train(four_bit: bool, rank: int, alpha: int, epochs: float, name: str,
          max_len: int | None = None, smoke: bool = False,
          liger: bool | None = None):
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    # Resolve the memory strategy before anything is loaded, because it decides
    # the sequence window and therefore how much of the corpus survives.
    if liger is None:
        liger = liger_available()
    if liger and "use_liger_kernel" not in SFTConfig.__dataclass_fields__:
        print("  this TRL has no use_liger_kernel — falling back to plain loss")
        liger = False
    print(f"  base {BASE}")
    print(f"  fused CE (liger): {'ON' if liger else 'off'}")

    # Turing has no flash-attention backend, so PyTorch picks between the
    # memory-efficient kernel (cost linear in sequence length) and the math
    # fallback, which materialises a heads x seq x seq score matrix — 4.5 GB at
    # seq 6144. Disable math so the efficient kernel is used or we get a loud
    # error, rather than a silent 4.5 GB.
    # Prefer the memory-efficient kernel, but NEVER disable math. On Turing the
    # efficient kernel rejects this shape and flash does not exist, so removing
    # math leaves no backend at all and SDPA raises "Invalid backend" — which is
    # a harder failure than the memory cost it was meant to avoid.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as exc:
            print(f"  could not configure SDPA backends ({type(exc).__name__})")

    dt = dtype_for_gpu()
    kw = dict(dtype=dt, device_map="auto", attn_implementation="sdpa")
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    if four_bit:
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=True)
        # prepare_model_for_kbit_training upcasts every non-quantized parameter
        # to fp32 "for stability". For LayerNorms that is the point. For the
        # embedding and lm_head it is 2.2 GB of a 14.6 GB card spent on two
        # frozen 152064 x 3584 matrices that never receive a gradient — which is
        # what made the load footprint 7.5 GB instead of 5.4 GB. Send the big
        # frozen 2-D tensors back to half precision and leave the 1-D norms in
        # fp32, where the numerical argument actually applies.
        before = torch.cuda.memory_allocated()
        for p in model.parameters():
            if not p.requires_grad and p.dtype is torch.float32 and p.ndim > 1:
                p.data = p.data.to(dt)
        freed = (before - torch.cuda.memory_allocated()) / 2**30
        if freed > 0.05:
            print(f"  recovered {freed:.1f} GiB by keeping frozen embeddings "
                  f"in {str(dt).split('.')[-1]}")
    model.config.use_cache = False

    # Size the sequence window from MEASURED free memory, now that the weights
    # are actually resident.
    #
    # Four OOMs got here by predicting the weights term instead of measuring it:
    # the formula said 3.7 GiB, the card said 4.9. fp32 LayerNorms, NF4
    # quantization constants and RoPE caches are all real and none of them are
    # in a parameter-count formula. Everything downstream of load is genuinely
    # per-sequence and can be predicted; the resident footprint cannot, so read
    # it off the device.
    if torch.cuda.is_available():
        free, cap = (x / 2**30 for x in torch.cuda.mem_get_info())
        budget = free * 0.85            # leave room for allocator fragmentation
        c_ = model.config
        sm = torch.cuda.get_device_capability()[0]
        heads, hid = c_.num_attention_heads, c_.hidden_size
        hd, nl, vsz = hid // heads, c_.num_hidden_layers, c_.vocab_size
        print(f"  {cap - free:.1f} GiB resident, {free:.1f} GiB free "
              f"-> budget {budget:.1f} GiB for activations")
        # The logits term assumes the whole vocab x seq tensor is materialised.
        # Recent TRL computes the loss in chunks (_chunked_ce_forward), which
        # makes this an UPPER bound, sometimes by a couple of GiB. So the
        # estimate is used to rank windows, not to refuse one: a pinned window
        # that the estimate dislikes is still attempted, because the estimate is
        # deliberately pessimistic and the real answer is a try.
        want = [max_len] if max_len else WINDOWS
        pinned = max_len is not None
        chosen = None
        for s in want:
            attn = (heads * s * hd * 4 if sm >= 8
                    else heads * s * s * 2 * 2) / 2**30
            lg = vsz * s * 8 / 2**30
            need = attn + nl * s * hid * 2 / 2**30 + lg + 1.0
            ok = need <= budget
            print(f"  seq {s:>5}: needs <={need:>5.1f} GiB "
                  f"(attn {attn:.1f} + acts {nl*s*hid*2/2**30:.1f} + "
                  f"logits <={lg:.1f} + grads/opt 1.0)  "
                  + ("fits" if ok else "tight — logits term is an upper bound"))
            if ok:
                chosen = s
                break
        if chosen is None:
            chosen = want[-1]
            if pinned:
                print(f"  seq {chosen} pinned. The estimate is above budget, but "
                      f"its logits term ignores TRL's chunked cross-entropy, so "
                      f"the real figure is lower. Attempting it — an OOM here is "
                      f"itself the reportable result.")
            else:
                print(f"  ! nothing fits; forcing {chosen} and hoping")
        if chosen < 8192:
            # qe targets are long by nature, so a shorter window does not shrink
            # the corpus evenly — below 6144 it deletes Velma's task, and 8 of
            # the 18 real production eval rows are qe.
            lost = {6144: "29% of dev, 2% of qe", 4096: "50% of dev and 99% of qe",
                    3072: "50% of dev and 99% of qe"}.get(chosen, "long examples")
            print(f"  !! seq {chosen} < 8192 drops {lost}. A 24 GB GPU fits "
                  f"12288, which keeps 100% of every task.")
        max_len = chosen
    elif max_len is None:
        max_len = MAX_LEN_PLAIN
    print(f"  seq window {max_len}")

    # Needed on the TRAINING path too, not just eval. SFTTrainer calls
    # get_peft_model(), which walks the same dispatcher list and hits the same
    # torchao probe. It never fired during QLoRA training because the
    # bitsandbytes dispatcher matches a 4-bit layer first and PEFT never reaches
    # torchao — so this only shows up when the base is bf16.
    _defuse_torchao_check()

    peft_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    def prepare(split: str, max_len: int):
        """Fit examples to max_len by shrinking the PROMPT, never the target.

        The naive options are both bad. Letting the tokenizer truncate cuts from
        the right, and the right is the assistant target — that teaches the model
        to stop mid-JSON, the exact defect the truncated teacher rows were dropped
        to avoid. Dropping every over-long example instead throws away a fifth of
        the corpus, and not at random: it removes precisely the examples with the
        richest code context.

        So trim the middle of the user message instead. The seed's code snippet is
        the only part that is padding-like; the instructions at its head and the
        question at its tail both matter, so cut from the centre and leave a
        marker. The target survives intact every time, and an example is only
        dropped when system+target alone will not fit.
        """
        rows = load_split(split)
        keep, trimmed, dropped = [], 0, []
        for r in rows:
            msgs = r["messages"]
            n = len(tok.apply_chat_template(msgs, tokenize=True))
            if n <= max_len:
                keep.append(msgs)
                continue
            # Everything that is NOT the user body: system, target, chat scaffold.
            fixed = len(tok.apply_chat_template(
                [msgs[0], {"role": "user", "content": ""}, msgs[2]], tokenize=True))
            budget = max_len - fixed
            if budget < 128:
                dropped.append((r, n))          # target itself is too long
                continue
            ids = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
            mark = tok("\n\n... [context trimmed] ...\n\n",
                       add_special_tokens=False)["input_ids"]
            head = (budget - len(mark)) // 2
            tail = budget - len(mark) - head
            body = tok.decode(ids[:head] + mark + ids[-tail:])
            keep.append([msgs[0], {"role": "user", "content": body}, msgs[2]])
            trimmed += 1

        if trimmed or dropped:
            print(f"  {split}: {trimmed} prompts trimmed to fit {max_len}, "
                  f"{len(dropped)} dropped (target alone too long)")
            if dropped:
                per: dict[str, int] = {}
                for r, _ in dropped:
                    per[r["task"]] = per.get(r["task"], 0) + 1
                print("    dropped: "
                      + " ".join(f"{k}={v}" for k, v in sorted(per.items())))
        if smoke:
            # Longest-first: if the smoke test passes it, nothing later can OOM.
            keep.sort(key=lambda m: -sum(len(x["content"]) for x in m))
            keep = keep[:8 if split == "train" else 2]
        # Only "text" — build_dataset also carries seed_id/teacher, and a stray
        # column makes SFTTrainer's collator guess at what to do with it.
        return Dataset.from_list(
            [{"text": tok.apply_chat_template(m, tokenize=False)} for m in keep])

    tr = prepare("train", max_len)
    va = prepare("val", max_len)
    print(f"  training on {len(tr)} examples, validating on {len(va)}")

    cfg = SFTConfig(
        output_dir=str(OUT / name),
        num_train_epochs=1 if smoke else epochs,
        max_steps=4 if smoke else -1,
        per_device_train_batch_size=1,
        # Defaults to 8, which would build eight logits tensors at once and OOM
        # during the epoch-boundary eval — after training had already succeeded.
        per_device_eval_batch_size=1,
        # Keep only the loss. Otherwise Trainer accumulates every eval logits
        # tensor on the GPU to hand to compute_metrics, and there is no
        # compute_metrics here to use them.
        prediction_loss_only=True,
        # Effective batch 16. In smoke mode drop it to 2 — otherwise "4 steps"
        # means 64 forward/backward passes and the quick check isn't quick.
        gradient_accumulation_steps=2 if smoke else 16,
        gradient_checkpointing=True,
        # use_reentrant=False is not a tuning knob here, it is the difference
        # between checkpointing working and silently doing nothing. The
        # reentrant autograd path requires its inputs to require grad; under
        # PEFT the base weights are all frozen, so the check fails and the
        # blocks get recomputed-but-still-retained. Activations for all 32
        # layers then stay live, which is where ~5 GB went at seq 6144.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if smoke else 5,
        eval_strategy="no" if smoke else "epoch",
        # Checkpoint by STEPS, not epochs. Without Colab Pro+ there is no
        # background execution, so a 7.7 h run depends on a browser tab staying
        # open — and one epoch is ~2.5 h of work to lose. 827 examples at an
        # effective batch of 16 is 52 steps per epoch (156 total), so every 10
        # steps is a restore point roughly every 30 minutes. Each one is ~170 MB
        # of adapter plus 8-bit optimizer state; save_total_limit caps Drive
        # usage at ~340 MB.
        save_strategy="no" if smoke else "steps",
        save_steps=10,
        save_total_limit=2,
        bf16=(dt is torch.bfloat16),
        fp16=(dt is torch.float16),
        max_length=max_len,
        # Same optimiser for BOTH arms. Making this conditional on four_bit was
        # a design error: it gave QLoRA 8-bit paged AdamW and LoRA fp32 AdamW,
        # so the two runs differed in optimiser state as well as base precision
        # and no difference between them could be attributed cleanly.
        #
        # It also caused the LoRA OOM. fp32 moments for 41.9M trainable params
        # are 320 MiB against 80 MiB for the 8-bit pair — and the run died
        # asking for 168 MiB with 95 MiB free, a ~73 MiB shortfall. Holding the
        # optimiser constant frees 240 MiB, roughly three times the gap.
        optim="paged_adamw_8bit",
        report_to=[],
        **({"use_liger_kernel": True} if liger else {}),
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=tr,
                         eval_dataset=va, peft_config=peft_cfg)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  trainable {trainable:,} / {total:,} = {100*trainable/total:.4f}%")
    if torch.cuda.is_available():
        free, cap = torch.cuda.mem_get_info()
        vocab = getattr(model.config, "vocab_size", 0)
        print(f"  VRAM {(cap-free)/2**30:.1f}/{cap/2**30:.1f} GiB used after load; "
              f"logits need ~{vocab*max_len*8/2**30:.1f} GiB at seq {max_len} "
              f"(vocab {vocab:,})")
    # Pick up from the newest checkpoint if one is there. Re-running the cell
    # after a disconnect then continues instead of restarting from step 0 —
    # which, at ~25 minutes between restore points, is the difference between
    # losing a coffee break and losing a night.
    ckpts = sorted((OUT / name).glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if ckpts and not smoke:
        print(f"  resuming from {ckpts[-1].name} "
              f"(delete {OUT / name} to start over)")
        trainer.train(resume_from_checkpoint=str(ckpts[-1]))
    else:
        trainer.train()
    trainer.save_model(str(OUT / name))
    print(f"  saved adapter -> {OUT / name}")


def require_capable_gpu(allow: bool) -> None:
    """Stop on Turing instead of quietly producing a crippled adapter.

    Selecting a Colab runtime type is a separate action from buying Pro, so it
    is easy to pay for an L4 and still be scheduled on a T4. The symptom is not
    an obvious error — the window auto-sizer just steps down to 4096, which
    silently drops 99% of qe and 50% of dev, and the run either OOMs anyway or
    produces an adapter that cannot author a test plan. Both outcomes waste an
    hour, so fail loudly and early.
    """
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    gib = torch.cuda.mem_get_info()[1] / 2**30
    ok = major >= 8 and gib >= 20
    print(f"  GPU {name}  sm_{major}{minor}  {gib:.1f} GiB")
    if ok or allow:
        if not ok:
            print("  (proceeding anyway: --allow-small-gpu)")
        return
    why = []
    if major < 8:
        why.append(f"sm_{major}{minor} has no flash attention, so attention "
                   f"memory is heads x seq^2 instead of linear")
    if gib < 20:
        why.append(f"{gib:.1f} GiB cannot hold a 12288 window")
    print("\n  STOP — this GPU cannot train the full corpus.")
    for w in why:
        print(f"    - {w}")
    print("\n  At the window this card can afford (4096), 99% of qe and 50% of")
    print("  dev training examples no longer fit their targets. The adapter")
    print("  would never learn to author a test plan, and 8 of the 18 real")
    print("  production eval rows are qe.")
    print("\n  Fix: Runtime -> Change runtime type -> L4 GPU -> Save, then")
    print("  re-run the install, upload and Drive cells. Buying Colab Pro does")
    print("  not change the runtime type on its own.")
    print("\n  To override and accept a degraded adapter: --allow-small-gpu\n")
    sys.exit(2)


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["baseline", "lora", "qlora", "eval"])
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--base", help="override the base model id")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-length", type=int, default=None,
                    help="sequence window; default is the largest of "
                         f"{WINDOWS} predicted to fit this GPU")
    ap.add_argument("--liger", dest="liger", action="store_true", default=None,
                    help="force fused cross-entropy on")
    ap.add_argument("--no-liger", dest="liger", action="store_false",
                    help="force it off (use if training crashes in Triton)")
    ap.add_argument("--per-task", type=int, default=8,
                    help="generated eval rows per task; the real production "
                         "slice is always used in full. 8 gives 74 eval rows, "
                         "which is enough for per-task numbers to mean "
                         "something rather than being one or two samples.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--four-bit", action="store_true", default=None)
    ap.add_argument("--allow-small-gpu", action="store_true",
                    help="run on a GPU that cannot fit the full corpus, "
                         "accepting that qe and dev will be largely untrained")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 4 train steps / 2 eval rows per task. Proves "
                         "the wiring end to end before committing GPU hours.")
    a = ap.parse_args()
    if a.base:
        BASE = a.base
    require_capable_gpu(a.allow_small_gpu)

    if a.stage in ("lora", "qlora"):
        four_bit = a.stage == "qlora" if a.four_bit is None else a.four_bit
        train(four_bit, a.rank, a.alpha, a.epochs, a.name or a.stage,
              a.max_length, a.smoke, a.liger)
        return

    name = a.name or ("baseline" if a.stage == "baseline" else
                      pathlib.Path(a.adapter or "adapter").name)
    # Eval in half precision when the card can hold the weights.
    #
    # 4-bit is a TRAINING technique — it exists so the frozen base fits while
    # gradients flow. At inference it is actively harmful: every matmul pays a
    # dequantization cost, measured here at 8.2 decode steps/s versus ~28 in
    # bf16. Across three evals that is 7.9 hours instead of 2.3, for identical
    # weights. Both the baseline and the adapter take this same path, so the
    # comparison stays like-for-like; only the wall clock changes.
    if a.four_bit is None:
        big = (torch.cuda.is_available()
               and torch.cuda.mem_get_info()[1] / 2**30 >= 20)
        four_bit = not big
        if big:
            print("  eval in bf16/fp16 (not 4-bit): same weights, ~3x faster "
                  "decode. 4-bit is for training, not inference.")
    else:
        four_bit = a.four_bit
    rows = build_eval_set(a.per_task, include_real=True)
    if a.smoke:
        # One per task, capped at 1200 tokens.
        #
        # The previous 400-token cap guaranteed a useless result: the SHORTEST
        # task's median target is 672 tokens, so every answer was truncated and
        # every score was 0.000 — which is indistinguishable from the model
        # emitting garbage. 1200 lets bug, review and plan finish a whole
        # object, so a non-zero emit on those is real evidence the chat
        # template, generation and JSON extraction all work end to end.
        seen: set[str] = set()
        picked = []
        for r in rows:
            if r["task"] not in seen:
                seen.add(r["task"])
                picked.append(r)
        rows = picked
        for k in GEN_TOKENS:
            GEN_TOKENS[k] = min(GEN_TOKENS[k], 1200)
        name = f"smoke_{name}"
        print("  smoke: 1 row/task at a 1200-token cap. bug/review/plan should "
              "parse; qe/dev will truncate and that is expected.")
    print(f"\n  eval set: {len(rows)} rows "
          f"({sum(1 for r in rows if r['slice']=='real')} real production, "
          f"{sum(1 for r in rows if r['slice']=='gen')} generated)")
    print(f"  model {BASE}  4bit={four_bit}  adapter={a.adapter or 'none'}\n")
    model, tok = load_model(a.adapter, four_bit)
    run_eval(model, tok, rows, a.batch, name)


if __name__ == "__main__":
    main()
