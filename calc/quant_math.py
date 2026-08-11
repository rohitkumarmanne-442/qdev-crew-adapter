"""
Quantization arithmetic: int8 affine, NF4, double quantization, QLoRA memory.

Everything is computed from first principles so the worked examples can be
published and checked.
"""
import math
from dataclasses import dataclass

BITS = {"float32": 32, "bfloat16": 16, "float16": 16, "int8": 8, "nf4": 4, "int4": 4}


# ══════════════════════ 1. how big is a model ══════════════════════════
def model_bytes(params: int, precision: str) -> int:
    return params * BITS[precision] // 8


def memory_table(params: int, precisions=("float32", "float16", "int8", "nf4")) -> list:
    """The 'why quantize at all' table: 7B in each precision."""
    rows = []
    base = model_bytes(params, "float32")
    for p in precisions:
        b = model_bytes(params, p)
        rows.append({
            "precision": p,
            "bits": BITS[p],
            "bytes_per_param": BITS[p] / 8,
            "gib": b / 1024**3,
            "gb": b / 1e9,
            "shrink": base / b,
        })
    return rows


# ══════════════════════ 2. int8 affine quantization ════════════════════
@dataclass
class AffineQuant:
    """Asymmetric (zero-point) quantization of a real range onto an int grid.

    quantize:    q     = round((x - zero_point) / scale)
    dequantize:  x_hat = scale * q + zero_point

    `force_zero_point` pins the zero point (0.0 gives the symmetric form used
    in most hand calculations); leave it None for the exact affine fit.
    """
    x_min: float
    x_max: float
    q_min: int
    q_max: int
    force_zero_point: float | None = None

    @property
    def scale(self) -> float:
        return (self.x_max - self.x_min) / (self.q_max - self.q_min)

    @property
    def zero_point(self) -> float:
        """The real value that maps to integer 0."""
        if self.force_zero_point is not None:
            return self.force_zero_point
        return self.x_min - self.q_min * self.scale

    def quantize(self, x: float) -> int:
        q = round((x - self.zero_point) / self.scale)
        return max(self.q_min, min(self.q_max, q))

    def dequantize(self, q: int) -> float:
        return self.scale * q + self.zero_point

    def roundtrip(self, x: float) -> dict:
        q = self.quantize(x)
        xh = self.dequantize(q)
        return {"x": x, "q": q, "x_hat": xh, "abs_err": abs(x - xh)}


def int8_symmetric(x_min=-1.0, x_max=1.0) -> AffineQuant:
    """The classic float32 -> int8 mapping used in the worked example.

    Grid runs -127..128, which is why the denominator is 255 and not 256.
    """
    return AffineQuant(x_min=x_min, x_max=x_max, q_min=-127, q_max=128)


# ══════════════════════ 3. NF4 (4-bit NormalFloat) ═════════════════════
# 16 levels placed at equal-probability quantiles of a standard normal,
# then normalised to [-1, 1]. Uniform int4 would space these evenly and waste
# most of its codes on the tails, where almost no weights live.
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def int4_uniform_levels() -> list:
    """Standard symmetric absmax int4 — the honest alternative to NF4.

    Real int4 absmax quantizers use scale = absmax/7 and codes in [-7, 7].
    That is 15 of the 16 available codes (one is left unused), and crucially
    it INCLUDES an exact zero, which matters because most weights sit near it.
    Using an evenly-spaced 16-level grid instead would have no zero level and
    would stack the comparison unfairly against int4.
    """
    return [i / 7.0 for i in range(-7, 8)]


def bin_probability_mass(n_bins: int = 16) -> float:
    """Quantile quantization gives every bin the same probability mass."""
    return 100.0 / n_bins


def _nearest(levels, v):
    best = min(levels, key=lambda L: abs(L - v))
    return levels.index(best), best


def quantize_block(weights, levels) -> dict:
    """Block-wise quantization: normalise by absmax, snap to the code book."""
    absmax = max(abs(w) for w in weights) or 1.0
    codes, recon = [], []
    for w in weights:
        i, L = _nearest(levels, w / absmax)
        codes.append(i)
        recon.append(L * absmax)
    err = [abs(a - b) for a, b in zip(weights, recon)]
    return {
        "absmax": absmax, "codes": codes, "recon": recon,
        "mae": sum(err) / len(err),
        "max_err": max(err),
        "rmse": math.sqrt(sum(e * e for e in err) / len(err)),
    }


def _blockwise(weights, levels, block: int = 64) -> dict:
    """Quantize a long weight vector the way bitsandbytes does: independent
    blocks, each normalised by its own absmax."""
    sq = tot = 0.0
    mx = 0.0
    n = 0
    for i in range(0, len(weights), block):
        chunk = weights[i:i + block]
        if not chunk:
            continue
        r = quantize_block(chunk, levels)
        for a, b in zip(chunk, r["recon"]):
            e = abs(a - b)
            sq += e * e
            tot += e
            mx = max(mx, e)
            n += 1
    return {"rmse": math.sqrt(sq / n), "mae": tot / n, "max_err": mx, "n": n}


def nf4_vs_int4(weights, block: int = 64) -> dict:
    """Head-to-head over many blocks — one block of 64 is pure noise.

    Needs a realistic sample size (>= ~50k weights) before the difference
    between the two code books is measurable rather than accidental.
    """
    nf4 = _blockwise(weights, NF4_LEVELS, block)
    i4 = _blockwise(weights, int4_uniform_levels(), block)
    return {
        "nf4": nf4, "int4": i4,
        "rmse_improvement_pct": 100.0 * (i4["rmse"] - nf4["rmse"]) / i4["rmse"],
    }


# ══════════════════════ 4. double quantization ═════════════════════════
def double_quant(block: int = 64, second_block: int = 256,
                 const_bits: int = 32, dq_bits: int = 8) -> dict:
    """Cost of the quantization constants, before and after quantizing them.

    Every block of `block` weights needs one absmax. Stored raw that is
    const_bits per block. Double quantization stores those absmaxes as int8
    and adds one fp32 scale per `second_block` of them.
    """
    before = const_bits / block                       # bits per weight
    after = dq_bits / block + const_bits / (block * second_block)
    return {
        "block": block, "second_block": second_block,
        "bits_per_param_before": before,
        "bits_per_param_after": after,
        "bits_saved_per_param": before - after,
        "gb_saved_per_1b_params": (before - after) * 1e9 / 8 / 1e9,
    }


# ══════════════════════ 5. QLoRA training memory ═══════════════════════
@dataclass
class TrainMemory:
    base_weights_gb: float
    quant_constants_gb: float
    adapter_gb: float
    gradients_gb: float
    optimizer_gb: float

    @property
    def total_gb(self) -> float:
        return (self.base_weights_gb + self.quant_constants_gb + self.adapter_gb
                + self.gradients_gb + self.optimizer_gb)


def training_memory(params: int, trainable: int, base_precision: str,
                    adapter_bits: int = 16, optimizer: str = "adamw",
                    use_double_quant: bool = True, paged: bool = False) -> TrainMemory:
    """Static (weights + states) training memory. Activations are separate and
    depend on batch/sequence length + gradient checkpointing."""
    base = model_bytes(params, base_precision) / 1e9

    const = 0.0
    if base_precision in ("nf4", "int4"):
        dq = double_quant()
        bits = dq["bits_per_param_after"] if use_double_quant else dq["bits_per_param_before"]
        const = params * bits / 8 / 1e9

    adapter = trainable * adapter_bits / 8 / 1e9
    grads = trainable * adapter_bits / 8 / 1e9
    # AdamW keeps two fp32 moments per trainable parameter
    opt_bytes = {"adamw": 8, "adamw8bit": 2, "sgd": 4}[optimizer]
    opt = trainable * opt_bytes / 1e9
    if paged:
        opt = 0.0  # paged to host RAM; counted against CPU, not VRAM

    return TrainMemory(base, const, adapter, grads, opt)


def full_finetune_memory(params: int, precision: str = "bfloat16",
                         optimizer: str = "adamw") -> float:
    """What full fine-tuning would have cost — the number LoRA is measured against."""
    w = model_bytes(params, precision) / 1e9
    g = model_bytes(params, precision) / 1e9
    opt_bytes = {"adamw": 8, "adamw8bit": 2, "sgd": 4}[optimizer]
    o = params * opt_bytes / 1e9
    master = params * 4 / 1e9 if precision in ("bfloat16", "float16") else 0.0
    return w + g + o + master
