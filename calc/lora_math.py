"""
Exact LoRA parameter arithmetic.

Every number this module prints is derived, not quoted — so the calculations
can go straight into a post and survive someone checking them.

Core identity
-------------
A full weight update for a (d_in x d_out) matrix costs d_in * d_out parameters.
LoRA never materialises it. It learns two thin matrices instead:

    dW  ~=  A @ B          A: (d_in x r)      B: (r x d_out)

    trainable = r * (d_in + d_out)

and at inference the update is folded back in as W' = W + (alpha / r) * A @ B,
so there is zero added latency once merged.
"""
from dataclasses import dataclass, field


# ─────────────────────────── single matrix ────────────────────────────
def lora_matrix(d_in: int, d_out: int, r: int) -> dict:
    """Parameter accounting for one LoRA-adapted weight matrix."""
    full = d_in * d_out
    a = d_in * r
    b = r * d_out
    low = a + b
    return {
        "d_in": d_in, "d_out": d_out, "r": r,
        "full": full,
        "A": a, "B": b, "lora": low,
        "reduction": full / low,
        "pct": 100.0 * low / full,
        "saved": full - low,
    }


# ─────────────────────────── model presets ────────────────────────────
@dataclass
class ModelSpec:
    """Only the dimensions that affect LoRA arithmetic."""
    name: str
    params: int              # total parameters
    hidden: int              # d_model
    layers: int
    heads: int
    kv_heads: int            # < heads means grouped-query attention
    intermediate: int        # MLP inner dimension
    vocab: int = 32000

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def target_shapes(self) -> dict:
        """(d_in, d_out) for every attention / MLP projection, per layer.

        Note q/k/v are NOT all square once GQA is in play: k and v project
        down to kv_heads * head_dim, which is where a lot of naive LoRA
        parameter estimates go wrong.
        """
        h, hd = self.hidden, self.head_dim
        kv = self.kv_heads * hd
        return {
            "q_proj":    (h, h),
            "k_proj":    (h, kv),
            "v_proj":    (h, kv),
            "o_proj":    (h, h),
            "gate_proj": (h, self.intermediate),
            "up_proj":   (h, self.intermediate),
            "down_proj": (self.intermediate, h),
        }


PRESETS = {
    # name                        params        hidden  L   H   KV  ffn     vocab
    "smollm2-360m":  ModelSpec("SmolLM2-360M",     362_000_000,  960, 32, 15,  5,  2560, 49152),
    "qwen2.5-0.5b":  ModelSpec("Qwen2.5-0.5B",     494_000_000,  896, 24, 14,  2,  4864, 151936),
    "llama3.2-1b":   ModelSpec("Llama-3.2-1B",   1_236_000_000, 2048, 16, 32,  8,  8192, 128256),
    "mistral-7b":    ModelSpec("Mistral-7B-v0.3",7_248_000_000, 4096, 32, 32,  8, 14336, 32768),
    "llama3.1-8b":   ModelSpec("Llama-3.1-8B",   8_030_000_000, 4096, 32, 32,  8, 14336, 128256),
    "llama2-70b":    ModelSpec("Llama-2-70B",   68_977_000_000, 8192, 80, 64,  8, 28672, 32000),
}

ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTN_ONLY + ["gate_proj", "up_proj", "down_proj"]


# ─────────────────────────── whole model ──────────────────────────────
@dataclass
class LoraBudget:
    spec: ModelSpec
    r: int
    alpha: int
    targets: list
    per_target: dict = field(default_factory=dict)
    trainable: int = 0
    full_delta: int = 0

    @property
    def pct_of_base(self) -> float:
        return 100.0 * self.trainable / self.spec.params

    @property
    def reduction_vs_full_ft(self) -> float:
        return self.spec.params / self.trainable

    @property
    def scaling(self) -> float:
        """The alpha/r factor actually applied to the update at merge time."""
        return self.alpha / self.r


def lora_budget(spec: ModelSpec, r: int, alpha: int, targets=None) -> LoraBudget:
    targets = targets or ALL_LINEAR
    shapes = spec.target_shapes()
    out = LoraBudget(spec=spec, r=r, alpha=alpha, targets=targets)
    for t in targets:
        d_in, d_out = shapes[t]
        m = lora_matrix(d_in, d_out, r)
        m["per_layer"] = m["lora"]
        m["all_layers"] = m["lora"] * spec.layers
        m["full_all_layers"] = m["full"] * spec.layers
        out.per_target[t] = m
        out.trainable += m["all_layers"]
        out.full_delta += m["full_all_layers"]
    return out


# ─────────────────────────── rank guidance ────────────────────────────
RANK_TABLE = [
    ("< 1B params", [4, 8, 16]),
    ("1B - 7B",     [8, 16, 32]),
    ("> 13B",       [16, 32, 64]),
    ("65B+",        [64, 128]),
]


def suggest_rank(params: int) -> list:
    if params < 1_000_000_000:
        return RANK_TABLE[0][1]
    if params <= 7_000_000_000:
        return RANK_TABLE[1][1]
    if params <= 13_000_000_000:
        return RANK_TABLE[1][1]
    if params < 65_000_000_000:
        return RANK_TABLE[2][1]
    return RANK_TABLE[3][1]


# ─────────────────────────── formatting ───────────────────────────────
def human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:,.0f}"


def textbook_example(d: int = 512, r: int = 8) -> str:
    """The square-matrix walkthrough, reproduced exactly."""
    m = lora_matrix(d, d, r)
    return "\n".join([
        f"  dW is ({d} x {d})        -> {m['full']:,} parameters",
        f"  A  is ({d} x {r})        -> {m['A']:,}",
        f"  B  is ({r} x {d})        -> {m['B']:,}",
        f"  {'-'*46}",
        f"  LoRA trains A + B       -> {m['lora']:,}",
        f"  reduction               -> {m['reduction']:.0f}x   "
        f"({m['pct']:.3f}% of the full update)",
        f"  parameters not trained  -> {m['saved']:,}",
    ])
