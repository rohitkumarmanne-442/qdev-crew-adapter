"""
Deterministic scorer for the QDev crew adapter.

No judge model, no API calls, no randomness — the same philosophy as the
platform's governance eval harness. Run it on a laptop, run it in CI, run it
a thousand times and get the same number.

Design note on fairness: `extract_json` is deliberately GENEROUS. A base model
that wraps valid JSON in prose or a markdown fence still gets full credit for
emitting it. Being strict there would inflate the before/after gap by
punishing formatting the tokenizer never saw, which would make the whole
comparison dishonest.
"""
from __future__ import annotations

import json
import re

from .schemas import SPECS, FILLER_PATTERNS, FILLER_EXACT, PRIORITY

FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_FILLER_RE = [re.compile(p, re.I) for p in FILLER_PATTERNS]


# ══════════════════════════ extraction ═══════════════════════════════════
def extract_json(text: str) -> tuple[dict | None, str]:
    """Pull the most plausible JSON object out of a raw completion.

    Returns (obj, how) where `how` records which strategy succeeded, so the
    report can show *how* the base model was failing, not just that it did.
    """
    if not text or not text.strip():
        return None, "empty"
    t = text.strip()

    # 1. the whole thing is JSON
    try:
        o = json.loads(t)
        return (o, "clean") if isinstance(o, dict) else (None, "not-object")
    except Exception:
        pass

    # 2. inside a markdown fence
    for m in FENCE.findall(t):
        try:
            o = json.loads(m.strip())
            if isinstance(o, dict):
                return o, "fenced"
        except Exception:
            continue

    # 3. brace matching — longest balanced object anywhere in the text
    best = None
    for start in (i for i, ch in enumerate(t) if ch == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = t[start:i + 1]
                    if best is None or len(frag) > len(best):
                        best = frag
                    break
        if best and len(best) > len(t) * 0.6:
            break
    if best:
        try:
            o = json.loads(best)
            if isinstance(o, dict):
                return o, "embedded"
        except Exception:
            pass

    # 4. a tool_use envelope
    for key in ("input", "arguments", "parameters", "tool_input"):
        m = re.search(rf'"{key}"\s*:\s*(\{{)', t)
        if m:
            sub, _ = extract_json(t[m.start(1):])
            if sub:
                return sub, "tool-envelope"
    return None, "unparseable"


# ══════════════════════════ generic checks ═══════════════════════════════
def _ratio(ok: int, total: int) -> float:
    return 1.0 if total == 0 else ok / total


def check_schema(obj: dict, spec: dict) -> tuple[float, list]:
    notes, ok, total = [], 0, 0
    arrays = spec.get("arrays", {})
    for k in spec["required"]:
        total += 1
        v = obj.get(k)
        # An empty array is only a failure when the array rule demands members.
        # `review.findings` has min=0 because a clean APPROVE genuinely has no
        # findings — counting that as missing would score correct output as
        # broken, and (worse) teach the adapter to always invent a finding.
        allow_empty = isinstance(v, list) and arrays.get(k, {}).get("min", 1) == 0
        if v is None or (isinstance(v, (str, list, dict))
                         and len(v) == 0 and not allow_empty):
            notes.append(f"missing/empty:{k}")
        else:
            ok += 1
    for k, want in spec["types"].items():
        if k in obj and obj[k] is not None:
            total += 1
            if isinstance(obj[k], want):
                ok += 1
            else:
                notes.append(f"wrongtype:{k}={type(obj[k]).__name__}")
    for k, rule in spec.get("arrays", {}).items():
        v = obj.get(k)
        if isinstance(v, list):
            total += 1
            if len(v) >= rule["min"] and all(isinstance(x, rule["of"]) for x in v):
                ok += 1
            else:
                notes.append(f"array:{k} n={len(v)} min={rule['min']}")
    for arr, keys in spec.get("item_required", {}).items():
        for i, item in enumerate(obj.get(arr, []) or []):
            if not isinstance(item, dict):
                notes.append(f"item:{arr}[{i}] not-object")
                total += 1
                continue
            for k in keys:
                total += 1
                if item.get(k):
                    ok += 1
                else:
                    notes.append(f"item:{arr}[{i}].{k}")
    return _ratio(ok, total), notes


def check_enums(obj: dict, spec: dict) -> tuple[float, list]:
    notes, ok, total = [], 0, 0
    for k, allowed in spec.get("enums", {}).items():
        if k in obj:
            total += 1
            if obj[k] in allowed:
                ok += 1
            else:
                notes.append(f"enum:{k}={obj[k]!r}")
    for arr, rules in spec.get("item_enums", {}).items():
        for i, item in enumerate(obj.get(arr, []) or []):
            if not isinstance(item, dict):
                continue
            for k, allowed in rules.items():
                if k in item:
                    total += 1
                    if item[k] in allowed:
                        ok += 1
                    else:
                        notes.append(f"enum:{arr}[{i}].{k}={item[k]!r}")
    return _ratio(ok, total), notes


def _walk_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)
    elif isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)


def check_clean(obj: dict) -> tuple[float, list]:
    """Penalise placeholder text — the classic way a weak model looks
    schema-correct while being useless.

    Deliberately conservative: matches whole words and template markers only.
    'fill in all fields' and 'input placeholder text' are real QA phrases and
    must NOT be flagged.
    """
    hits: list[str] = []
    for s in _walk_strings(obj):
        low = s.strip().lower()
        if low in FILLER_EXACT:
            hits.append(f"empty-field:{low}")
            continue
        for pat in _FILLER_RE:
            m = pat.search(low)
            if m:
                hits.append(f"filler:{m.group(0)[:24]}")
                break
    uniq = list(dict.fromkeys(hits))
    return max(0.0, 1.0 - 0.25 * len(uniq)), uniq[:4]


# ══════════════════════════ per-task rigor ═══════════════════════════════
def rigor_plan(o: dict) -> tuple[float, list]:
    n, notes = [], []
    ac = o.get("acceptance_criteria") or []
    n.append(1.0 if len(ac) >= 3 else len(ac) / 3)
    good = sum(1 for a in ac if isinstance(a, str) and len(a) >= 20)
    n.append(_ratio(good, len(ac)))
    if len(ac) < 3:
        notes.append(f"thin-ac:{len(ac)}")
    n.append(1.0 if 0 < len(str(o.get("title", ""))) <= 120 else 0.0)
    n.append(1.0 if len(str(o.get("description_md", ""))) >= 100 else 0.0)
    return sum(n) / len(n), notes


def rigor_dev(o: dict) -> tuple[float, list]:
    ch = [c for c in (o.get("changes") or []) if isinstance(c, dict)]
    if not ch:
        return 0.0, ["no-changes"]
    n, notes = [], []
    rel = sum(1 for c in ch if isinstance(c.get("path"), str)
              and not c["path"].startswith("/") and "\\" not in c["path"])
    n.append(_ratio(rel, len(ch)))
    if rel < len(ch):
        notes.append("bad-paths")
    paths = [c.get("path") for c in ch]
    n.append(1.0 if len(set(paths)) == len(paths) else 0.5)
    body = str(o.get("pr_body", ""))
    n.append(1.0 if ("##" in body and len(body) > 80) else 0.0)
    cm = str(o.get("commit_message", ""))
    n.append(1.0 if re.match(r"^(feat|fix|chore|refactor|docs|test|perf)(\(.+\))?: .+", cm) else 0.0)
    if not re.match(r"^(feat|fix|chore|refactor|docs|test|perf)", cm):
        notes.append("non-conventional-commit")
    return sum(n) / len(n), notes


def rigor_review(o: dict) -> tuple[float, list]:
    f = [x for x in (o.get("findings") or []) if isinstance(x, dict)]
    n, notes = [], []
    located = sum(1 for x in f if x.get("file") and x.get("issue"))
    n.append(_ratio(located, len(f)) if f else 1.0)
    # verdict must be consistent with the findings it reports
    sev = {str(x.get("severity", "")).lower() for x in f}
    v = str(o.get("verdict", "")).upper()
    bad = ("critical" in sev or "major" in sev)
    consistent = (v in ("REQUEST CHANGES", "BLOCK")) if bad else True
    n.append(1.0 if consistent else 0.0)
    if not consistent:
        notes.append(f"verdict-inconsistent:{v}-with-{sorted(sev)}")
    n.append(1.0 if len(str(o.get("summary", ""))) >= 60 else 0.0)
    return sum(n) / len(n), notes


def rigor_qe(o: dict) -> tuple[float, list]:
    suites = [s for s in (o.get("suites") or []) if isinstance(s, dict)]
    cases = [c for s in suites for c in (s.get("cases") or []) if isinstance(c, dict)]
    if not cases:
        return 0.0, ["no-cases"]
    n, notes = [], []
    # ids contiguous TC-001, TC-002, …
    ids = [str(c.get("id", "")) for c in cases]
    want = [f"TC-{i:03d}" for i in range(1, len(cases) + 1)]
    n.append(_ratio(sum(1 for a, b in zip(ids, want) if a == b), len(cases)))
    if ids[:1] and ids != want:
        notes.append("ids-not-contiguous")
    # every step is an action/expected pair
    steps = [s for c in cases for s in (c.get("steps") or [])]
    paired = sum(1 for s in steps if isinstance(s, dict) and s.get("action") and s.get("expected"))
    n.append(_ratio(paired, len(steps)) if steps else 0.0)
    if steps and paired < len(steps):
        notes.append(f"unpaired-steps:{len(steps)-paired}")
    # traceability back to acceptance criteria
    traced = sum(1 for c in cases if c.get("covers_ac"))
    n.append(_ratio(traced, len(cases)))
    if traced < len(cases) * 0.8:
        notes.append("weak-traceability")
    # priorities valid
    n.append(_ratio(sum(1 for c in cases if c.get("priority") in PRIORITY), len(cases)))
    # coverage breadth: the prompt asks for happy + negative + edge + security
    cats = {s.get("category") for s in suites}
    n.append(min(1.0, len(cats & {"happy", "negative", "edge", "security", "accessibility"}) / 4))
    # volume: the system prompt asks for 12-20 cases
    n.append(1.0 if 12 <= len(cases) <= 22 else max(0.0, 1 - abs(len(cases) - 16) / 16))
    return sum(n) / len(n), notes


def rigor_bug(o: dict) -> tuple[float, list]:
    n, notes = [], []
    steps = o.get("steps_to_reproduce") or []
    n.append(1.0 if len(steps) >= 2 else 0.0)
    exp, act = str(o.get("expected", "")), str(o.get("actual", ""))
    n.append(1.0 if exp and act and exp.strip() != act.strip() else 0.0)
    if exp.strip() == act.strip():
        notes.append("expected==actual")
    n.append(1.0 if len(str(o.get("suspected_cause", ""))) >= 40 else 0.0)
    return sum(n) / len(n), notes


def rigor_devops(o: dict) -> tuple[float, list]:
    n, notes = [], []
    n.append(1.0 if len(o.get("steps") or []) >= 3 else 0.5)
    n.append(1.0 if len(str(o.get("rollback", ""))) >= 40 else 0.0)
    if len(str(o.get("rollback", ""))) < 40:
        notes.append("thin-rollback")
    n.append(1.0 if len(o.get("checks") or []) >= 2 else 0.5)
    return sum(n) / len(n), notes


def rigor_custom(o: dict) -> tuple[float, list]:
    acts = [a for a in (o.get("actions") or []) if isinstance(a, dict)]
    if not acts:
        return 0.0, ["no-actions"]
    full = sum(1 for a in acts if a.get("title") and len(str(a.get("detail", ""))) >= 30)
    return _ratio(full, len(acts)), ([] if full == len(acts) else ["thin-actions"])


RIGOR = {"plan": rigor_plan, "dev": rigor_dev, "review": rigor_review,
         "qe": rigor_qe, "bug": rigor_bug, "devops": rigor_devops,
         "custom": rigor_custom}

WEIGHTS = {"schema": 0.30, "enums": 0.20, "rigor": 0.35, "clean": 0.15}


# ══════════════════════════ public API ═══════════════════════════════════
def score_one(task: str, raw: str) -> dict:
    """Score a single raw completion for a single task."""
    spec = SPECS[task]
    obj, how = extract_json(raw)
    if obj is None:
        return {"task": task, "emitted": 0, "extract": how,
                "schema": 0.0, "enums": 0.0, "rigor": 0.0, "clean": 0.0,
                "score": 0.0, "notes": [f"extract:{how}"],
                "chars": len(raw or "")}
    s, n1 = check_schema(obj, spec)
    e, n2 = check_enums(obj, spec)
    r, n3 = RIGOR[task](obj)
    c, n4 = check_clean(obj)
    total = (WEIGHTS["schema"] * s + WEIGHTS["enums"] * e
             + WEIGHTS["rigor"] * r + WEIGHTS["clean"] * c)
    return {"task": task, "emitted": 1, "extract": how,
            "schema": s, "enums": e, "rigor": r, "clean": c,
            "score": total, "notes": (n1 + n2 + n3 + n4)[:8],
            "chars": len(raw or "")}


def aggregate(rows: list[dict]) -> dict:
    """Per-task and overall roll-up. Overall is the mean of task means so a
    task with more examples cannot dominate the headline number."""
    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    tasks = {}
    for t, rs in sorted(by_task.items()):
        tasks[t] = {
            "n": len(rs),
            "emit_rate": mean([r["emitted"] for r in rs]),
            "schema": mean([r["schema"] for r in rs]),
            "enums": mean([r["enums"] for r in rs]),
            "rigor": mean([r["rigor"] for r in rs]),
            "clean": mean([r["clean"] for r in rs]),
            "score": mean([r["score"] for r in rs]),
        }
    keys = ["emit_rate", "schema", "enums", "rigor", "clean", "score"]
    overall = {k: mean([tasks[t][k] for t in tasks]) for k in keys}
    overall["n"] = len(rows)
    # how the failures were distributed — the interesting half of the story
    modes: dict[str, int] = {}
    for r in rows:
        modes[r["extract"]] = modes.get(r["extract"], 0) + 1
    return {"tasks": tasks, "overall": overall, "extract_modes": modes}
