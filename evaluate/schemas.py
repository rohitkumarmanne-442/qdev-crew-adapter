"""
The seven SDLC tasks the crew adapter has to learn, and what a correct
output looks like for each.

Four of these mirror tool schemas that already exist in the platform
(`draft_story`, `propose_changes`, `author_test_plan`). Three are new
structured contracts for phases that currently return prose:

  * review  — Brain returns markdown today and the verdict is recovered by
              substring matching in manager._extract_verdict(). A real enum
              plus per-finding file/line/severity removes that guesswork.
  * bug     — Scooby's triage output.
  * devops  — Perry's release plan.

`custom` is the generalisation test: an agent persona the adapter never saw,
which must still emit house-style structured output.
"""

TASKS = ["plan", "dev", "review", "qe", "bug", "devops", "custom"]

AGENT_FOR = {
    "plan": "Phineas", "dev": "Ferb", "review": "Brain", "qe": "Velma",
    "bug": "Scooby", "devops": "Perry", "custom": "Custom",
}

TOOL_FOR = {
    "plan": "draft_story",
    "dev": "propose_changes",
    "review": "submit_review",
    "qe": "author_test_plan",
    "bug": "file_bug",
    "devops": "plan_release",
    "custom": "respond",
}

# ── enums, taken from the live platform where one already exists ──────────
EFFORT = ["S", "M", "L", "XL"]
PRIORITY = ["P0", "P1", "P2", "P3"]
SUITE_CATEGORY = ["happy", "negative", "edge", "security",
                  "accessibility", "performance", "regression"]
CHANGE_ACTION = ["create", "modify"]
VERDICT = ["APPROVE", "APPROVE WITH NITS", "REQUEST CHANGES", "BLOCK"]
SEVERITY = ["critical", "major", "minor", "nit"]
BUG_SEVERITY = ["S1", "S2", "S3", "S4"]
RELEASE_STRATEGY = ["direct", "canary", "blue-green", "feature-flag"]

# ── field contracts ───────────────────────────────────────────────────────
# ("path.to.field", required?, type, enum-or-None, min_len)
SPECS: dict[str, dict] = {
    "plan": {
        "required": ["title", "description_md", "acceptance_criteria", "effort"],
        "types": {"title": str, "description_md": str,
                  "acceptance_criteria": list, "suggested_labels": list,
                  "effort": str, "file_hints": list},
        "enums": {"effort": EFFORT},
        "arrays": {"acceptance_criteria": {"min": 3, "of": str}},
    },
    "dev": {
        "required": ["commit_message", "pr_title", "pr_body", "summary", "changes"],
        "types": {"commit_message": str, "pr_title": str, "pr_body": str,
                  "summary": str, "changes": list},
        "enums": {},
        "arrays": {"changes": {"min": 1, "of": dict}},
        "item_required": {"changes": ["path", "action", "description"]},
        "item_enums": {"changes": {"action": CHANGE_ACTION}},
    },
    "review": {
        "required": ["verdict", "summary", "findings"],
        "types": {"verdict": str, "summary": str, "findings": list},
        "enums": {"verdict": VERDICT},
        "arrays": {"findings": {"min": 0, "of": dict}},
        "item_required": {"findings": ["file", "severity", "issue"]},
        "item_enums": {"findings": {"severity": SEVERITY}},
    },
    "qe": {
        "required": ["summary", "suites"],
        "types": {"summary": str, "suites": list},
        "enums": {},
        "arrays": {"suites": {"min": 1, "of": dict}},
        "item_required": {"suites": ["name", "category", "cases"]},
        "item_enums": {"suites": {"category": SUITE_CATEGORY}},
    },
    "bug": {
        "required": ["title", "severity", "steps_to_reproduce",
                     "expected", "actual", "suspected_cause"],
        "types": {"title": str, "severity": str, "steps_to_reproduce": list,
                  "expected": str, "actual": str, "suspected_cause": str},
        "enums": {"severity": BUG_SEVERITY},
        "arrays": {"steps_to_reproduce": {"min": 2, "of": str}},
    },
    "devops": {
        "required": ["strategy", "steps", "rollback", "checks"],
        "types": {"strategy": str, "steps": list, "rollback": str, "checks": list},
        "enums": {"strategy": RELEASE_STRATEGY},
        "arrays": {"steps": {"min": 2, "of": str}, "checks": {"min": 1, "of": str}},
    },
    "custom": {
        "required": ["summary", "actions"],
        "types": {"summary": str, "actions": list},
        "enums": {},
        "arrays": {"actions": {"min": 1, "of": dict}},
        "item_required": {"actions": ["title", "detail"]},
        "item_enums": {},
    },
}

# Placeholder detection — deterministic proxy for "the model padded instead of
# doing the work", no judge model required.
#
# These MUST be precise. An earlier substring version fired on "fill in all
# fields" and "input placeholder text" in real, correct QA test steps, which
# penalised good output and would have understated the tuned model. Anything
# that can legitimately appear in QA/dev prose does not belong here.
FILLER_PATTERNS = [
    r"\btodo\b", r"\btbd\b", r"\bfixme\b",
    r"\blorem ipsum\b",
    r"\byour code here\b",
    r"\bverify it works\b", r"\bverify functionality\b",
    r"\bcoming soon\b",
    r"\bsame as above\b",
    # Angle-bracket placeholders ONLY when the contents read like an
    # instruction. A bare `<[a-z ]+>` pattern matches <script> and <label for>,
    # which are real content in XSS and accessibility test cases.
    r"<(?:insert|your|add|todo|replace|fill|placeholder|value|name)\b[^>]{0,28}>",
    r"\{\{[a-z_ ]{2,30}\}\}",           # {{placeholder}}
    r"\[(?:insert|add|todo|placeholder)[^\]]{0,30}\]",
    r"example\.com/(?:path|foo|bar)",
]

# A field whose ENTIRE value is one of these is empty padding, even though the
# same token inside a sentence would be fine.
FILLER_EXACT = {"n/a", "na", "tbd", "todo", "...", "-", "none", "?", "xxx"}
