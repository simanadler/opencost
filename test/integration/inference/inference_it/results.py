"""Case results, the five states of plan Section 5.4, and report writers."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PASS, FAIL, XFAIL, BLOCKED, INFO = "PASS", "FAIL", "XFAIL-GAP", "BLOCKED", "INFO"
STATES = (PASS, FAIL, XFAIL, BLOCKED, INFO)


@dataclass
class CaseResult:
    id: str
    title: str
    state: str
    summary: str = ""
    evidence: dict = field(default_factory=dict)
    gap_ref: str = ""
    duration_s: float = 0.0


def passed(id, title, summary="", **evidence) -> CaseResult:
    return CaseResult(id, title, PASS, summary, evidence)


def failed(id, title, summary, **evidence) -> CaseResult:
    return CaseResult(id, title, FAIL, summary, evidence)


def blocked(id, title, reason, **evidence) -> CaseResult:
    return CaseResult(id, title, BLOCKED, reason, evidence)


def info(id, title, summary, **evidence) -> CaseResult:
    return CaseResult(id, title, INFO, summary, evidence)


def gap(id, title, summary, ref, holds: bool = True, **evidence) -> CaseResult:
    """A known contract gap. While it holds the case is XFAIL-GAP. When the behaviour is fixed
    upstream it becomes FAIL until the expectation is updated (plan Section 5.4)."""
    if holds:
        return CaseResult(id, title, XFAIL, summary, evidence, ref)
    return CaseResult(id, title, FAIL,
                      f"documented gap {ref} no longer reproduces; update the expectation. {summary}",
                      evidence, ref)


def probe(id, title, summary, ref="", gap_holds: bool = False, **evidence) -> CaseResult:
    """A case whose outcome is discovered: PASS when the contract is met, XFAIL-GAP otherwise."""
    if gap_holds:
        return CaseResult(id, title, XFAIL, summary, evidence, ref)
    return CaseResult(id, title, PASS, summary, evidence)


def exit_code(results: list[CaseResult]) -> int:
    return 1 if any(r.state in (FAIL, BLOCKED) for r in results) else 0


def write_json(path: Path, meta: dict, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "results": [asdict(r) for r in results]},
                               indent=2, default=str) + "\n", encoding="utf-8")


CHECKLIST = (  # requirements Section 10, plan Section 7.1
    (1, "Token counters carry tenant, workload, served model, deployment namespace, user",
     ("P-02", "P-08", "G-01")),
    (2, "Prompt tokens recorded for failures after prefill (F9)", ("G-05", "G-06")),
    (3, "Streaming without usage marked usage_captured=false", ("G-02", "G-08")),
    (4, "Missing identity recorded as unattributed, not defaulted", ("G-03",)),
    (5, "serving_model equals the engine model_name after rewrites", ("P-04", "G-04")),
)


def write_markdown(path: Path, meta: dict, results: list[CaseResult]) -> None:
    by_id = {}
    for r in results:
        by_id.setdefault(r.id.split("/")[0], []).append(r)
    lines = [f"# {meta.get('phase_title', 'Phase 1 run report')} ({meta.get('run_id')})", "",
             f"Started {meta.get('started')}, finished {meta.get('finished')}.", "",
             "## Versions and provenance", ""]
    for k, v in (meta.get("versions") or {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Results", "", "| ID | Case | State | Summary |", "|---|---|---|---|"]
    for r in results:
        ref = f" ({r.gap_ref})" if r.gap_ref else ""
        lines.append(f"| {r.id} | {r.title} | **{r.state}**{ref} | {r.summary.replace('|', '/')[:300]} |")
    counts = {s: sum(1 for r in results if r.state == s) for s in STATES}
    lines += ["", "Totals: " + ", ".join(f"{s} {n}" for s, n in counts.items()), "",
              "## Selection checklist outcome (requirements Section 10)", "",
              "| # | Item | Evidence | Outcome |", "|---|---|---|---|"]
    for n, item, ids in CHECKLIST:
        found = [r for i in ids for k, rs in by_id.items() if k == i for r in rs]
        outcome = ", ".join(f"{r.id}:{r.state}" for r in found) or "not run"
        lines.append(f"| {n} | {item} | {', '.join(ids)} | {outcome} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
