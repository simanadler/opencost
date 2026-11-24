"""Command line runner: select cases, run them against the lab, write the report.

    python3 -m inference_it list
    python3 -m inference_it run                       # Phase 0 prerequisites + non-disruptive Phase 1
    python3 -m inference_it run --cases G-01 G-04
    python3 -m inference_it run --disruptive          # also G-09 and G-16b (restart pods)
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from . import cases  # noqa: F401  (registers the cases)
from . import results as R
from .config import Config
from .lab import Lab, LabNotReady
from .registry import CASES, Case, ordered


def select(ids: list[str] | None, disruptive: bool, phase0: bool) -> list[Case]:
    if ids:
        unknown = [i for i in ids if i not in CASES]
        if unknown:
            sys.exit(f"unknown case id(s): {', '.join(unknown)}. Use `list`.")
        chosen = [CASES[i] for i in ids]
    else:
        chosen = [c for c in CASES.values() if (phase0 or c.phase != 0)]
    return ordered([c for c in chosen if disruptive or not c.disruptive or ids and c.id in ids])


def run_case(lab: Lab, c: Case) -> list[R.CaseResult]:
    start = time.monotonic()
    try:
        out = c.fn(lab)
    except LabNotReady as e:
        out = R.blocked(c.id, c.title, str(e))
    except Exception as e:   # a harness or lab error is BLOCKED, never silently skipped
        out = R.blocked(c.id, c.title, f"{type(e).__name__}: {e}",
                        traceback=traceback.format_exc(limit=6))
    items = out if isinstance(out, list) else [out]
    for r in items:
        r.duration_s = round(time.monotonic() - start, 1)
    return items


def cmd_list(_: argparse.Namespace) -> int:
    for c in ordered(list(CASES.values())):
        flag = " [disruptive]" if c.disruptive else ""
        print(f"{c.id:8} phase {c.phase}{flag:14} {c.title}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config()
    if args.out:
        cfg.out_dir = Path(args.out)
    chosen = select(args.cases, args.disruptive, not args.skip_phase0)
    if not chosen:
        sys.exit("no cases selected")
    started = R.now()
    lab = Lab(cfg)
    try:
        lab.prepare(wait_s=args.wait)
    except LabNotReady as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        results = [R.blocked(c.id, c.title, str(e)) for c in chosen]
        _finish(cfg, lab, started, results)
        return 2
    results: list[R.CaseResult] = []
    try:
        for c in chosen:
            print(f"--> {c.id} {c.title}", flush=True)
            for r in run_case(lab, c):
                results.append(r)
                ref = f" ({r.gap_ref})" if r.gap_ref else ""
                print(f"    {r.id}: {r.state}{ref} {r.summary[:160]}", flush=True)
    finally:
        _finish(cfg, lab, started, results)
        lab.shutdown()
    return R.exit_code(results)


def _finish(cfg: Config, lab: Lab, started: str, results: list[R.CaseResult]) -> None:
    phases = [CASES[r.id.split("/")[0]].phase for r in results if r.id.split("/")[0] in CASES]
    top = max(phases, default=1)
    meta = {"run_id": cfg.run_id, "started": started, "finished": R.now(), "versions": lab.versions(),
            "namespaces": {"opencost": cfg.oc_ns, "litellm": cfg.litellm_ns,
                           "engines": [cfg.engine_ns_a, cfg.engine_ns_b]},
            "phase": top, "phase_title": f"Phase {top} run report"}
    base = cfg.out_dir / f"phase{top}-{cfg.run_id}"
    R.write_json(base.with_suffix(".json"), meta, results)
    R.write_markdown(base.with_suffix(".md"), meta, results)
    counts = {s: sum(1 for r in results if r.state == s) for s in R.STATES}
    print("\n" + ", ".join(f"{s} {n}" for s, n in counts.items() if n))
    print(f"report: {base.with_suffix('.md')}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inference_it", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list cases").set_defaults(func=cmd_list)
    r = sub.add_parser("run", help="run cases against the lab")
    r.add_argument("--cases", nargs="+",
                   help="case ids; default: Phase 0 prerequisites, Phases 1, 2 and 3")
    r.add_argument("--disruptive", action="store_true", help="also run cases that restart pods")
    r.add_argument("--skip-phase0", action="store_true", help="skip the P-xx prerequisite checks")
    r.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                   help="wait this long for engine and gateway pods to become ready")
    r.add_argument("--out", help="directory for the JSON and Markdown report")
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
