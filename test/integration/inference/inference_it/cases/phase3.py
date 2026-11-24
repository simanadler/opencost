"""Phase 3: API, metrics and authorization (plan Section 9, A-01 to A-08).

Builds on the Phase 2 window and row helpers (imported from .phase2, which holds the shared
Phase 2/3 machinery). Two cases switch the OpenCost deployment temporarily: A-03 toggles
INFERENCE_ATTRIBUTION_ENABLED off and A-04 toggles INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED
off, both through Lab.opencost_env(), which restores the original values in `finally`. The
queries those cases compare run against the SAME closed window, so the Prometheus data under
them does not change while the deployment restarts.
"""
from __future__ import annotations

import threading
import time

from ..lab import Lab
from ..opencost import BASES, UNIDENTIFIED, UNMETERED, ApiError, close, expected_cost, total_cost
from ..registry import case
from .. import results as R
from .phase2 import (MODEL, MN, MNT, MT, MTW, MTWU, Win, _fmt, _no_traffic, dim_rows, find,
                     gauge_samples, one, run_win, tok)

HOURLY_COST = "llm_dimension_hourly_cost"


def _signature(entries):
    """Comparable form of an entry list: key plus the additive fields."""
    return {e.key: (e.cost, e.prompt, e.generation, e.model, e.namespace) for e in entries}


def _signatures_match(a, b, rel=1e-9) -> list[str]:
    problems = []
    if set(a) != set(b):
        problems.append(f"row keys differ: only-before {sorted(set(a) - set(b))}, "
                        f"only-after {sorted(set(b) - set(a))}")
        return problems
    for k in a:
        for i, field in enumerate(("cost", "prompt", "generation")):
            if not close(a[k][i], b[k][i], rel=rel, abs_=1e-9):
                problems.append(f"{k}: {field} {a[k][i]} -> {b[k][i]}")
    return problems


# ============================================================================================= A-01
@case("A-01", "aggregate by tenant: every tenant plus __unmetered__ and __unidentified__",
      phase=3)
def a01(lab: Lab):
    def bursts():
        return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice"),
                lab.traffic(MODEL, "chat", n=2, prompt_tokens=110, out_tokens=6, user="erin"),
                lab.traffic(MODEL, "noteam-key", n=1, prompt_tokens=90, out_tokens=6)]

    win = run_win(lab, bursts)
    b = _no_traffic("A-01", "Aggregate by tenant", win)
    if b:
        return b
    problems = []
    detail = {}
    for aggregate, keeps_ns in ((MT, False), (MNT, True), (MTW, False)):
        rows = dim_rows(win, "allocation", aggregate)
        tenants = {e.tenant for e in rows}
        want = {"acme-corp/engineering", "globex", UNIDENTIFIED, UNMETERED}
        detail[aggregate] = sorted(tenants)
        if not want <= tenants:
            problems.append(f"{aggregate}: tenant set {sorted(tenants)} misses "
                            f"{sorted(want - tenants)}")
        if keeps_ns and {e.namespace for e in rows} - {lab.cfg.engine_ns_a, ""}:
            problems.append(f"{aggregate}: unexpected namespaces "
                            f"{sorted({e.namespace for e in rows})}")
        if not keeps_ns and any(e.namespace for e in rows):
            problems.append(f"{aggregate}: namespace not cleared from the key")
    # tokens per tenant equal the per-burst oracle
    trows = dim_rows(win, "allocation", MT)
    for tenant, out in (("acme-corp/engineering", win.outs[0]), ("globex", win.outs[1]),
                        (UNIDENTIFIED, win.outs[2])):
        p, g = tok(trows, tenant=tenant)
        if not (close(p, out.e_in, abs_=1) and close(g, out.e_out, abs_=1)):
            problems.append(f"tenant {tenant}: ({_fmt(p)}, {_fmt(g)}) != engine "
                            f"({out.e_in}, {out.e_out})")
    ev = {**win.evidence(), "tenant_sets": detail}
    if problems:
        return R.failed("A-01", "Aggregate by tenant", "; ".join(problems), **ev)
    return R.passed("A-01", "Aggregate by tenant",
                    "tenant rows plus __unmetered__ and __unidentified__ under model_name[,"
                    "namespace][,workload_id], tokens equal the oracle per tenant", **ev)


# ============================================================================================= A-02
@case("A-02", "Aggregation key collision: (acme-corp/engineering, rag) vs (acme-corp, engineering/rag)",
      phase=3)
def a02(lab: Lab):
    results = []
    for i, (u1, u2) in enumerate((("alice", "dave"), ("bob", "erin"))):
        cid = f"A-02/run-{i + 1}"

        def bursts(u1=u1, u2=u2):
            return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user=u1),
                    lab.traffic(MODEL, "engineering-rag", n=2, prompt_tokens=150, out_tokens=10,
                                user=u2)]

        win = run_win(lab, bursts)
        b = _no_traffic(cid, "Aggregation key collision", win)
        if b:
            results.append(b)
            continue
        wrows = dim_rows(win, "allocation", MTW)
        pair_a = one(wrows, tenant="acme-corp/engineering", workload="rag")
        pair_b = one(wrows, tenant="acme-corp", workload="engineering/rag")
        urows = dim_rows(win, "allocation", MTWU)
        problems = []
        if pair_a is None or pair_b is None:
            problems.append(f"rows {[(e.tenant, e.workload) for e in wrows]}: the collision "
                            "pairs merged or one is missing")
        else:
            for row, out in ((pair_a, win.outs[0]), (pair_b, win.outs[1])):
                if (row.prompt, row.generation) != (out.e_in, out.e_out):
                    problems.append(f"({row.tenant}, {row.workload}): "
                                    f"({row.prompt}, {row.generation}) != engine "
                                    f"({out.e_in}, {out.e_out}) (overwritten?)")
        if not (find(urows, tenant="acme-corp/engineering", workload="rag", user=u1) and
                find(urows, tenant="acme-corp", workload="engineering/rag", user=u2)):
            problems.append("the pairs do not stay separate per user under "
                            "model_name,tenant_id,workload_id,user_id")
        ev = {**win.evidence(), "rows": [(e.tenant, e.workload, e.user, e.prompt) for e in urows],
              "note": "OpenCost escapes '/' in key components, so a slash-joined key cannot "
                      "merge the pairs"}
        if problems:
            results.append(R.failed(cid, "Aggregation key collision", "; ".join(problems), **ev))
        else:
            results.append(R.passed(cid, "Aggregation key collision",
                                    f"pairs separate with users {u1}/{u2}, tokens exact, totals "
                                    "not overwritten", **ev))
    return results


# ============================================================================================= A-03
@case("A-03", "Non-attribution queries unchanged with attribution disabled", phase=3)
def a03(lab: Lab):
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=130, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("A-03", "Non-attribution queries unchanged", win)
    if b:
        return b
    oc = lab.opencost()
    queries = {"namespace": "namespace", "model_name": "model_name", "none": None}
    before = {name: _signature(oc.total(win.w(), aggregate=agg)) for name, agg in queries.items()}
    problems = []
    try:
        with lab.opencost_env(INFERENCE_ATTRIBUTION_ENABLED="false"):
            after = {name: _signature(oc.total(win.w(), aggregate=agg))
                     for name, agg in queries.items()}
            # with attribution disabled no dimension entries may appear
            disabled_rows = oc.total(win.w(), aggregate=MT)
            if any(e.tenant not in ("", UNMETERED) for e in disabled_rows):
                problems.append(f"dimension entries returned while attribution is disabled: "
                                f"{[(e.key, e.tenant) for e in disabled_rows][:3]}")
    except Exception as e:
        return R.blocked("A-03", "Non-attribution queries unchanged",
                         f"could not toggle INFERENCE_ATTRIBUTION_ENABLED: {e}", **win.evidence())
    for name in queries:
        problems += [f"{name}: {p}" for p in _signatures_match(before[name], after[name])]
    ev = {**win.evidence(), "rows_before": {k: len(v) for k, v in before.items()}}
    if problems:
        return R.failed("A-03", "Non-attribution queries unchanged", "; ".join(problems), **ev)
    return R.passed("A-03", "Non-attribution queries unchanged",
                    "aggregate=namespace, aggregate=model_name and the natural key return "
                    "identical rows with attribution on and off; no dimension entries", **ev)


# ============================================================================================= A-04
@case("A-04", "user_id authorization: restricted without flag or admin, merged when unauthorized",
      phase=3)
def a04(lab: Lab):
    def bursts():
        return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=110, out_tokens=6, user="alice"),
                lab.traffic(MODEL, "rag", n=1, prompt_tokens=130, out_tokens=6, user="bob"),
                lab.traffic(MODEL, "rag", n=1, prompt_tokens=90, out_tokens=6)]

    win = run_win(lab, bursts)
    b = _no_traffic("A-04", "user_id authorization", win)
    if b:
        return b
    oc = lab.opencost()
    problems = []
    detail = {}

    # (a) user_id aggregate or filter without admin: a validation error naming the restriction
    for label, kw in (("aggregate", {"aggregate": MTWU, "admin": False}),
                      ("filter", {"aggregate": MTW, "filter": 'user_id:"alice"', "admin": False})):
        try:
            oc.total(win.w(), **kw)
            problems.append(f"user_id {label} without admin returned 200, expected 400")
        except ApiError as e:
            detail[f"no_admin_{label}"] = f"HTTP {e.status}: {e.message[:120]}"
            if e.status != 400 or "INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED" not in str(e):
                problems.append(f"user_id {label} without admin: HTTP {e.status}, message does "
                                f"not name the restriction: {e.message[:120]}")

    # (b) filter-only from an unauthorized caller: merged across users, user_id stripped
    merged = None
    for flt in ('tenant_id:"acme-corp/engineering"', 'workload_id:"rag"'):
        try:
            rows = oc.total(win.w(), filter=flt, admin=False)
        except ApiError as e:
            problems.append(f"filter-only {flt} unauthorized: HTTP {e.status}")
            continue
        if flt.startswith("tenant_id"):
            merged = one(rows, tenant="acme-corp/engineering", workload="rag")
        if not rows:
            problems.append(f"filter-only {flt} unauthorized: no rows")
            continue
        if any(e.user for e in rows):
            problems.append(f"filter-only {flt} unauthorized: user_id not stripped")
        p, g = sum(e.prompt for e in rows), sum(e.generation for e in rows)
        if not (close(p, win.e_in, abs_=1) and close(g, win.e_out, abs_=1)):
            problems.append(f"filter-only {flt} unauthorized: tokens ({_fmt(p)}, {_fmt(g)}) != "
                            f"engine ({_fmt(win.e_in)}, {_fmt(win.e_out)})")

    # with the flag and the admin token: per-user rows that sum to the merged row
    urows = dim_rows(win, "allocation", MTWU)
    per_user = {e.user: e for e in urows if e.tenant == "acme-corp/engineering"}
    detail["per_user_rows"] = {u: (e.prompt, e.generation, e.cost) for u, e in per_user.items()}
    if "" not in per_user or "alice" not in per_user or "bob" not in per_user:
        problems.append(f"per-user rows {sorted(per_user)}: expected alice, bob and an empty user")
    elif merged is not None:
        summed = (sum(e.prompt for e in per_user.values()), sum(e.generation for e in per_user.values()),
                  sum(e.cost for e in per_user.values()))
        if not (close(summed[0], merged.prompt, abs_=1) and close(summed[1], merged.generation, abs_=1)
                and close(summed[2], merged.cost, rel=1e-6)):
            problems.append(f"per-user rows sum {summed} != merged row "
                            f"({merged.prompt}, {merged.generation}, {merged.cost})")

    # tenant_id and workload_id queries are unrestricted
    for agg in (MT, MTW):
        try:
            oc.total(win.w(), aggregate=agg, admin=False)
        except ApiError as e:
            problems.append(f"{agg} without admin: HTTP {e.status} (should be unrestricted)")

    # (c) without the flag: user_id is refused even for the admin
    try:
        with lab.opencost_env(INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED="false"):
            try:
                oc.total(win.w(), aggregate=MTWU, admin=True)
                problems.append("user_id aggregate with flag off returned 200, expected 400")
            except ApiError as e:
                detail["flag_off_admin"] = f"HTTP {e.status}: {e.message[:120]}"
                if e.status != 400:
                    problems.append(f"user_id with flag off: HTTP {e.status}, expected 400")
    except Exception as e:
        problems.append(f"could not toggle the flag: {e}")

    ev = {**win.evidence(), "detail": detail}
    if problems:
        return R.failed("A-04", "user_id authorization", "; ".join(problems), **ev)
    return R.passed("A-04", "user_id authorization",
                    "user_id needs the flag and admin; filter-only unauthorized callers get "
                    "rows merged across users with user_id stripped; per-user rows sum to the "
                    "merged row; tenant and workload queries are unrestricted", **ev)


# ============================================================================================= A-05
@case("A-05", "llm_dimension_hourly_cost labels, empty user_id, and V3", phase=3)
def a05(lab: Lab):
    # background traffic with and without a body user, so both a populated and an empty
    # user_id label appear in the same cycle
    def traffic(gw, prompt):
        gw.chat(lab.key("rag"), gw.chat_body(MODEL, prompt, 4, "gauge", False))
        gw.chat(lab.key("rag"), gw.chat_body(MODEL, prompt, 4, None, False))

    labels_want = {"model_name", "namespace", "cost_basis", "tenant_id", "workload_id", "user_id"}
    samples = gauge_samples(lab, HOURLY_COST, {"model_name": MODEL}, traffic=traffic)
    if not samples:
        return R.failed("A-05", "Metric exposure", f"{HOURLY_COST} never exported under traffic")
    problems = []
    label_sets = [set(dict(s.labels)) for s in samples]
    if any(ls != labels_want for ls in label_sets):
        problems.append(f"label sets {sorted(map(sorted, label_sets))} != {sorted(labels_want)}")
    users = {s.label("user_id") for s in samples}
    if "gauge" not in users:
        problems.append(f"no populated user_id label (values seen: {sorted(users)})")
    if "" not in users:
        problems.append(f"no empty user_id label value (values seen: {sorted(users)})")

    # under V3 (user label off) every user_id is empty and the totals are unchanged
    def totals(sample_list):
        out = {}
        for s in sample_list:
            k = (s.label("cost_basis"), s.label("tenant_id"), s.label("workload_id"))
            out[k] = out.get(k, 0.0) + s.value
        return out

    baseline = totals(samples)
    try:
        variant = Lab.set_mapping_label(lab.attribution_config_text(), "userLabel", "")
    except Exception as e:
        return R.failed("A-05", "Metric exposure",
                        f"label check {'failed; ' if problems else 'passed; '}cannot build the V3 "
                        f"config: {e}", label_values=sorted(users), label_sets=sorted(map(sorted, label_sets)))
    with lab.attribution_config(variant):
        v3_samples = gauge_samples(lab, HOURLY_COST, {"model_name": MODEL}, traffic=traffic)
    v3 = totals(v3_samples)
    for k, v in baseline.items():
        if k not in v3:
            problems.append(f"V3: row {k} disappeared")
        elif not close(v3[k], v, rel=0.05):
            problems.append(f"V3: {k} total {v3[k]:.4f} != baseline {v:.4f}")
    if any(s.label("user_id") for s in v3_samples):
        problems.append("V3: user_id not empty on every row")

    # per-user cost is readable by anyone with Prometheus query access: record how the lab
    # limits that (the only path to the lab Prometheus is a loopback port-forward behind oc)
    # JSON reports need string keys; totals() above keys by (cost_basis, tenant, workload) tuples
    ev = {"label_values_user_id": sorted(users),
          "baseline_totals": {"/".join(k): v for k, v in baseline.items()},
          "v3_totals": {"/".join(k): v for k, v in v3.items()},
          "prometheus_access": "the lab Prometheus has no exposed route; it is reachable only "
                               "through `oc port-forward` (loopback), which requires cluster "
                               "privileges. Anyone with Prometheus query access can read "
                               "per-user cost (requirements 4.2)"}
    if problems:
        return R.failed("A-05", "Metric exposure", "; ".join(problems), **ev)
    return R.passed("A-05", "Metric exposure",
                    "documented labels with populated and empty user_id; under V3 user_id is "
                    "empty everywhere and the per-tenant/workload totals are unchanged", **ev)


# ============================================================================================= A-06
@case("A-06", "Timeseries: steps sum to /total, reconciliation holds per step", phase=3)
def a06(lab: Lab):
    win = run_win(lab, lambda: [
        lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice"),
        lab.traffic(MODEL, "chat", n=2, prompt_tokens=110, out_tokens=6, user="erin")])
    b = _no_traffic("A-06", "Timeseries", win)
    if b:
        return b
    oc = lab.opencost()
    problems = []
    detail = {"steps": 0}
    for basis in BASES:
        steps = oc.timeseries(win.w(), accumulate="hour", basis=basis, aggregate=MT)
        model_steps = oc.timeseries(win.w(), accumulate="hour", basis=basis, aggregate=MN)
        detail["steps"] = len(steps)
        if not steps:
            problems.append(f"{basis}: no steps returned")
            continue
        # dim_rows() keeps only MODEL's rows, so the steps must be cut to the same scope:
        # the lab's other engine deployments (the second gemma namespace, the idle sglang
        # llama engine) carry allocation-basis infrastructure cost with no traffic and
        # would otherwise inflate only the timeseries side of the comparison
        detail.setdefault("step_models", sorted({e.model for s in steps for e in s}))
        ts_total = total_cost([e for s in steps for e in s if e.model == MODEL])
        api_total = total_cost(dim_rows(win, basis, MT))
        if not close(ts_total, api_total, rel=1e-6):
            problems.append(f"{basis}: sum over steps ${ts_total:.6f} != /total ${api_total:.6f}")
        for i, (step, mstep) in enumerate(zip(steps, model_steps)):
            step_cost, model_cost = total_cost(step), total_cost(mstep)
            if not close(step_cost, model_cost, rel=1e-6):
                problems.append(f"{basis} step {i}: tenant rows ${step_cost:.6f} != model rows "
                                f"${model_cost:.6f}")
    detail["note"] = ("a window under one hour yields a single step (accumulate=hour); "
                      "multi-step parity needs a window over an hour")
    ev = {**win.evidence(), **detail}
    if problems:
        return R.failed("A-06", "Timeseries", "; ".join(problems), **ev)
    return R.passed("A-06", "Timeseries",
                    f"{detail['steps']} step(s) on both bases: steps sum to /total and tenant "
                    "rows reconcile with model rows per step", **ev)


# ============================================================================================= A-07
@case("A-07", "Computation path parity and the shared-collector race", phase=3)
def a07(lab: Lab):
    # --- parity: the background runner's exported cost for its cycle vs an API window that
    # contains the same burst. Quiet before and after the burst keeps every counter static, so
    # any window containing the whole burst yields the same token delta; the remaining cost
    # difference is the model-cost sliver from the window edges (bounded by the poll interval
    # against the 60s cycle), hence the 5% tolerance.
    parity_problems = []
    parity_ev = {}
    for attempt in range(3):
        time.sleep(lab.cfg.collection_interval_s + lab.cfg.scrape_interval_s + 5)  # quiet
        t0 = lab.mark()
        burst = lab.traffic(MODEL, "rag", n=1, prompt_tokens=120, out_tokens=8, user="alice",
                            settle_s=4)
        m = lab.mark()
        lab.wait_scrapes(m, 2)
        # catch the first collection cycle whose window contains the burst (no background
        # traffic here: later cycles see no engine tokens and reset the gauge to absent)
        sample = None
        deadline = time.monotonic() + lab.cfg.collection_interval_s + 30
        while time.monotonic() < deadline and sample is None:
            try:
                found = [s for s in parse_metrics(lab, HOURLY_COST)
                         if s.label("tenant_id") == "acme-corp/engineering"]
                if found:
                    sample = found[0]
                else:
                    time.sleep(2)
            except (OSError, ApiError):
                time.sleep(2)
        if sample is None:
            parity_problems = [f"{HOURLY_COST} never reflected the burst (attempt {attempt + 1})"]
            continue
        t = lab.mark()
        parity_win = Win(lab, t - 60, t, [burst])
        row = one(dim_rows(parity_win, "allocation", MT), tenant="acme-corp/engineering")
        if row is None:
            parity_problems = ["the API window missed the burst (no tenant row)"]
            continue
        if not (close(row.prompt, burst.e_in, abs_=1) and close(row.generation, burst.e_out, abs_=1)):
            parity_problems = [f"the API window does not contain the whole burst: "
                               f"({row.prompt}, {row.generation}) vs ({burst.e_in}, {burst.e_out})"]
            continue
        cycle_cost = sample.value / 60.0     # hourly cost normalized by the 60s cycle window
        parity_problems = []
        if not close(row.cost, cycle_cost, rel=0.05):
            parity_problems = [f"runner cycle cost ${cycle_cost:.6f} vs API window cost "
                               f"${row.cost:.6f} (outside the 5% window-edge tolerance)"]
        parity_ev = {"attempt": attempt + 1, "runner_hourly": sample.value,
                     "runner_cycle_cost": cycle_cost, "api_window_cost": row.cost,
                     "api_window": parity_win.w(), "api_tokens": (row.prompt, row.generation)}
        break

    # --- race: identical concurrent API queries under load must agree and must not fail
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("A-07", "Computation path parity", win)
    if b:
        return b
    stop = threading.Event()

    def load():
        from loadgen import build_prompt
        gw = lab.gw()
        prompt = build_prompt(80)
        while not stop.is_set():
            try:
                gw.chat(lab.key("batch"), gw.chat_body(MODEL, prompt, 4, "race", False))
            except OSError:
                pass
            stop.wait(6)

    lt = threading.Thread(target=load, daemon=True)
    lt.start()
    answers, errors = [], []
    try:
        def query():
            try:
                answers.append(_signature(lab.opencost().total(win.w(), aggregate=MT)))
            except Exception as e:   # a 500 or a corrupt answer is exactly the race
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=query) for _ in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=180)
    finally:
        stop.set()
        lt.join(timeout=30)
    race_problems = [f"query failed: {e}" for e in errors]
    if len(answers) != 6:
        race_problems.append(f"only {len(answers)}/6 queries completed")
    for i in range(1, len(answers)):
        race_problems += [f"query {i} disagrees with query 0: {p}"
                          for p in _signatures_match(answers[0], answers[i])]

    ev = {**win.evidence(), "parity": parity_ev, "concurrent_queries": len(answers),
          "query_errors": errors[:3]}
    problems = parity_problems + race_problems
    if problems:
        return R.failed("A-07", "Computation path parity", "; ".join(problems), **ev)
    return R.passed("A-07", "Computation path parity",
                    f"runner and API agree for the same burst (within the window-edge "
                    f"tolerance); 6 concurrent queries under load returned identical rows",
                    **ev)


def parse_metrics(lab: Lab, name: str):
    from ..metrics import parse
    return parse(lab.opencost().metrics_text(), {name})


# ============================================================================================= A-08
@case("A-08", "Response shape: empty dimensions omitted, approximate only where applied",
      phase=3)
def a08(lab: Lab):
    def bursts():
        return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice"),
                lab.traffic(MODEL, "rag", n=1, prompt_tokens=100, out_tokens=6, user="bob")]

    win = run_win(lab, bursts)
    b = _no_traffic("A-08", "Response shape", win)
    if b:
        return b
    oc = lab.opencost()
    problems = []

    # raw JSON: tenant_id/workload_id/user_id omitted when empty; approximate only on rows
    # produced by the proportional split or pro-rata scaling
    status, body = oc.raw("/inferenceCost/total",
                          {"window": win.w(), "costBasis": "allocation", "aggregate": MTWU},
                          admin=True)
    costs = (body.get("data") or {}).get("inferenceCosts") or {}
    unmetered = [d for d in costs.values() if (d.get("properties") or {}).get("tenant_id") == UNMETERED]
    user_rows = [d for d in costs.values()
                 if (d.get("properties") or {}).get("user_id") == "alice"]
    if not unmetered:
        problems.append("no __unmetered__ row in the raw response")
    else:
        props = unmetered[0].get("properties") or {}
        for field in ("workload_id", "user_id"):
            if field in props:
                problems.append(f"__unmetered__ row carries {field}={props[field]!r}; empty "
                                "dimensions must be omitted")
    if not user_rows:
        problems.append("no per-user row in the raw response")
    else:
        props = user_rows[0].get("properties") or {}
        for field in ("tenant_id", "workload_id", "user_id"):
            if field not in props:
                problems.append(f"populated {field} missing from the row properties")
    dim_approx = [bool((d.get("properties") or {}).get("approximate")) for d in costs.values()]
    if not all(dim_approx):
        problems.append("not every dimension row is marked approximate (this lab always splits "
                        "across two namespaces)")
    _, model_body = oc.raw("/inferenceCost/total",
                           {"window": win.w(), "costBasis": "allocation", "aggregate": MN},
                           admin=True)
    model_costs = (model_body.get("data") or {}).get("inferenceCosts") or {}
    if any((d.get("properties") or {}).get("approximate") for d in model_costs.values()):
        problems.append("model-level rows carry approximate")

    # a single-row tenant response: non-zero TotalTokens, rates consistent with its own cost
    merged = one(oc.total(win.w(), filter='tenant_id:"acme-corp/engineering"', admin=False),
                 tenant="acme-corp/engineering", workload="rag")
    if merged is None:
        problems.append("no merged tenant row for the filter-only query")
    elif merged.total_tokens <= 0:
        problems.append(f"merged row TotalTokens {merged.total_tokens}")
    elif not close(merged.cost, expected_cost(merged.prompt, merged.generation,
                                              merged.in_rate, merged.out_rate), rel=1e-6):
        problems.append(f"merged row cost {merged.cost:.6f} inconsistent with its own tokens and "
                        f"rates ({merged.prompt} x {merged.in_rate} + {merged.generation} x "
                        f"{merged.out_rate})")

    ev = {**win.evidence(),
          "note": "approximate after over-attribution scaling needs V6 (a second producer); "
                  "every dimension row here is approximate from the two-namespace split",
          "merged_row": None if merged is None else
          {"tokens": (merged.prompt, merged.generation), "cost": merged.cost,
           "rates": (merged.in_rate, merged.out_rate)}}
    if problems:
        return R.failed("A-08", "Response shape", "; ".join(problems), **ev)
    return R.passed("A-08", "Response shape",
                    "empty dimensions omitted, approximate only on split rows, single-row tenant "
                    "response has non-zero TotalTokens with self-consistent rates", **ev)
