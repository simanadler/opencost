"""Phase 2: OpenCost attribution correctness (plan Section 8, O-01 to O-21).

Every case sends traffic with known identity and token counts, closes an explicit window
only after two Prometheus scrapes followed the last request (plan Sections 5.1 and 5.3),
and asserts OpenCost's API rows and llm_dimension_* gauges against the engine oracle and
OpenCost's own model-level rates. The harness never recomputes a rate from first principles
(plan Section 5.2).

Aggregation lists always name model_name (and, where a case filters by namespace,
namespace too): OpenCost clears any property that is not part of the aggregation key, so
`aggregate=tenant_id` alone would lose the model. The plan's `aggregate=tenant_id` is run
as `model_name,tenant_id`, the same join with the model kept visible.

Cases that need a configuration variant the lab cannot switch at run time report BLOCKED
with the exact change they need, following the Phase 1 treatment of G-12 and G-13. O-20 and
O-21 (V3 and V7) DO switch the configuration: Lab.attribution_config() patches the
attribution ConfigMap, restarts OpenCost and restores the original in `finally`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from loadgen import build_prompt

from ..lab import Lab, Outcome
from ..metrics import parse
from ..opencost import (BASES, UNIDENTIFIED, UNMETERED, ApiError, Entry, close, expected_cost,
                        total_cost, window)
from ..registry import case
from .. import results as R

MODEL = "gemma-4-31B"
MN = "model_name,namespace"                       # model-level, no dimensions
MT = "model_name,tenant_id"
MNT = "model_name,namespace,tenant_id"            # tenant rows with the namespace kept
MTW = "model_name,tenant_id,workload_id"
MTWU = "model_name,tenant_id,workload_id,user_id"
MU = "model_name,user_id"


# --- window primitive -------------------------------------------------------------------------

@dataclass
class Win:
    """One traffic case's explicit window and everything observed inside it."""
    lab: Lab
    t0: float
    t1: float
    outs: list[Outcome] = field(default_factory=list)
    scrape: dict = field(default_factory=dict)

    def w(self) -> str:
        return window(self.t0, self.t1)

    def rows(self, basis: str = "allocation", aggregate: str | None = MN, *,
             admin: bool = True, flt: str | None = None) -> list[Entry]:
        return self.lab.opencost().total(self.w(), basis=basis, aggregate=aggregate,
                                          admin=admin, filter=flt)

    @property
    def e_in(self) -> float:
        return sum(o.e_in for o in self.outs)

    @property
    def e_out(self) -> float:
        return sum(o.e_out for o in self.outs)

    def evidence(self, **extra) -> dict:
        ev = {"window": self.w(), "E_in": self.e_in, "E_out": self.e_out,
              "client_in": sum(o.client_in for o in self.outs),
              "client_out": sum(o.client_out for o in self.outs),
              "statuses": [s for o in self.outs for s in o.statuses],
              "oracle_warnings": [w for o in self.outs for w in o.oracle.warnings],
              "scrape_wait": self.scrape}
        ev.update(extra)
        return ev


def run_win(lab: Lab, fn, *, scrapes: int = 2) -> Win:
    """Run `fn` (one burst or a list of bursts) inside an explicit window.

    The window closes only after `scrapes` Prometheus scrapes followed the last request, so
    the `last_over_time @ end` read inside OpenCost sees the final counter values (the whole
    point of the exact-delta strategy; an earlier `increase()`-based version reported 2.00x).
    """
    t0 = lab.mark()
    outs = fn()
    outs = [outs] if isinstance(outs, Outcome) else list(outs)
    after = lab.mark()
    scrape = lab.wait_scrapes(after, scrapes)
    t1 = lab.mark()
    return Win(lab, t0, t1, outs, scrape)


def _no_traffic(cid: str, title: str, win: Win):
    statuses = [s for o in win.outs for s in o.statuses]
    if not any(s == 200 for s in statuses):
        return R.blocked(cid, title, f"no request succeeded (statuses {statuses})", **win.evidence())
    if not win.scrape.get("ok"):
        return R.blocked(cid, title, f"Prometheus did not scrape twice after the last request: "
                        f"{win.scrape}", **win.evidence())
    return None


# --- row helpers ------------------------------------------------------------------------------

def find(rows: list[Entry], **props) -> list[Entry]:
    return [e for e in rows if all(getattr(e, k) == v for k, v in props.items())]


def one(rows: list[Entry], **props) -> Entry | None:
    r = find(rows, **props)
    return r[0] if r else None


def tok(rows: list[Entry], **props) -> tuple[float, float]:
    """(prompt, generation) tokens summed over the matching rows."""
    p = sum(e.prompt for e in find(rows, **props))
    g = sum(e.generation for e in find(rows, **props))
    return p, g


def model_rows(win: Win, basis: str, model: str = MODEL) -> list[Entry]:
    return [e for e in win.rows(basis, MN) if e.model == model]


def dim_rows(win: Win, basis: str, aggregate: str, model: str = MODEL) -> list[Entry]:
    return [e for e in win.rows(basis, aggregate) if e.model == model]


def gauge_samples(lab: Lab, name: str, match: dict | None = None, *, seconds: float = 100,
                  every: float = 5, traffic=None):
    """Read llm_dimension_* gauge samples while light traffic keeps the last collection cycle
    non-empty.

    The gauges are reset on every collection cycle (INFERENCE_COLLECTION_INTERVAL, 60s in the
    lab) and only set when that cycle saw engine tokens, so a gauge read after the traffic
    has stopped is usually absent. The helper keeps a little traffic flowing (outside any
    case window) while polling. `traffic` defaults to gateway-only requests; O-04 passes a
    generator that also bypasses the gateway, O-05 one that uses only a sentinel key.
    Returns every sample of `name` whose labels contain `match` (all samples when match is
    None), or [] when none appeared.
    """
    stop = threading.Event()

    def send():
        gw = lab.gw()
        prompt = build_prompt(80)
        while not stop.is_set():
            try:
                if traffic is not None:
                    traffic(gw, prompt)
                else:
                    gw.chat(lab.key("rag"), gw.chat_body(MODEL, prompt, 4, "gauge", False))
            except OSError:
                pass
            stop.wait(8)

    t = threading.Thread(target=send, daemon=True)
    t.start()
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            try:
                text = lab.opencost().metrics_text()
                found = [s for s in parse(text, {name})
                         if match is None or all(s.label(k) == v for k, v in match.items())]
                if found:
                    return found
            except (OSError, ApiError):
                pass
            time.sleep(every)
    finally:
        stop.set()
        t.join(timeout=30)
    return []


def gauge(lab: Lab, name: str, match: dict, *, seconds: float = 100, every: float = 5,
          traffic=None):
    """The value of one llm_dimension_* gauge sample, or None when it never appeared."""
    found = gauge_samples(lab, name, match, seconds=seconds, every=every, traffic=traffic)
    return found[0].value if found else None


def _fmt(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"


# --- reconciliation (O-03, reused by other cases) ---------------------------------------------

def reconcile(win: Win, basis: str, model: str = MODEL) -> dict:
    """Plan O-03, both release gates, plus the token reconciliation.

    gate1: aggregate=model_name equals the model-level total (model_name,namespace).
    gate2: the sum over tenant rows, remainder included, equals the same total.
    tokens: attributed + __unmetered__ equals the engine oracle for the window.
    """
    mrows = model_rows(win, basis, model)
    model_total = total_cost(mrows)
    by_model = total_cost(dim_rows(win, basis, "model_name", model))
    trows = dim_rows(win, basis, MT, model)
    tenant_total = total_cost(trows)
    attributed = (sum(e.prompt for e in trows if e.tenant != UNMETERED),
                  sum(e.generation for e in trows if e.tenant != UNMETERED))
    unmetered = tok(trows, tenant=UNMETERED)
    return {"basis": basis, "model_total": model_total, "aggregate_model_name": by_model,
            "tenant_rows_total": tenant_total,
            "gate1": close(by_model, model_total), "gate2": close(tenant_total, model_total),
            "tokens_attributed": attributed, "tokens_unmetered": unmetered,
            "tokens_engine": (win.e_in, win.e_out),
            "tokens_reconcile": (close(attributed[0] + unmetered[0], win.e_in, abs_=1) and
                                 close(attributed[1] + unmetered[1], win.e_out, abs_=1)),
            "per_namespace": {e.namespace: {"cost": e.cost, "tokens": (e.prompt, e.generation)}
                              for e in mrows}}


# ============================================================================================= O-01
@case("O-01", "Token coverage within tolerance of 1.0, input and output separately", phase=2)
def o01(lab: Lab):
    def bursts():
        # spread over ~90s so at least one full collection cycle sees traffic (the coverage
        # gauges reset every cycle and are only set when the cycle saw engine tokens)
        outs = []
        for k, u in (("rag", "alice"), ("batch", "bob"), ("chat", "erin")):
            outs.append(lab.traffic(MODEL, k, n=2, prompt_tokens=150, out_tokens=8, user=u))
            time.sleep(15)
        return outs

    win = run_win(lab, bursts)
    b = _no_traffic("O-01", "Token coverage", win)
    if b:
        return b
    ns = lab.cfg.engine_ns_a
    gauges = {d: gauge(lab, "llm_dimension_token_coverage",
                       {"model_name": MODEL, "namespace": ns, "direction": d})
              for d in ("input", "output")}
    rec = reconcile(win, "allocation")
    attributed, engine = rec["tokens_attributed"], rec["tokens_engine"]
    api_cov = {d: (attributed[i] / engine[i]) if engine[i] else None
               for i, d in enumerate(("input", "output"))}
    ev = {**win.evidence(), "gauge_coverage": gauges, "api_coverage": api_cov,
          "tolerance": [lab.cfg.coverage_under, lab.cfg.coverage_over]}
    bad = {d: v for d, v in {**gauges, **{f"api_{k}": v for k, v in api_cov.items()}}.items()
           if v is None or not lab.cfg.coverage_under <= v <= lab.cfg.coverage_over}
    if bad:
        return R.failed("O-01", "Token coverage",
                        f"coverage outside [{lab.cfg.coverage_under}, {lab.cfg.coverage_over}]: "
                        f"{bad}", **ev)
    return R.passed("O-01", "Token coverage",
                    f"gauge in={gauges['input']:.3f} out={gauges['output']:.3f}, "
                    f"api in={api_cov['input']:.3f} out={api_cov['output']:.3f}", **ev)


# ============================================================================================= O-02
@case("O-02", "Join arithmetic: row cost equals in x inRate + out x outRate, both bases", phase=2)
def o02(lab: Lab):
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=3, prompt_tokens=200, out_tokens=16,
                                           user="alice"))
    b = _no_traffic("O-02", "Join arithmetic", win)
    if b:
        return b
    problems = []
    detail = {}
    for basis in BASES:
        mrow = one(model_rows(win, basis), namespace=lab.cfg.engine_ns_a)
        trow = one(dim_rows(win, basis, MT), tenant="acme-corp/engineering")
        if mrow is None or trow is None:
            problems.append(f"{basis}: missing rows (model {mrow is not None}, "
                            f"tenant {trow is not None})")
            continue
        want = expected_cost(trow.prompt, trow.generation, mrow.in_rate, mrow.out_rate)
        detail[basis] = {"row_cost": trow.cost, "expected_from_model_rates": want,
                         "row_rates": (trow.in_rate, trow.out_rate),
                         "model_rates": (mrow.in_rate, mrow.out_rate),
                         "tokens": (trow.prompt, trow.generation)}
        if not close(trow.cost, want):
            problems.append(f"{basis}: cost {trow.cost:.6f} != expected {want:.6f}")
        if not (close(trow.in_rate, mrow.in_rate, rel=1e-6, abs_=1e-9) and
                close(trow.out_rate, mrow.out_rate, rel=1e-6, abs_=1e-9)):
            problems.append(f"{basis}: row rates ({trow.in_rate}, {trow.out_rate}) differ from "
                            f"model rates ({mrow.in_rate}, {mrow.out_rate})")
        if (trow.prompt, trow.generation) != (win.e_in, win.e_out):
            problems.append(f"{basis}: tokens ({_fmt(trow.prompt)}, {_fmt(trow.generation)}) != "
                            f"engine ({_fmt(win.e_in)}, {_fmt(win.e_out)})")
    ev = {**win.evidence(), "per_basis": detail}
    if problems:
        return R.failed("O-02", "Join arithmetic", "; ".join(problems), **ev)
    return R.passed("O-02", "Join arithmetic",
                    "both bases: cost equals tokens x model-level rates, tokens equal the engine "
                    "oracle exactly", **ev)


# ============================================================================================= O-03
@case("O-03", "Release gates: tenant rows plus remainder equal the model total, both bases", phase=2)
def o03(lab: Lab):
    win = run_win(lab, lambda: [
        lab.traffic(MODEL, k, n=2, prompt_tokens=120, out_tokens=8, user=u)
        for k, u in (("rag", "alice"), ("batch", "bob"), ("engineering-rag", "dave"),
                     ("chat", "erin"), ("noteam-key", None))
    ])
    b = _no_traffic("O-03", "Release gates", win)
    if b:
        return b
    recs = {basis: reconcile(win, basis) for basis in BASES}
    problems = [f"{basis}: gate1 {rec['aggregate_model_name']:.6f} vs model "
                f"{rec['model_total']:.6f}; gate2 {rec['tenant_rows_total']:.6f}; tokens "
                f"{rec['tokens_reconcile']}"
                for basis, rec in recs.items()
                if not (rec["gate1"] and rec["gate2"] and rec["tokens_reconcile"])]
    ev = {**win.evidence(), "reconciliation": recs}
    if problems:
        return R.failed("O-03", "Release gates", "fails the suite: " + "; ".join(problems), **ev)
    return R.passed("O-03", "Release gates",
                    "; ".join(f"{b}: total ${recs[b]['model_total']:.4f}, gates hold, tokens "
                              f"{_fmt(recs[b]['tokens_attributed'][0])}+"
                              f"{_fmt(recs[b]['tokens_unmetered'][0])}="
                              f"{_fmt(recs[b]['tokens_engine'][0])}" for b in BASES), **ev)


# ============================================================================================= O-04
@case("O-04", "Bypass traffic lands in __unmetered__, never with a tenant", phase=2)
def o04(lab: Lab):
    def bursts():
        gw_traffic = lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice")
        # a new identity's first burst (G-17): OpenCost's exact delta starts the series at 0,
        # so it is attributed, unlike a Prometheus increase() over a single-sample series
        key = lab.new_identity(f"o04-{lab.cfg.run_id}", f"o04-key-{lab.cfg.run_id}")
        new_out = lab.traffic(MODEL, key, n=1, prompt_tokens=90, out_tokens=6, user="burst")
        # direct-to-engine requests: counted by the engine oracle, invisible to LiteLLM
        bypass = lab.observe(MODEL, lambda: [lab.engine_chat(MODEL, build_prompt(130), 8)
                                             for _ in range(2)])
        return [gw_traffic, new_out, bypass]

    win = run_win(lab, bursts)
    b = _no_traffic("O-04", "Bypass and remainder", win)
    if b:
        return b
    trows = dim_rows(win, "allocation", MT)
    unmetered = one(trows, tenant=UNMETERED)
    rag = one(trows, tenant="acme-corp/engineering")
    new_row = one(trows, tenant=f"o04-{lab.cfg.run_id}")
    # bypass = engine delta minus the gateway-counted bursts (rag and the new identity)
    gateway_e = (win.outs[0].e_in + win.outs[1].e_in, win.outs[0].e_out + win.outs[1].e_out)
    bypass_e = (win.e_in - gateway_e[0], win.e_out - gateway_e[1])
    cov = gauge(lab, "llm_dimension_token_coverage",
                {"model_name": MODEL, "namespace": lab.cfg.engine_ns_a, "direction": "input"},
                traffic=lambda gw, prompt: (gw.chat(lab.key("rag"),
                                                    gw.chat_body(MODEL, prompt, 4, "g", False)),
                                            lab.engine_chat(MODEL, prompt, 4)))
    ev = {**win.evidence(),
          "unmetered": None if unmetered is None else
          {"tokens": (unmetered.prompt, unmetered.generation), "cost": unmetered.cost},
          "rag_row": None if rag is None else (rag.prompt, rag.generation),
          "new_identity_row": None if new_row is None else (new_row.prompt, new_row.generation),
          "bypass_engine_delta": bypass_e, "coverage_gauge_under_mixed_traffic": cov}
    problems = []
    if unmetered is None:
        problems.append("no __unmetered__ row")
    elif not (close(unmetered.prompt, bypass_e[0], abs_=1) and
              close(unmetered.generation, bypass_e[1], abs_=1)):
        problems.append(f"unmetered tokens ({_fmt(unmetered.prompt)}, {_fmt(unmetered.generation)})"
                        f" != bypass engine delta ({_fmt(bypass_e[0])}, {_fmt(bypass_e[1])})")
    if rag is None or (rag.prompt, rag.generation) != (win.outs[0].e_in, win.outs[0].e_out):
        problems.append("rag row does not equal its own engine delta (bypass leaked into a tenant)")
    if new_row is None or not close(new_row.prompt + new_row.generation,
                                    win.outs[1].e_in + win.outs[1].e_out, abs_=1):
        problems.append("new-identity burst not attributed (G-17 consequence)")
    if cov is not None and cov >= 1.0:
        problems.append(f"coverage gauge {cov:.3f} did not drop below 1 under bypass traffic")
    if problems:
        return R.failed("O-04", "Bypass and remainder", "; ".join(problems), **ev)
    return R.passed("O-04", "Bypass and remainder",
                    f"bypass ({_fmt(bypass_e[0])}/{_fmt(bypass_e[1])}) in __unmetered__, rag and "
                    f"new identity attributed exactly, coverage gauge {cov}", **ev)


# ============================================================================================= O-05
@case("O-05", "Identity coverage: sentinel traffic under __unidentified__", phase=2)
def o05(lab: Lab):
    team = f"o05-{lab.cfg.run_id}"
    noalias_key = lab.new_identity(team, None)

    def bursts():
        return [lab.traffic(MODEL, "rag", n=1, prompt_tokens=100, out_tokens=6, user="alice"),
                lab.traffic(MODEL, "noteam-key", n=2, prompt_tokens=110, out_tokens=6),
                lab.traffic(MODEL, noalias_key, n=1, prompt_tokens=90, out_tokens=6, user="zed")]

    win = run_win(lab, bursts)
    b = _no_traffic("O-05", "Identity coverage", win)
    if b:
        return b
    trows = dim_rows(win, "allocation", MTW)
    noteam = one(trows, tenant=UNIDENTIFIED)
    noalias = one(trows, tenant=team)
    rag = one(trows, tenant="acme-corp/engineering", workload="rag")
    ident_gauge = gauge(lab, "llm_dimension_identity_coverage",
                        {"model_name": MODEL, "namespace": lab.cfg.engine_ns_a},
                        traffic=lambda gw, prompt: gw.chat(lab.key("noteam-key"),
                                                           gw.chat_body(MODEL, prompt, 4, None,
                                                                        False)))
    ev = {**win.evidence(),
          "noteam_row": None if noteam is None else
          (noteam.tenant, noteam.workload, noteam.prompt, noteam.generation),
          "noalias_row": None if noalias is None else
          (noalias.tenant, noalias.workload, noalias.prompt, noalias.generation),
          "identity_coverage_gauge_sentinel_only": ident_gauge}
    problems = []
    if noteam is None or noteam.workload != UNIDENTIFIED:
        problems.append(f"noteam-key row {noteam and (noteam.tenant, noteam.workload)}: expected "
                        f"tenant and workload {UNIDENTIFIED}")
    elif not close(noteam.prompt, win.outs[1].e_in, abs_=1):
        problems.append("noteam-key tokens differ from its engine delta")
    if noalias is None or noalias.workload != UNIDENTIFIED or noalias.tenant != team:
        problems.append(f"team-without-alias row {noalias and (noalias.tenant, noalias.workload)}: "
                        "expected real tenant with __unidentified__ workload")
    if rag is None:
        problems.append("no real-identity row")
    if ident_gauge is None or ident_gauge > 1e-9:
        problems.append(f"sentinel-only identity coverage gauge {ident_gauge}: expected 0, not 100")
    if problems:
        return R.failed("O-05", "Identity coverage", "; ".join(problems), **ev)
    return R.passed("O-05", "Identity coverage",
                    f"noteam-key and no-alias keys under {UNIDENTIFIED}, real tenant kept, "
                    f"sentinel-only coverage gauge {ident_gauge}", **ev)


# ============================================================================================= O-06
@case("O-06", "Workload dimension: rows split by the key alias", phase=2)
def o06(lab: Lab):
    def bursts():
        forged = {"x-workload-id": "forged-workload"}
        return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice"),
                lab.traffic(MODEL, "batch", n=2, prompt_tokens=180, out_tokens=12, user="bob"),
                lab.traffic(MODEL, "chat", n=1, prompt_tokens=90, out_tokens=6, user="erin",
                            headers=forged)]

    win = run_win(lab, bursts)
    b = _no_traffic("O-06", "Workload dimension", win)
    if b:
        return b
    wrows = dim_rows(win, "allocation", MTW)
    rag, batch, chat = one(wrows, workload="rag"), one(wrows, workload="batch"), \
        one(wrows, workload="chat")
    tenant_row = one(dim_rows(win, "allocation", MT), tenant="acme-corp/engineering")
    rec = reconcile(win, "allocation")
    ev = {**win.evidence(), "rows": [(e.tenant, e.workload, e.prompt, e.generation) for e in wrows],
          "reconciliation": rec}
    problems = []
    for name, row, out in (("rag", rag, win.outs[0]), ("batch", batch, win.outs[1])):
        if row is None or (row.prompt, row.generation) != (out.e_in, out.e_out):
            problems.append(f"{name} row {row and (row.prompt, row.generation)} != engine "
                            f"({out.e_in}, {out.e_out})")
    if chat is None or (chat.prompt, chat.generation) != (win.outs[2].e_in, win.outs[2].e_out):
        problems.append("forged-header request not attributed to its own key's workload")
    if tenant_row is None or not close(tenant_row.prompt, win.outs[0].e_in + win.outs[1].e_in,
                                       abs_=1):
        problems.append("tenant row does not equal the sum of its workloads")
    if any(e.workload == "forged-workload" for e in wrows):
        problems.append("a forged-workload row exists")
    if not (rec["gate2"] and rec["tokens_reconcile"]):
        problems.append("workload rows plus remainder do not equal the model total")
    if problems:
        return R.failed("O-06", "Workload dimension", "; ".join(problems), **ev)
    return R.passed("O-06", "Workload dimension",
                    "rag and batch split by key alias with exact tokens, tenant row sums them, "
                    "forged workload header changed nothing", **ev)


# ============================================================================================= O-07
@case("O-07", "User dimension: rows split by user, no-user rows have an empty user_id", phase=2)
def o07(lab: Lab):
    users = ("alice", "bob", "carol", None)

    def bursts():
        return [lab.traffic(MODEL, "rag", n=1, prompt_tokens=100 + 20 * i, out_tokens=6, user=u)
                for i, u in enumerate(users)]

    win = run_win(lab, bursts)
    b = _no_traffic("O-07", "User dimension", win)
    if b:
        return b
    urows = dim_rows(win, "allocation", MTWU)
    wrow = one(dim_rows(win, "allocation", MTW), tenant="acme-corp/engineering", workload="rag")
    ev = {**win.evidence(), "rows": [(e.tenant, e.workload, e.user, e.prompt, e.generation)
                                     for e in urows]}
    problems = []
    for i, u in enumerate(users):
        row = one(urows, tenant="acme-corp/engineering", workload="rag", user=u or "")
        if row is None or (row.prompt, row.generation) != (win.outs[i].e_in, win.outs[i].e_out):
            problems.append(f"user {u!r} row {row and (row.prompt, row.generation)} != engine "
                            f"({win.outs[i].e_in}, {win.outs[i].e_out})")
    if any(e.user == "None" for e in urows):
        problems.append("a row carries the literal user 'None' instead of an empty user_id")
    if wrow is None or not close(wrow.prompt, win.e_in, abs_=1):
        problems.append("workload row does not equal the sum of its users")
    if problems:
        return R.failed("O-07", "User dimension", "; ".join(problems), **ev)
    return R.passed("O-07", "User dimension",
                    "alice/bob/carol split exactly, no-user requests under an empty user_id "
                    "(sentinel normalized), workload row sums them", **ev)


# ============================================================================================= O-08
def _both_ns_traffic(lab: Lab):
    """Traffic to the same served model in both namespaces (O-08c, O-09)."""
    return [lab.traffic(MODEL, "rag", n=2, prompt_tokens=140, out_tokens=8, user="alice"),
            lab.traffic("gemma-4-31B-b", "chat", n=2, prompt_tokens=100, out_tokens=8, user="erin")]


@case("O-08", "Namespace resolution without a deployment_namespace label", phase=2)
def o08(lab: Lab):
    ns_a, ns_b = lab.cfg.engine_ns_a, lab.cfg.engine_ns_b
    results = []

    # (a) model active in one namespace: exact join via engine token share
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=130, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("O-08a", "Single active namespace", win)
    if b:
        results.append(b)
    else:
        trows = dim_rows(win, "allocation", MNT)
        a_row = one(trows, tenant="acme-corp/engineering", namespace=ns_a)
        b_tokens = tok(trows, namespace=ns_b)
        ev = {**win.evidence(), "rows": [(e.tenant, e.namespace, e.prompt) for e in trows]}
        problems = []
        if a_row is None or not close(a_row.prompt, win.e_in, abs_=1):
            problems.append(f"llm-a row {a_row and (a_row.namespace, a_row.prompt)}")
        if b_tokens[0] or b_tokens[1]:
            problems.append(f"tokens leaked to the idle namespace {ns_b}: {b_tokens}")
        if problems:
            results.append(R.failed("O-08a", "Single active namespace", "; ".join(problems), **ev))
        else:
            results.append(R.passed(
                "O-08a", "Single active namespace",
                f"exact join to {ns_a} (engine token share 1.0; approximate={a_row.approximate} "
                "is set conservatively by the split path, the tokens are exact)", **ev))

    # (b) V2 namespace map: needs an OpenCost environment change
    results.append(R.blocked(
        "O-08b", "Namespace map (V2)",
        "needs INFERENCE_ATTRIBUTION_NAMESPACE_MAP set on the OpenCost deployment (plan variant "
        "V2) and a rollout; not switched by the harness. Set it to map gemma-4-31B to "
        f"{ns_a}, restart OpenCost, re-run the O-08c traffic and assert the join is exact per map"))

    # (c) model active in both namespaces: proportional split by engine token share
    win = run_win(lab, lambda: _both_ns_traffic(lab))
    b = _no_traffic("O-08c", "Two namespaces, no map", win)
    if b:
        results.append(b)
        return results
    problems = []
    detail = {}
    trows = dim_rows(win, "allocation", MNT)
    for ns, out in ((ns_a, win.outs[0]), (ns_b, win.outs[1])):
        rows_ns = [e for e in trows if e.namespace == ns]
        tenant_tok = (sum(e.prompt for e in rows_ns if e.tenant != UNMETERED),
                      sum(e.generation for e in rows_ns if e.tenant != UNMETERED))
        mrow = one(model_rows(win, "allocation"), namespace=ns)
        detail[ns] = {"tenant_tokens": tenant_tok, "engine": (out.e_in, out.e_out),
                      "approximate": [e.approximate for e in rows_ns],
                      "model_cost": mrow is not None and mrow.cost,
                      "rows_cost": total_cost(rows_ns)}
        if not (close(tenant_tok[0], out.e_in, abs_=1) and close(tenant_tok[1], out.e_out, abs_=1)):
            problems.append(f"{ns}: attributed {tenant_tok} != engine ({out.e_in}, {out.e_out})")
        if rows_ns and not all(e.approximate for e in rows_ns):
            problems.append(f"{ns}: rows not marked approximate")
        if mrow is not None and not close(total_cost(rows_ns), mrow.cost):
            problems.append(f"{ns}: tenant rows + remainder {total_cost(rows_ns):.6f} != model "
                            f"cost {mrow.cost:.6f}")
    if any(e.namespace == lab.cfg.litellm_ns for e in trows):
        problems.append("rows joined to the LiteLLM scrape-target namespace")
    ev = {**win.evidence(), "per_namespace": detail}
    if problems:
        results.append(R.failed("O-08c", "Two namespaces, no map", "; ".join(problems), **ev))
    else:
        results.append(R.passed("O-08c", "Two namespaces, no map",
                                "proportional split by engine token share, each namespace exact, "
                                "rows marked approximate, reconciliation holds per namespace", **ev))
    return results


# ============================================================================================= O-09
@case("O-09", "Same model in two namespaces, end to end (#4163)", phase=2)
def o09(lab: Lab):
    win = run_win(lab, lambda: _both_ns_traffic(lab))
    b = _no_traffic("O-09", "Same model, two namespaces", win)
    if b:
        return b
    ns_a, ns_b = lab.cfg.engine_ns_a, lab.cfg.engine_ns_b
    problems = []
    detail = {}
    for basis in BASES:
        mrows = {e.namespace: e for e in model_rows(win, basis)}
        for ns in (ns_a, ns_b):
            cost = mrows[ns].cost if ns in mrows else 0.0
            trows = [e for e in dim_rows(win, basis, MNT) if e.namespace == ns]
            detail[f"{basis}/{ns}"] = {"model_cost": cost, "dimension_rows": len(trows),
                                       "dimension_cost": total_cost(trows)}
            if cost <= 0:
                problems.append(f"{basis} {ns}: model cost {cost} (missing #4163?)")
            if not trows:
                problems.append(f"{basis} {ns}: no dimension rows joined")
            elif not close(total_cost(trows), cost):
                problems.append(f"{basis} {ns}: dimension rows {total_cost(trows):.6f} != model "
                                f"cost {cost:.6f}")
    ev = {**win.evidence(), "detail": detail}
    if problems:
        return R.failed("O-09", "Same model, two namespaces", "; ".join(problems), **ev)
    return R.passed("O-09", "Same model, two namespaces",
                    "both namespaces carry non-zero model cost with dimension rows joined and "
                    "reconciled per namespace, both bases", **ev)


# ============================================================================================= O-10
@case("O-10", "Over-attribution and double counting", phase=2)
def o10(lab: Lab):
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("O-10", "Over-attribution", win)
    if b:
        return b
    rec = reconcile(win, "allocation")
    logs = lab.oc_logs()
    warns = [ln for ln in logs.splitlines()
             if "double-counting" in ln or "unmatched attribution" in ln]
    unm = tok(dim_rows(win, "allocation", MT), tenant=UNMETERED)
    ev = {**win.evidence(), "reconciliation": rec, "matching_log_lines": warns[:5]}
    problems = []
    if warns:
        problems.append(f"OpenCost logged {warns[:2]}")
    if unm[0] < -1e-9 or unm[1] < -1e-9:
        problems.append(f"negative remainder {unm}")
    if not (rec["gate2"] and rec["tokens_reconcile"]):
        problems.append("reconciliation failed")
    return [(R.failed if problems else R.passed)(
        "O-10/baseline", "Over-attribution (baseline, one producer)",
        "; ".join(problems) or "no double-counting or unmatched warnings, remainder non-negative, "
        "reconciliation holds", **ev),
        R.blocked("O-10/V6", "Second producer without scope (V6)",
                  "needs a second producers entry in attribution.yaml metering the same series "
                  "with no scope (plan variant V6): expect a per-cycle WARN naming both producers, "
                  "tenant rows scaled pro rata and marked approximate, remainder exactly zero; "
                  "then add scope and verify the warning clears. Not switched by the harness "
                  "because it must not contaminate the other cases' windows"),
        R.blocked("O-10/G-12", "Cached usage counted twice",
                  "needs LiteLLM response caching on (plan A9 keeps it off); run G-12 first")]


# ============================================================================================= O-11
@case("O-11", "Counter reset mid-window: attributed totals equal the engine oracle",
      disruptive=True, phase=2)
def o11(lab: Lab):
    results = [R.blocked(
        "O-11/engine", "Engine replica restart mid-window",
        "restarting a gemma pod deletes its emptyDir model cache, so the reload takes 20 to 60 "
        "minutes (install guide Step 1.3) and would starve every later case. The reset-aware "
        "engine delta IS in this build (per-series last_over_time end minus start, a reset counts "
        "as its new value), so a manual run is authoritative: scale the engine deployment down "
        "and up between two bursts in one window and assert attributed == engine delta")]
    if len(lab.gateways()) < 2:
        results.append(R.blocked("O-11/litellm", "LiteLLM replica restart mid-window",
                                 f"{len(lab.gateways())} LiteLLM replica(s) ready, need 2"))
        return results
    victim = sorted(lab.litellm_pods.forwards)[0]

    t0 = lab.mark()
    out1 = lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8, user="alice",
                       gw=lab.gw(0))
    # the victim's final counters must be in Prometheus BEFORE it dies, or the window-end read
    # sees only its pre-burst value and undercounts the burst
    m1 = lab.mark()
    lab.wait_scrapes(m1, 2)
    before = lab.gw(0).counters()
    extra = (sum(v for (n, _), v in before.items() if n.endswith("input_tokens_metric_total")),
             sum(v for (n, _), v in before.items() if n.endswith("output_tokens_metric_total")))
    lab.cluster.delete_pod(lab.cfg.litellm_ns, victim)
    lab.cluster.wait_pods(lab.cfg.litellm_ns, lab.cfg.litellm_selector, 2, 240)
    lab.litellm_pods.sync()
    out2 = lab.traffic(MODEL, "rag", n=2, prompt_tokens=150, out_tokens=10, user="alice")
    m2 = lab.mark()
    scrape = lab.wait_scrapes(m2, 2)
    t1 = lab.mark()
    win = Win(lab, t0, t1, [out1, out2], scrape)
    b = _no_traffic("O-11/litellm", "LiteLLM replica restart mid-window", win)
    if b:
        results.append(b)
        return results
    trow = one(dim_rows(win, "allocation", MT), tenant="acme-corp/engineering")
    want = (win.e_in, win.e_out)
    ev = {**win.evidence(), "deleted_pod": victim, "victim_final_counters": extra,
          "row": None if trow is None else (trow.prompt, trow.generation, trow.cost)}
    # the dangerous case from the plan: start 100, reset to 0, end 300 must read 300, not 200
    if trow is None or not (close(trow.prompt, want[0], abs_=1) and
                            close(trow.generation, want[1], abs_=1)):
        results.append(R.failed("O-11/litellm", "LiteLLM replica restart mid-window",
                                f"attributed {trow and (trow.prompt, trow.generation)} != engine "
                                f"{want} (a reset hidden behind a growing counter was mis-read)",
                                **ev))
    else:
        results.append(R.passed("O-11/litellm", "LiteLLM replica restart mid-window",
                                f"attributed {_fmt(trow.prompt)}/{_fmt(trow.generation)} equals "
                                f"the engine oracle after a mid-window replica restart", **ev))
    return results


# ============================================================================================= O-12
@case("O-12", "Window exactness: each window contains only its own team", phase=2)
def o12(lab: Lab):
    t0 = lab.mark()
    out1 = lab.traffic(MODEL, "rag", n=2, prompt_tokens=110, out_tokens=6, user="alice")
    m1 = lab.mark()
    lab.wait_scrapes(m1, 2)
    t1_1 = lab.mark()
    t0_2 = lab.mark()
    out2 = lab.traffic(MODEL, "chat", n=2, prompt_tokens=130, out_tokens=6, user="erin")
    m2 = lab.mark()
    scrape = lab.wait_scrapes(m2, 2)
    t1_2 = lab.mark()
    win1, win2 = Win(lab, t0, t1_1, [out1], scrape), Win(lab, t0_2, t1_2, [out2], scrape)
    b = _no_traffic("O-12", "Window exactness", win2)
    if b:
        return b
    rows1, rows2 = dim_rows(win1, "allocation", MT), dim_rows(win2, "allocation", MT)
    rag1, chat1 = tok(rows1, tenant="acme-corp/engineering"), tok(rows1, tenant="globex")
    rag2, chat2 = tok(rows2, tenant="acme-corp/engineering"), tok(rows2, tenant="globex")
    ev = {"window1": win1.w(), "window2": win2.w(),
          "window1_rows": [(e.tenant, e.prompt) for e in rows1],
          "window2_rows": [(e.tenant, e.prompt) for e in rows2],
          "E": {"rag": (out1.e_in, out1.e_out), "chat": (out2.e_in, out2.e_out)}}
    problems = []
    if rag1 != (out1.e_in, out1.e_out) or chat1 != (0.0, 0.0):
        problems.append(f"window 1: rag {rag1}, chat {chat1} (expected rag only "
                        f"({out1.e_in}, {out1.e_out}))")
    if chat2 != (out2.e_in, out2.e_out) or rag2 != (0.0, 0.0):
        problems.append(f"window 2: rag {rag2}, chat {chat2} (expected chat only "
                        f"({out2.e_in}, {out2.e_out}))")
    if problems:
        return R.failed("O-12", "Window exactness",
                        "the preceding window is charged: " + "; ".join(problems), **ev)
    return R.passed("O-12", "Window exactness",
                    "each window contains only its own team's tokens, both directions", **ev)


# ============================================================================================= O-13
@case("O-13", "SaaS exclusion by scope (V0)", phase=2)
def o13(lab: Lab):
    def bursts():
        return [lab.traffic(MODEL, "rag", n=1, prompt_tokens=110, out_tokens=6, user="alice"),
                lab.traffic("saas-stub", "chat", n=2, prompt_tokens=50, out_tokens=0, user="erin",
                            oracle_aliases=())]

    win = run_win(lab, bursts)
    b = _no_traffic("O-13", "SaaS exclusion", win)
    if b:
        return b
    trows, mrows = dim_rows(win, "allocation", MT), model_rows(win, "allocation")
    saas_rows = [e for e in trows + mrows if "saas" in e.model]
    unmatched = gauge(lab, "llm_dimension_unmatched_keys", {"producer": "litellm-gateway"})
    logs = lab.oc_logs()
    ev = {**win.evidence(), "saas_rows": len(saas_rows), "unmatched_gauge": unmatched,
          "unmatched_log_lines": [ln for ln in logs.splitlines() if "unmatched" in ln][:3],
          "gemma_rows": [(e.tenant, e.prompt) for e in trows]}
    problems = []
    if saas_rows:
        problems.append(f"{len(saas_rows)} saas-stub rows reached the API (scope did not drop them)")
    if unmatched is not None and unmatched > 0:
        problems.append(f"unmatched keys gauge {unmatched} (SaaS series counted unmatched)")
    if "unmatched" in logs:
        problems.append("unmatched-attribution warnings in the OpenCost log")
    if not any(e.tenant == "acme-corp/engineering" for e in trows):
        problems.append("gemma traffic not attributed while SaaS was excluded")
    return [(R.failed if problems else R.passed)(
        "O-13/V0", "SaaS exclusion (V0, servingModels)",
        "; ".join(problems) or "saas-stub series dropped before the join, not counted unmatched, "
        "no health-check error, gemma attribution unaffected", **ev),
        R.blocked("O-13/V4", "No scope (V4)",
                  "remove scope from the producer mapping in attribution.yaml and restart "
                  "OpenCost: the SaaS series must be counted unmatched and the alert condition "
                  "must hold. Not switched by the harness: it would fail every other case"),
        R.blocked("O-13/V5", "excludeServingModels (V5)",
                  "replace servingModels with excludeServingModels in attribution.yaml and "
                  "restart OpenCost: same expectations as V0. Not switched by the harness")]


# ============================================================================================= O-14
@case("O-14", "SGLang decode exclusion [B]", phase=2)
def o14(_: Lab):
    return R.blocked(
        "O-14", "SGLang decode exclusion [B]",
        "the lab runs one non-disaggregated SGLang worker (install guide Step 1.4), so there is "
        "no engine_type=decode series to exclude. Needs a prefill/decode disaggregated SGLang "
        "behind LiteLLM; then coverage must read ~1.0, not ~0.5, with the reference excluding "
        'engine_type="decode" for prompt tokens and "prefill" for completion tokens. '
        "Cross-engine disaggregation additionally needs INFERENCE_ATTRIBUTION_REFERENCE_ENGINE")


# ============================================================================================= O-15
@case("O-15", "Model name canonicalization", phase=2)
def o15(lab: Lab):
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("O-15", "Model name canonicalization", win)
    if b:
        return b
    trow = one(dim_rows(win, "allocation", MT), tenant="acme-corp/engineering")
    unmatched = gauge(lab, "llm_dimension_unmatched_keys", {"producer": "litellm-gateway"})
    ev = {**win.evidence(), "row": None if trow is None else (trow.model, trow.prompt),
          "unmatched_gauge": unmatched,
          "note": "the lab's LiteLLM model label equals the engine model_name exactly (P-04), so "
                  "the canonicalization path (org/model vs model) is not exercised here; it needs "
                  "an alias whose model label is provider-qualified"}
    if trow is None or not close(trow.prompt, win.e_in, abs_=1):
        return R.failed("O-15", "Model name canonicalization",
                        f"row {trow and (trow.model, trow.prompt)} did not join", **ev)
    if unmatched is not None and unmatched > 0:
        return R.failed("O-15", "Model name canonicalization",
                        f"unmatched keys gauge {unmatched}: the join missed", **ev)
    return R.passed("O-15", "Model name canonicalization",
                    "joined without unmatched keys; the qualified-name variant is not exercisable "
                    "in this lab (see note)", **ev)


# ============================================================================================= O-16
@case("O-16", "Idle policy and zero-token window", phase=2)
def o16(lab: Lab):
    # (a) with traffic: the allocation basis carries idle, the usage basis does not
    win = run_win(lab, lambda: lab.traffic(MODEL, "rag", n=2, prompt_tokens=120, out_tokens=8,
                                           user="alice"))
    b = _no_traffic("O-16a", "Idle carried on the allocation basis", win)
    results = []
    if b:
        results.append(b)
    else:
        alloc = one(dim_rows(win, "allocation", MT), tenant="acme-corp/engineering")
        usage = one(dim_rows(win, "usage", MT), tenant="acme-corp/engineering")
        m_alloc = one(model_rows(win, "allocation"), namespace=lab.cfg.engine_ns_a)
        m_usage = one(model_rows(win, "usage"), namespace=lab.cfg.engine_ns_a)
        ev = {**win.evidence(), "tenant_alloc": alloc is not None and alloc.cost,
              "tenant_usage": usage is not None and usage.cost,
              "model_alloc": m_alloc is not None and m_alloc.cost,
              "model_usage": m_usage is not None and m_usage.cost}
        if None in (alloc, usage, m_alloc, m_usage):
            results.append(R.failed("O-16a", "Idle carried on the allocation basis",
                                    "missing rows", **ev))
        elif m_alloc.cost <= m_usage.cost or alloc.cost < usage.cost - 1e-9:
            results.append(R.failed(
                "O-16a", "Idle carried on the allocation basis",
                f"model alloc ${m_alloc.cost:.4f} vs usage ${m_usage.cost:.4f}; tenant alloc "
                f"${alloc.cost:.4f} vs usage ${usage.cost:.4f}: idle is not allocation-only", **ev))
        else:
            results.append(R.passed(
                "O-16a", "Idle carried on the allocation basis",
                f"model alloc ${m_alloc.cost:.4f} > usage ${m_usage.cost:.4f}; the tenant's "
                "allocation row carries idle in proportion to tokens, the usage row does not",
                **ev))

    # (b) a window with no engine tokens: the whole model cost sits in __unmetered__
    quiet = lab.cfg.collection_interval_s + lab.cfg.scrape_interval_s + 15
    time.sleep(quiet)
    t1 = lab.mark()
    win0 = Win(lab, t1 - 60, t1, [])
    mrows = model_rows(win0, "allocation")
    trows = dim_rows(win0, "allocation", MNT)
    non_metered = [e for e in trows if e.tenant != UNMETERED]
    ev = {"window": win0.w(), "quiet_seconds": quiet,
          "model_rows": [(e.model, e.namespace, e.cost, e.prompt) for e in mrows],
          "dimension_rows": [(e.tenant, e.namespace, e.cost, e.prompt) for e in trows]}
    problems = []
    if not mrows:
        problems.append("no model rows in a zero-token window")
    if non_metered:
        problems.append(f"tenant rows in an idle window: "
                        f"{[(e.tenant, e.prompt) for e in non_metered]}")
    for m in [e for e in mrows if e.model == MODEL]:
        unm = [e for e in trows if e.namespace == m.namespace]
        if not unm:
            problems.append(f"{m.namespace}: no __unmetered__ row for cost ${m.cost:.4f}")
        elif not close(total_cost(unm), m.cost):
            problems.append(f"{m.namespace}: __unmetered__ ${total_cost(unm):.4f} != model "
                            f"${m.cost:.4f}")
    if problems:
        results.append(R.failed("O-16b", "Zero-token window", "; ".join(problems), **ev))
    else:
        results.append(R.passed("O-16b", "Zero-token window",
                                "the whole model cost sits in __unmetered__ (recorded as idle, "
                                "not bypass)", **ev))
    return results


# ============================================================================================= O-17
@case("O-17", "Cached tokens priced at full input rate [B]", phase=2)
def o17(lab: Lab):
    prefix = build_prompt(1000)

    def bursts():
        return [lab.traffic(MODEL, "rag", n=1, content=prefix, out_tokens=8, user="alice")
                for _ in range(3)]

    win = run_win(lab, bursts)
    b = _no_traffic("O-17", "Cached tokens at full input rate", win)
    if b:
        return b
    trow = one(dim_rows(win, "allocation", MT), tenant="acme-corp/engineering")
    mrow = one(model_rows(win, "allocation"), namespace=lab.cfg.engine_ns_a)
    cached = sum(o.oracle.cached for o in win.outs)
    full_cost = expected_cost(trow.prompt, trow.generation, mrow.in_rate, mrow.out_rate) \
        if trow is not None and mrow is not None else None
    rec = reconcile(win, "allocation")
    ev = {**win.evidence(), "cached_tokens_engine": cached,
          "row": None if trow is None else {"prompt": trow.prompt, "cost": trow.cost},
          "full_rate_cost": full_cost, "reconciliation": rec}
    problems = []
    if trow is None or not close(trow.prompt, win.e_in, abs_=1):
        problems.append(f"tenant prompt {trow and trow.prompt} != engine full count {win.e_in}")
    elif not close(trow.cost, full_cost):
        problems.append(f"cost {trow.cost:.6f} != full-input-rate cost {full_cost:.6f}")
    if not rec["gate2"]:
        problems.append("tenant rows plus remainder do not reconcile with the model cost")
    if problems:
        return R.failed("O-17", "Cached tokens at full input rate", "; ".join(problems), **ev)
    return R.passed("O-17", "Cached tokens at full input rate",
                    f"prompt {_fmt(trow.prompt)} (engine full count, cached {cached}) priced at "
                    "the full input rate; reconciliation holds. A future cache-read rate is a "
                    "deliberate change", **ev)


# ============================================================================================= O-18
@case("O-18", "Estimated token share is zero for this producer", phase=2)
def o18(lab: Lab):
    share = gauge(lab, "llm_dimension_estimated_token_share",
                  {"model_name": MODEL, "namespace": lab.cfg.engine_ns_a})
    ev = {"gauge": share,
          "note": "LiteLLM has no estimated series, so estimated tokens are indistinguishable "
                  "from exact on this path (G-05/G-08 gap traffic looks exact)"}
    if share is None:
        return R.failed("O-18", "Estimated token share",
                        "llm_dimension_estimated_token_share never exported under traffic", **ev)
    if abs(share) > 1e-9:
        return R.failed("O-18", "Estimated token share",
                        f"estimated share {share} for a producer with no estimated series", **ev)
    return R.passed("O-18", "Estimated token share", "0, as expected for LiteLLM", **ev)


# ============================================================================================= O-19
MATRIX_USERS = ("u1", "u2", "u3", None)


def _matrix_traffic(lab: Lab, prefix: str = "o19", reuse: dict | None = None):
    """3 teams x 2 workloads x (3 users + no user), one request per cell with a distinct size.

    With `reuse` (the keys dict stored by O-19) the same identities are used again, so the
    variant cases O-20 and O-21 re-run the same traffic under the same tenant labels and their
    per-tenant totals are comparable with O-19's. Returns (bursts, cells, keys)."""
    cells, bursts, keys = [], [], reuse or {}
    idx = 0
    for t in range(3):
        team = f"{prefix}-t{t}-{lab.cfg.run_id}"
        for w in range(2):
            workload = f"w{t}{w}"   # key aliases must be unique across ALL keys in LiteLLM
            key = keys.get((t, w)) or lab.new_identity(team, workload)
            keys[(t, w)] = key
            for u in MATRIX_USERS:
                out = lab.traffic(MODEL, key, n=1, prompt_tokens=60 + 5 * idx,
                                  out_tokens=4 + idx % 5, user=u, settle_s=4)
                cells.append((team, workload, u, out.e_in, out.e_out))
                bursts.append(out)
                idx += 1
    return bursts, cells, keys


def _matrix_expectations(cells):
    per_tw, per_t, per_u = {}, {}, {}
    for team, w, u, e_in, e_out in cells:
        for d, k in ((per_tw, (team, w)), (per_t, team), (per_u, u or "")):
            d[k] = (d.get(k, (0, 0))[0] + e_in, d.get(k, (0, 0))[1] + e_out)
    return per_tw, per_t, per_u


@case("O-19", "Three-dimension matrix: every cell exact, every aggregation consistent", phase=2)
def o19(lab: Lab):
    t0 = lab.mark()
    bursts, cells, keys = _matrix_traffic(lab)
    after = lab.mark()
    scrape = lab.wait_scrapes(after, 2)
    t1 = lab.mark()
    win = Win(lab, t0, t1, bursts, scrape)
    if not any(s == 200 for o in bursts for s in o.statuses):
        return R.blocked("O-19", "Three-dimension matrix",
                         f"no request succeeded ({[s for o in bursts for s in o.statuses]})",
                         **win.evidence())
    lab.store["o19"] = {"keys": keys, "per_t": _matrix_expectations(cells)[1],
                        "per_tw": _matrix_expectations(cells)[0]}
    per_tw, per_t, per_u = _matrix_expectations(cells)
    problems = []
    for basis in BASES:
        urows = dim_rows(win, basis, MTWU)
        by_cell = {(e.tenant, e.workload, e.user): e for e in urows}
        for team, w, u, e_in, e_out in cells:
            e = by_cell.get((team, w, u or ""))
            if e is None or not (close(e.prompt, e_in, abs_=1) and close(e.generation, e_out, abs_=1)):
                problems.append(f"{basis} cell {(team, w, u)}: "
                                f"{e and (e.prompt, e.generation)} != ({_fmt(e_in)}, {_fmt(e_out)})")
            elif not close(e.cost, expected_cost(e.prompt, e.generation, e.in_rate, e.out_rate)):
                problems.append(f"{basis} cell {(team, w, u)}: cost {e.cost:.6f} != tokens x rates")
        # (2) summing the matrix over any dimension equals aggregating on the rest
        for aggregate, expect, keyfn in ((MT, per_t, lambda e: e.tenant),
                                         (MTW, per_tw, lambda e: (e.tenant, e.workload)),
                                         (MU, per_u, lambda e: e.user)):
            got = {keyfn(e): (e.prompt, e.generation) for e in dim_rows(win, basis, aggregate)}
            for k, want in expect.items():
                if aggregate == MU and k == "":
                    continue    # empty user merges with __unmetered__ under MU; not a cell loss
                have = got.get(k, (0, 0))
                if not (close(have[0], want[0], abs_=1) and close(have[1], want[1], abs_=1)):
                    problems.append(f"{basis} aggregate {aggregate} {k}: {have} != "
                                    f"({want[0]}, {want[1]})")
        # (3) matrix total + __unmetered__ + __unidentified__ equals the model total
        mtotal = total_cost(model_rows(win, basis))
        dtotal = total_cost(dim_rows(win, basis, MT))
        if not close(dtotal, mtotal):
            problems.append(f"{basis}: dimension rows + remainder ${dtotal:.6f} != model total "
                            f"${mtotal:.6f}")
        if any(e.tenant == UNIDENTIFIED for e in dim_rows(win, basis, MT)):
            problems.append(f"{basis}: __unidentified__ rows in an all-real-identity window")
        # (4) row count: no cell overwritten (a "/"-joined key would merge cells)
        non_zero = [e for e in dim_rows(win, basis, MTWU) if e.prompt > 0 or e.generation > 0]
        if len(non_zero) != len(cells):
            problems.append(f"{basis}: {len(non_zero)} non-zero rows for {len(cells)} cells")
    ev = win.evidence()
    if problems:
        return R.failed("O-19", "Three-dimension matrix",
                        f"{len(problems)} problem(s): " + "; ".join(problems[:6]), **ev)
    return R.passed("O-19", "Three-dimension matrix",
                    f"{len(cells)} cells exact on both bases, tenant/workload/user aggregations "
                    "consistent, matrix + __unmetered__ equals the model total, no cell "
                    "overwritten", **ev)


# ============================================================================================= O-20, O-21
def _dimension_off_case(lab: Lab, cid: str, title: str, label: str, dimension: str,
                        keeps_workload: bool):
    """Shared body of O-20 (V3, user off) and O-21 (V7, workload off): re-run the O-19 traffic
    with one mapping label emptied, assert the dimension is empty everywhere and nothing else
    moved, then restore the configuration (Lab.attribution_config does it in `finally`)."""
    try:
        variant = Lab.set_mapping_label(lab.attribution_config_text(), label, "")
    except Exception as e:    # a config format change is a BLOCKED case, not a crash
        return R.blocked(cid, title, f"cannot build the variant config: {e}")
    if "o19" not in lab.store:
        return R.blocked(cid, title, "O-19 did not run in this session, so there is no baseline "
                        "to compare the per-tenant totals against; run with O-19 in the selection")
    # the aggregate must still NAME the switched-off dimension: a row that carried a value
    # would show up as a non-empty value, while clearing it from the list would hide the bug
    aggregate = MTW if keeps_workload else MTWU
    with lab.attribution_config(variant):
        t0 = lab.mark()
        bursts, cells, _ = _matrix_traffic(lab, cid.split("-")[0].lower(),
                                           reuse=lab.store["o19"]["keys"])
        after = lab.mark()
        scrape = lab.wait_scrapes(after, 2)
        t1 = lab.mark()
        win = Win(lab, t0, t1, bursts, scrape)
        if not any(s == 200 for o in bursts for s in o.statuses):
            return R.blocked(cid, title, "no request succeeded", **win.evidence())
        per_tw, per_t, _ = _matrix_expectations(cells)
        problems = []
        for basis in BASES:
            rows = dim_rows(win, basis, aggregate)
            if any(getattr(e, dimension) for e in rows):
                problems.append(f"{basis}: rows carry a {dimension}")
            got_t = {e.tenant: (e.prompt, e.generation) for e in dim_rows(win, basis, MT)}
            for team, want in per_t.items():
                have = got_t.get(team, (0, 0))
                if not (close(have[0], want[0], abs_=1) and close(have[1], want[1], abs_=1)):
                    problems.append(f"{basis} tenant {team}: {have} != ({want[0]}, {want[1]})")
            if not keeps_workload:
                continue     # workload off: there are no per-workload rows to compare
            got_tw = {(e.tenant, e.workload): (e.prompt, e.generation)
                      for e in dim_rows(win, basis, MTW)}
            for (team, w), want in per_tw.items():
                have = got_tw.get((team, w), (0, 0))
                if not (close(have[0], want[0], abs_=1) and close(have[1], want[1], abs_=1)):
                    problems.append(f"{basis} workload {(team, w)}: {have} != ({want[0]}, {want[1]})")
        # against O-19's own run: same traffic, same identities, so the totals must match
        for team, want in lab.store["o19"]["per_t"].items():
            for basis in BASES:
                row = one(dim_rows(win, basis, MT), tenant=team)
                have = (row.prompt, row.generation) if row else (0, 0)
                if not (close(have[0], want[0], abs_=1) and close(have[1], want[1], abs_=1)):
                    problems.append(f"{basis} tenant {team} differs from O-19: {have} != "
                                    f"({want[0]}, {want[1]})")
        ev = win.evidence()
        if problems:
            return R.failed(cid, title,
                            f"{len(problems)} problem(s): " + "; ".join(problems[:6]), **ev)
        return R.passed(cid, title,
                        f"{dimension} empty everywhere, per-tenant (and per-workload) tokens "
                        "equal both this run's oracle and O-19's totals: disabling the dimension "
                        "changed nothing else (config restored)", **ev)


@case("O-20", "User label off (V3): user_id empty, everything else unchanged", phase=2)
def o20(lab: Lab):
    return _dimension_off_case(lab, "O-20", "User label off (V3)", "userLabel", "user", True)


@case("O-21", "Workload label off (V7): tenant-only attribution", phase=2)
def o21(lab: Lab):
    return _dimension_off_case(lab, "O-21", "Workload label off (V7)", "workloadLabel",
                               "workload", False)
