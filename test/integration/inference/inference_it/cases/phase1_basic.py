"""Phase 1: gateway conformance cases that need no fault injection.

Assertions are on LiteLLM's counters against the engine oracle. OpenCost cost output is not used
(plan Section 7).
"""
from __future__ import annotations

from ..config import (LABEL_MODEL, LABEL_REQUESTED_MODEL, LABEL_TENANT, LABEL_USER, LABEL_WORKLOAD,
                      MATRIX, SENTINEL)
from ..lab import Lab, Outcome
from ..registry import case
from .. import results as R
from .common import exact, ident, outside, summarize, traffic_resilient

NO_SPANS = ("span assertions (gen_ai.usage.*, one span per request) were not run: the lab has no "
            "OTel exporter, so only the counters are checked")


def _no_traffic(id_, title, out: Outcome):
    """BLOCKED when the engine never served anything, so a missing delta is not read as a failure."""
    if not any(s == 200 for s in out.statuses):
        return R.blocked(id_, title, f"no request succeeded (statuses {out.statuses}); engine or "
                         "gateway unavailable", **summarize(out))
    return None


# --------------------------------------------------------------------------------------- G-01
@case("G-01", "Non-stream success with identity: exact token match")
def g01(lab: Lab):
    users = {"rag": "alice", "batch": "bob", "engineering-rag": "dave", "chat": "erin", "agent": "erin"}
    res = []
    for key, user in users.items():
        tenant, workload = MATRIX[key]
        out, notes = traffic_resilient(lab, "gemma-4-31B", key, n=3, prompt_tokens=200,
                                       out_tokens=16, user=user)
        blocked = _no_traffic(f"G-01/{key}", f"Exact match, key {key}", out)
        if blocked:
            res.append(blocked)
            continue
        ok, i, o = exact(out, **ident(tenant, workload, user, "gemma-4-31B"))
        leak = outside(out, **ident(tenant, workload))
        client_ok = out.client_in == out.e_in and out.client_out == out.e_out
        ev = {**summarize(out), "litellm_in": i, "litellm_out": o, "leaked_elsewhere": leak,
              "notes": notes + [NO_SPANS]}
        if ok and leak == 0 and client_ok:
            res.append(R.passed(f"G-01/{key}", f"Exact match, key {key}",
                                f"in {i:.0f}/{out.e_in:.0f}, out {o:.0f}/{out.e_out:.0f}", **ev))
        else:
            res.append(R.failed(f"G-01/{key}", f"Exact match, key {key}",
                                f"litellm {i:.0f}/{o:.0f} vs engine {out.e_in:.0f}/{out.e_out:.0f}, "
                                f"client {out.client_in}/{out.client_out}, leaked {leak}", **ev))
    return res


# --------------------------------------------------------------------------------------- G-02
@case("G-02", "Streaming success with usage in the final chunk")
def g02(lab: Lab):
    res = []
    for include_usage in (True, False):
        title = f"Streaming, client include_usage={include_usage}"
        out, notes = traffic_resilient(lab, "gemma-4-31B", "rag", n=3, prompt_tokens=200,
                                       out_tokens=16, user="alice", stream=True,
                                       include_usage=include_usage)
        cid = f"G-02/{'with' if include_usage else 'without'}-usage-option"
        blocked = _no_traffic(cid, title, out)
        if blocked:
            res.append(blocked)
            continue
        ok, i, o = exact(out, **ident("acme-corp/engineering", "rag", "alice", "gemma-4-31B"))
        usage_seen = [r.prompt_tokens is not None for r in out.results]
        ev = {**summarize(out), "litellm_in": i, "litellm_out": o,
              "client_received_usage_chunk": usage_seen,
              "usage_chunk_has_empty_choices": [r.usage_chunk_empty_choices for r in out.results],
              "litellm_injected_include_usage": (not include_usage) and all(usage_seen), "notes": notes}
        if ok:
            res.append(R.passed(cid, title, "stream counters equal the engine oracle exactly", **ev))
        else:
            res.append(R.failed(cid, title,
                                f"litellm {i:.0f}/{o:.0f} vs engine {out.e_in:.0f}/{out.e_out:.0f}", **ev))
    return res


# --------------------------------------------------------------------------------------- G-03
@case("G-03", "Missing identity appears under one documented sentinel")
def g03(lab: Lab):
    res = []
    # (a) no team, no key alias
    out = lab.traffic("gemma-4-31B", "noteam-key", n=2, prompt_tokens=100, out_tokens=8)
    b = _no_traffic("G-03a", "No team and no key alias", out)
    if b:
        res.append(b)
    else:
        vals = {(r.labels.get(LABEL_TENANT), r.labels.get(LABEL_WORKLOAD), r.labels.get(LABEL_USER))
                for r in out.rows}
        ok = vals == {(SENTINEL, SENTINEL, SENTINEL)}
        ev = {**summarize(out), "label_values": sorted(vals), "expected_sentinel": SENTINEL}
        res.append(R.passed("G-03a", "No team and no key alias", f"all three labels are {SENTINEL!r}", **ev)
                   if ok else R.failed("G-03a", "No team and no key alias",
                                       f"labels {sorted(vals)}; update SENTINEL or file a gap", **ev))
    # (b) team without key alias, harness-made identity
    key = lab.new_identity(f"g03b-{lab.cfg.run_id}", None)
    out = lab.traffic("gemma-4-31B", key, n=2, prompt_tokens=100, out_tokens=8, user="zed")
    b = _no_traffic("G-03b", "Team but no key alias", out)
    if b:
        res.append(b)
    else:
        vals = {(r.labels.get(LABEL_TENANT), r.labels.get(LABEL_WORKLOAD)) for r in out.rows}
        want = {(f"g03b-{lab.cfg.run_id}", SENTINEL)}
        res.append((R.passed if vals == want else R.failed)(
            "G-03b", "Team but no key alias",
            f"tenant kept, workload label {sorted({str(v[1]) for v in vals})}", **summarize(out),
            label_values=sorted(vals)))
    # (c) no body user
    out = lab.traffic("gemma-4-31B", "rag", n=2, prompt_tokens=100, out_tokens=8, user=None)
    b = _no_traffic("G-03c", "No body user", out)
    if b:
        res.append(b)
    else:
        users = {r.labels.get(LABEL_USER) for r in out.rows}
        teams = {r.labels.get(LABEL_TENANT) for r in out.rows}
        ok = users == {SENTINEL} and teams == {"acme-corp/engineering"}
        res.append((R.passed if ok else R.failed)(
            "G-03c", "No body user",
            f"end_user={sorted(users, key=str)}, tenant unchanged={sorted(teams, key=str)}",
            **summarize(out)))
    sentinels = {"tenant": SENTINEL, "workload": SENTINEL, "user": SENTINEL}
    res.append(R.info("G-03", "Pinned sentinels", "tenant, workload and user use the same sentinel, "
                      "so one OpenCost identitySentinel per producer is enough", **sentinels))
    return res


# --------------------------------------------------------------------------------------- G-04
@case("G-04", "Router rewrite: model label equals the engine model_name")
def g04(lab: Lab):
    out, notes = traffic_resilient(lab, "llama-chat", "chat", n=3, prompt_tokens=200, out_tokens=16,
                                   user="frank")
    b = _no_traffic("G-04", "Router rewrite", out)
    if b:
        return b
    pairs = sorted({(r.labels.get(LABEL_MODEL), r.labels.get(LABEL_REQUESTED_MODEL)) for r in out.rows})
    ok, i, o = exact(out, **ident("globex", "chat", "frank"))
    ev = {**summarize(out), "litellm_in": i, "litellm_out": o, "model_requested_pairs": pairs,
          "notes": notes + ["span half (gen_ai.request.model, BerriAI/litellm#51029) not run: no OTel "
                            "exporter; GAP on stable builds"]}
    model_ok = all(m == "llama" for m, _ in pairs) and all(rq == "llama-chat" for _, rq in pairs)
    if ok and model_ok:
        return R.passed("G-04", "Router rewrite", "model='llama', requested_model='llama-chat', "
                        "tokens equal the SGLang counters", **ev)
    return R.failed("G-04", "Router rewrite",
                    f"pairs {pairs}; tokens {i:.0f}/{o:.0f} vs engine {out.e_in:.0f}/{out.e_out:.0f}", **ev)


# --------------------------------------------------------------------------------------- G-07
@case("G-07", "Rejected before inference records nothing")
def g07(lab: Lab):
    gw = lab.gw()
    chat = lambda key, body: gw.chat(key, body)  # noqa: E731
    codes: dict = {}

    def action():
        good = gw.chat_body("gemma-4-31B", "hi", 4, "x", False)
        out = [chat("sk-invalid", good)]
        codes["invalid_key"] = out[-1].status
        out.append(chat(lab.key("chat"), {"model": "gemma-4-31B"}))
        codes["malformed"] = out[-1].status
        out.append(chat(lab.key("chat"), gw.chat_body("gemma-4-31B", "word " * 400000, 4, "x", False)))
        codes["oversize_context"] = out[-1].status
        out.append(chat(lab.key("chat"), gw.chat_body("no-such-model", "hi", 4, "x", False)))
        codes["unknown_model"] = out[-1].status
        return out

    out = lab.observe("gemma-4-31B", action, oracle_aliases=("gemma-4-31B", "llama-chat", "gemma-4-31B-b"))
    ev = {**summarize(out), "status_by_case": codes, "counter_rows": [(r.metric, r.labels, r.delta) for r in out.rows]}
    if out.e_in or out.e_out or out.rows:
        return R.failed("G-07", "Rejected before inference",
                        f"engine moved {out.e_in}/{out.e_out} or counters changed {len(out.rows)} rows", **ev)
    # rate limit: the allowed request is counted, everything rejected must not be
    lim = lab.traffic("gemma-4-31B", "limited", n=4, prompt_tokens=60, out_tokens=4, user="x")
    lim_ok, i, o = exact(lim, **ident("limited", "limited"))
    ev["rate_limit"] = {"statuses": lim.statuses, "attributed": (i, o), "engine": (lim.e_in, lim.e_out)}
    if not lim_ok:
        return R.failed("G-07", "Rejected before inference",
                        "rate-limited key: counters differ from engine", **ev)
    rejected = [s for s in lim.statuses if s in (429, 400, 401)]
    note = "" if rejected else " (limited key was never rejected: rpm/budget not enforced)"
    return R.passed("G-07", "Rejected before inference",
                    f"engine and counters unchanged for {codes}; rate limit statuses {lim.statuses}{note}", **ev)


# --------------------------------------------------------------------------------------- G-10
@case("G-10", "Forged identity: tenant and workload cannot be forged")
def g10(lab: Lab):
    res = []
    forged = {"x-tenant-id": "forged-tenant", "x-workload-id": "forged-workload",
              "x-litellm-team-id": "forged-team", "x-litellm-key-alias": "forged-alias"}
    meta = {"metadata": {"team_alias": "forged-tenant", "user_api_key_alias": "forged-workload",
                         "user_api_key_team_alias": "forged-tenant"}}
    out = lab.traffic("gemma-4-31B", "chat", n=2, prompt_tokens=100, out_tokens=8, user="erin",
                      headers=forged, extra=meta)
    b = _no_traffic("G-10a", "Forged headers and metadata", out)
    if b:
        res.append(b)
    else:
        _, text = lab.gw().metrics_text(True)
        series = sorted({(r.labels.get(LABEL_TENANT), r.labels.get(LABEL_WORKLOAD)) for r in out.rows})
        ok = series == [("globex", "chat")] and "forged" not in text
        res.append((R.passed if ok else R.failed)(
            "G-10a/c", "Forged headers and metadata",
            f"attribution followed the key: {series}; 'forged' in scrape: {'forged' in text}",
            **summarize(out)))
    out = lab.traffic("gemma-4-31B", "chat", n=1, prompt_tokens=100, out_tokens=8, user="alice")
    b = _no_traffic("G-10b", "Body user naming another person", out)
    if b:
        res.append(b)
    else:
        res.append(R.gap(
            "G-10b", "Body user is caller-asserted",
            "user 'alice' (a rag/acme user) was accepted under globex/chat: user_id is reported by the "
            "caller, not verified. Impact stays inside the caller's own tenant and workload",
            "by design; document in the report", holds=True,
            users=sorted({r.labels.get(LABEL_USER) for r in out.rows}, key=str),
            tenants=sorted({r.labels.get(LABEL_TENANT) for r in out.rows}, key=str)))
    return res


# --------------------------------------------------------------------------------------- G-11
@case("G-11", "Unauthorized metrics read and write")
def g11(lab: Lab):
    gw, cluster, cfg = lab.gw(), lab.cluster, lab.cfg
    anon, _ = gw.metrics_text(False)
    prom = lab.promq()
    writes = {p: prom.probe_write(p) for p in ("/api/v1/write", "/api/v1/otlp/v1/metrics")}
    push, push_idle = [], []
    for ns in (cfg.oc_ns, cfg.litellm_ns):
        for s in cluster.get_json("svc", "-n", ns).get("items", []):
            name = s["metadata"]["name"]
            if "pushgateway" not in name:
                continue
            eps = cluster.get_json("endpoints", name, "-n", ns).get("subsets") or []
            has_addr = any(sub.get("addresses") for sub in eps)
            (push if has_addr else push_idle).append(f"{ns}/{name}")
    netpol = {ns: len(cluster.get_json("networkpolicy", "-n", ns).get("items", []))
              for ns in (cfg.litellm_ns, cfg.oc_ns)}
    ev = {"litellm_anonymous_scrape": anon, "prometheus_write_probe": writes,
          "pushgateway_with_backing_pods": push, "pushgateway_service_without_pods": push_idle,
          "network_policies": netpol,
          "not_tested": "reachability from another namespace (needs an in-cluster client); the forward "
                        "bypasses NetworkPolicy"}
    accepted = [p for p, s in writes.items() if 200 <= s < 300]
    if accepted or push:
        return R.failed("G-11", "Unauthorized metrics write",
                        f"series can be injected: write endpoints accepting {accepted}, pushgateway {push}", **ev)
    if anon not in (401, 403):
        return R.failed("G-11", "Unauthorized metrics read", f"anonymous scrape returned {anon}", **ev)
    return R.passed("G-11", "Unauthorized metrics write/read",
                    f"anonymous scrape {anon}; no write endpoint accepted an unauthenticated POST "
                    f"({writes}); no pushgateway", **ev)


# --------------------------------------------------------------------------------------- G-12
@case("G-12", "Response-cache short circuit")
def g12(lab: Lab):
    return R.blocked("G-12", "Response-cache short circuit",
                     "needs LiteLLM response caching on (plan A9 keeps it off for the baseline). Set "
                     "litellm_settings.cache: true with an in-memory cache in litellm-config.yaml, roll "
                     "LiteLLM, then run this case (not yet automated)")


# --------------------------------------------------------------------------------------- G-13
@case("G-13", "Retries and fallbacks count once")
def g13(lab: Lab):
    return R.blocked("G-13", "Retries and fallbacks",
                     "the lab model list defines no fallback or second deployment per alias, so a failed "
                     "first backend cannot be provoked without a config change (add a fallbacks entry "
                     "for a deployment pointing at a dead api_base, then re-run)")


# --------------------------------------------------------------------------------------- G-14
@case("G-14", "Engine prefix-cache reuse")
def g14(lab: Lab):
    from loadgen import build_prompt
    long_prompt = build_prompt(1000)
    per, bad = [], False
    for _ in range(3):
        out = lab.traffic("gemma-4-31B", "rag", n=1, content=long_prompt, out_tokens=8, user="alice")
        b = _no_traffic("G-14", "Prefix-cache reuse", out)
        if b:
            return b
        ok, i, o = exact(out, **ident("acme-corp/engineering", "rag", "alice"))
        per.append({"E_in": out.e_in, "E_out": out.e_out, "litellm": (i, o),
                    "engine_cached": out.oracle.cached, "client_in": out.client_in})
        bad |= not ok
    cached_seen = any(p["engine_cached"] for p in per)
    ev = {"requests": per, "prefix_cache_hit_observed": cached_seen}
    if bad:
        return R.failed("G-14", "Prefix-cache reuse", "attributed tokens differ from the engine", **ev)
    return R.passed("G-14", "Prefix-cache reuse",
                    "attributed input equals the engine prompt count on every request"
                    + ("" if cached_seen else " (no cache hit observed; the engine may not export "
                       "prompt_tokens_cached_total or prefix caching is off)"), **ev)


# --------------------------------------------------------------------------------------- G-15
@case("G-15", "SaaS model through the same LiteLLM")
def g15(lab: Lab):
    out = lab.traffic("saas-stub", "chat", n=2, prompt_tokens=50, out_tokens=0, user="erin",
                      oracle_aliases=())
    b = _no_traffic("G-15", "SaaS model", out)
    if b:
        return b
    i, o = out.attributed(**ident("globex", "chat", "erin", "saas-stub"))
    ev = {**summarize(out), "litellm_in": i, "litellm_out": o}
    if (i, o) == (11.0 * len(out.results), 7.0 * len(out.results)):
        return R.passed("G-15", "SaaS model", "series emitted with the stub's usage (11/7 per request); "
                        "no engine counters exist; exclusion is OpenCost's job (O-13)", **ev)
    return R.failed("G-15", "SaaS model", f"expected 11/7 per request, got {i}/{o}", **ev)


# --------------------------------------------------------------------------------------- G-16
@case("G-16", "Two LiteLLM replicas: per-replica counters sum to the engine delta")
def g16(lab: Lab):
    gws = lab.gateways()
    if len(gws) < 2:
        return R.blocked("G-16", "Multi-replica", f"{len(gws)} LiteLLM replica(s) ready, need 2")
    key = lab.key("rag")
    from loadgen import build_prompt
    prompt = build_prompt(150)

    def action():
        out = []
        for n in range(6):
            g = gws[n % len(gws)]
            out.append(g.chat(key, g.chat_body("gemma-4-31B", prompt, 8, "alice", False)))
        return out

    out = lab.observe("gemma-4-31B", action, gw=gws)
    b = _no_traffic("G-16", "Multi-replica", out)
    if b:
        return b
    ok, i, o = exact(out, **ident("acme-corp/engineering", "rag", "alice"))
    ev = {**summarize(out), "summed_in": i, "summed_out": o, "replicas": [g.name for g in gws]}
    if ok:
        return R.passed("G-16", "Multi-replica", f"per-replica deltas sum to {i:.0f}/{o:.0f}, equal to the "
                        "engine delta. Restart behaviour is G-16b (--disruptive)", **ev)
    return R.failed("G-16", "Multi-replica", f"sum {i:.0f}/{o:.0f} vs engine {out.e_in:.0f}/{out.e_out:.0f}", **ev)


# --------------------------------------------------------------------------------------- G-17
@case("G-17", "New-identity burst: is the series exported at zero first?")
def g17(lab: Lab):
    team = f"g17-{lab.cfg.run_id}"
    key = lab.new_identity(team, f"g17-key-{lab.cfg.run_id}")
    gw = lab.gw()
    before = [k for k in gw.counters() if dict(k[1]).get(LABEL_TENANT) == team]
    out = lab.traffic("gemma-4-31B", key, n=1, prompt_tokens=60, out_tokens=4, user="burst")
    b = _no_traffic("G-17", "New-identity burst", out)
    if b:
        return b
    i, o = out.attributed(**{LABEL_TENANT: team})
    ev = {**summarize(out), "series_before_first_request": len(before), "first_scrape_value": (i, o),
          "consequence": "with no zero sample, Prometheus increase() over a window that holds only the "
                         "first value omits the series: the burst lands in __unmetered__ (Phase 2, O-04)"}
    if not before:
        return R.gap("G-17", "New-identity burst",
                     "the series does not exist until its first increment, so a first burst is invisible "
                     "to increase()", "requirements 7.3; Group B C5", holds=True, **ev)
    return R.gap("G-17", "New-identity burst", "series now pre-exported at zero", "Group B C5",
                 holds=False, **ev)
