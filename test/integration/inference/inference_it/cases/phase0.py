"""Phase 0 checks that Phase 1 depends on (plan Section 6): P-02, P-03, P-04, P-08, P-09.

P-01 and P-05..P-07 test OpenCost and belong to Phase 2.
"""
from __future__ import annotations

import http.client
import re

from ..config import (LABEL_MODEL, LABEL_REQUESTED_MODEL, LABEL_TENANT, LABEL_USER, LABEL_WORKLOAD,
                      MATRIX, SENTINEL)
from ..lab import Lab
from ..registry import case
from .. import results as R
from .common import ident, summarize

REQUIRED_LABELS = {LABEL_TENANT, LABEL_WORKLOAD, LABEL_USER, LABEL_MODEL}


@case("P-02", "Label census on the LiteLLM token counters", phase=0)
def p02(lab: Lab):
    out = lab.traffic("gemma-4-31B", "rag", n=1, prompt_tokens=60, out_tokens=4, user="census")
    labels = sorted({k for r in out.rows for k in r.labels})
    missing = sorted(REQUIRED_LABELS - set(labels))
    evidence = {"labels": labels, "user_email_dropped": "user_email" not in labels,
                "extra_labels_present": sorted(set(labels) - REQUIRED_LABELS)}
    if missing:
        return R.failed("P-02", "Label census", f"counters lack {missing}", **evidence)
    if "user_email" in labels:
        return R.failed("P-02", "Label census", "user_email is exported (requirements 4.2)", **evidence)
    return R.passed("P-02", "Label census", f"labels: {', '.join(labels)}", **evidence)


@case("P-03", "Scrape path and authentication", phase=0)
def p03(lab: Lab):
    gw = lab.gw()
    authed, _ = gw.metrics_text(True)
    anon, _ = gw.metrics_text(False)
    # /metrics without the slash: do not follow redirects, record the raw status
    conn = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=20)
    conn.request("GET", "/metrics", headers={"Authorization": f"Bearer {lab.master_key}"})
    noslash = conn.getresponse().status
    conn.close()
    ev = {"authenticated": authed, "unauthenticated": anon, "no_trailing_slash": noslash}
    if authed == 200 and anon in (401, 403):
        return R.passed("P-03", "Scrape path", "200 with credential, refused without. "
                        f"/metrics without slash returns {noslash}", **ev)
    return R.failed("P-03", "Scrape path", "expected 200 authenticated and 401/403 anonymous", **ev)


@case("P-04", "Served-model equality (plain and rewrite aliases)", phase=0)
def p04(lab: Lab):
    findings, bad = {}, []
    for alias, key, engine_model in (("gemma-4-31B", "agent", "gemma-4-31B"),
                                     ("gemma-4-31B-b", "agent", "gemma-4-31B"),
                                     ("llama-chat", "chat", "llama")):
        out = lab.traffic(alias, key, n=1, prompt_tokens=60, out_tokens=4, user="p04")
        models = sorted({r.labels.get(LABEL_MODEL) for r in out.rows})
        requested = sorted({r.labels.get(LABEL_REQUESTED_MODEL) for r in out.rows})
        findings[alias] = {"litellm_model": models, "requested_model": requested,
                           "engine_model_name": engine_model, "statuses": out.statuses}
        if models != [engine_model]:
            bad.append(alias)
    if bad:
        return R.failed("P-04", "Served-model equality",
                        f"model label differs from engine model_name for {bad}. If provider-qualified "
                        "this is Group B change C4", **findings)
    return R.passed("P-04", "Served-model equality",
                    "model label equals the engine model_name for every alias", **findings)


@case("P-08", "Workload and user label discovery across the key matrix", phase=0)
def p08(lab: Lab):
    seen, bad = {}, []
    users = {"rag": "alice", "batch": "bob", "engineering-rag": "dave", "chat": "erin", "agent": "erin"}
    for key, user in users.items():
        out = lab.traffic("gemma-4-31B", key, n=1, prompt_tokens=60, out_tokens=4, user=user)
        tenant, workload = MATRIX[key]
        want = ident(tenant, workload, user)
        got = [r.labels for r in out.rows]
        ok = bool(got) and all(all(l.get(k) == v for k, v in want.items()) for l in got)
        seen[key] = {"expected": want, "series": sorted({str(sorted(l.items())) for l in got})[:1]}
        if not ok:
            bad.append(key)
    if bad:
        return R.blocked("P-08", "Label discovery",
                         f"labels do not carry team/key alias/body user for {bad}; dependent cases "
                         "O-05..O-07, O-19, A-02, A-04 are BLOCKED", **seen)
    return R.passed("P-08", "Label discovery",
                    f"{LABEL_TENANT}=team, {LABEL_WORKLOAD}=key alias, {LABEL_USER}=body user", **seen)


@case("P-09", "LiteLLM configuration check", phase=0)
def p09(lab: Lab):
    cluster, cfg = lab.cluster, lab.cfg
    ev: dict = {}
    pods = cluster.pods(cfg.litellm_ns, cfg.litellm_selector)
    ev["ready_replicas"] = len(pods)
    ev["restarts"] = [p["restarts"] for p in pods]
    log_hits: list[str] = []
    for p in pods:
        text = cluster.logs(cfg.litellm_ns, p["name"], tail=2000)
        log_hits += [ln[:160] for ln in text.splitlines()
                     if re.search(r"unknown (setting|param)|invalid (setting|config)|unexpected keyword", ln, re.I)]
    ev["unknown_setting_log_lines"] = log_hits[:5]
    # (b) positive half: end_user populated for a request carrying a body user
    out = lab.traffic("gemma-4-31B", "rag", n=1, prompt_tokens=60, out_tokens=4, user="p09-user")
    ev["end_user_values"] = sorted({r.labels.get(LABEL_USER) for r in out.rows})
    # (c) user_email absent
    status, text = lab.gw().metrics_text(True)
    ev["user_email_in_scrape"] = "user_email" in text
    # (d) unauthenticated scrape refused
    anon, _ = lab.gw().metrics_text(False)
    ev["anonymous_status"] = anon
    # (e) Group B state: does the failure path or an estimated series exist in this build?
    names = set(re.findall(r"^# TYPE (litellm_\w*(?:estimated|usage_captured)\w*)", text, re.M))
    ev["group_b_estimated_series"] = sorted(names)
    problems = []
    if log_hits:
        problems.append("unknown-setting messages in LiteLLM log")
    if ev["end_user_values"] != ["p09-user"]:
        problems.append("end_user not populated (A2)")
    if ev["user_email_in_scrape"]:
        problems.append("user_email exported (A4)")
    if anon not in (401, 403):
        problems.append("anonymous scrape allowed (A8)")
    if problems:
        return R.failed("P-09", "LiteLLM configuration", "; ".join(problems), **ev)
    ev["note"] = ("negative half of (b), end_user empty when A2 is off, needs a config change and is "
                  "not run here")
    return R.passed("P-09", "LiteLLM configuration",
                    "settings accepted, end_user populated, user_email dropped, anonymous scrape refused",
                    **ev)
