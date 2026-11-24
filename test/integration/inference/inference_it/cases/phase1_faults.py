"""Phase 1: cases that inject faults. G-09 and G-16b restart pods and need --disruptive.

These are the cases plan Section 7 marks [B] or GAP. Where the plan expects a known gap the case
asserts the *current wrong behaviour* (XFAIL-GAP). When the behaviour is fixed it becomes FAIL
until the expectation is updated (plan Section 5.4).
"""
from __future__ import annotations

import time

from loadgen import build_prompt

from ..config import LABEL_TENANT
from ..gateway import ChatResult
from ..lab import Lab, Outcome
from ..registry import case
from .. import results as R
from .common import exact, ident, summarize

GAP_C1 = "BerriAI/litellm#51221 (spans only); counters: Group B C1/C2, upstream issue to file"
GAP_G08 = "requirements 7.3: estimated usage mixed into the exact counter; Group B C2 (open)"
# OpenCost tolerates this much over-attribution before it warns (attribution plan, default 0.1)
OVER_TOLERANCE = 0.1


def _verdict(out: Outcome, **labels) -> tuple[str, float, float]:
    """Compare attributed tokens with the engine oracle: equal, under, or over."""
    i, o = out.attributed(**labels)
    if i > out.e_in or o > out.e_out:
        return "over", i, o
    if i == out.e_in and o == out.e_out:
        return "equal", i, o
    return "under", i, o


# --------------------------------------------------------------------------------------- G-05
@case("G-05", "Failure after prefill began, before any output")
def g05(lab: Lab):
    return R.blocked("G-05", "Failure after prefill, before output",
                     "needs an engine error injected after prefill: a simulator error mode (Tier A) or a "
                     "fault-injected decode error (Tier B). The real vLLM here has no such hook and a "
                     "request rejected before prefill (G-07) is a different case. G-06 covers the same "
                     "counter path (failure after prefill) with a gateway timeout instead")


# --------------------------------------------------------------------------------------- G-06
@case("G-06", "Gateway timeout after prefill [B]")
def g06(lab: Lab):
    gw = lab.gw()
    key = lab.key("rag")
    prompt = build_prompt(1450)

    def action() -> list[ChatResult]:
        body = gw.chat_body("gemma-4-31B", prompt, 1500, "alice", False,
                            extra={"timeout": 3, "num_retries": 0})
        return [gw.chat(key, body, timeout=120)]

    out = lab.observe("gemma-4-31B", action, gw=gw, settle_s=10)
    res = out.results[0]
    ev = {**summarize(out), "request_status": res.status, "request_error": res.error,
          "litellm_rows": [(r.metric, r.delta) for r in out.rows]}
    if res.status == 200:
        return R.blocked("G-06", "Gateway timeout after prefill",
                         "the request completed inside the 3 s timeout, so no timeout occurred", **ev)
    if out.e_in <= 0:
        return R.blocked("G-06", "Gateway timeout after prefill",
                         "the engine did not process the prompt (rejected before prefill)", **ev)
    i, o = out.attributed(**ident("acme-corp/engineering", "rag"))
    ev.update(litellm_in=i, litellm_out=o,
              span_half="not run: needs a v1.111.0-dev image and an OTel exporter")
    if i == 0 and o == 0:
        return R.gap("G-06", "Gateway timeout after prefill",
                     f"engine counted +{out.e_in:.0f} prompt and +{out.e_out:.0f} generated tokens; "
                     "LiteLLM counters +0. Expected today: the difference lands in __unmetered__ (O-04)",
                     GAP_C1, holds=True, **ev)
    if (i, o) == (out.e_in, out.e_out):
        return R.gap("G-06", "Gateway timeout after prefill",
                     "LiteLLM now counts the failed request exactly", GAP_C1, holds=False, **ev)
    return R.gap("G-06", "Gateway timeout after prefill",
                 f"LiteLLM counted {i:.0f}/{o:.0f} against engine {out.e_in:.0f}/{out.e_out:.0f}: partial "
                 "recovery; update the expectation", GAP_C1, holds=False, **ev)


# --------------------------------------------------------------------------------------- G-08
@case("G-08", "Client disconnect mid-stream [B]")
def g08(lab: Lab):
    out = lab.traffic("gemma-4-31B", "rag", n=1, prompt_tokens=300, out_tokens=600, user="alice",
                      stream=True, abort_after_chunks=5)
    res = out.results[0]
    if not res.aborted:
        return R.blocked("G-08", "Client disconnect mid-stream",
                         f"the stream ended before the 5th chunk (status {res.status}, error {res.error})",
                         **summarize(out))
    verdict, i, o = _verdict(out, **ident("acme-corp/engineering", "rag"))
    ev = {**summarize(out), "chunks_received": res.content_chunks, "litellm_in": i, "litellm_out": o,
          "engine_generated": out.e_out, "difference_in": i - out.e_in, "difference_out": o - out.e_out,
          "mechanism": "on disconnect LiteLLM assembles the response from the chunks it received "
                       "(_bill_partial_streamed_spend_on_disconnect). With no provider usage chunk the "
                       "input count is LiteLLM's own tokenizer estimate, which need not equal the engine's"}
    if verdict == "equal":
        return R.gap("G-08", "Client disconnect mid-stream",
                     "attributed equals the engine; the gap no longer reproduces", GAP_G08, holds=False, **ev)
    # The contract wants an exact count or an estimate flagged as such. Today an estimate goes into
    # the exact counter. That is the gap; only an over-attribution beyond OpenCost's tolerance, or a
    # counter that records nothing for a request the engine served, is a defect in this build.
    if i > out.e_in * (1 + OVER_TOLERANCE) or o > out.e_out * (1 + OVER_TOLERANCE):
        return R.failed("G-08", "Client disconnect mid-stream",
                        f"LiteLLM attributed {i:.0f}/{o:.0f}, beyond the {OVER_TOLERANCE:.0%} tolerance over "
                        f"the engine {out.e_in:.0f}/{out.e_out:.0f}", **ev)
    kind = "above" if verdict == "over" else "below"
    return R.gap("G-08", "Client disconnect mid-stream",
                 f"attributed {i:.0f}/{o:.0f} is {kind} the engine {out.e_in:.0f}/{out.e_out:.0f}: an "
                 "estimate recorded as exact usage, with no usage_captured flag. The difference is "
                 "small here; a longer generation after the disconnect widens it (O-04)",
                 GAP_G08, holds=True, **ev)


# --------------------------------------------------------------------------------------- G-09
@case("G-09", "Engine killed mid-stream [B]", disruptive=True)
def g09(lab: Lab):
    alias = "gemma-4-31B-b"           # single replica in llm-b: the pod that dies is the one serving
    pf = lab.engines[alias]
    pods = sorted(pf.forwards)
    if len(pods) != 1:
        return R.blocked("G-09", "Engine killed mid-stream", f"need exactly one {alias} pod, have {pods}")
    pod = pods[0]
    gw, key = lab.gw(), lab.key("agent")
    oracle = lab.oracle(alias)
    e0 = oracle.wait_idle(quiet_s=4)
    c0 = gw.wait_counters_stable()
    snap: dict = {}

    def kill_after(chunk: int) -> None:
        if chunk == 20 and not snap:
            snap["engine_at_kill"] = oracle.snapshot()
            snap["killed_at"] = time.time()
            lab.cluster.delete_pod(pf.ns, pod, force=True)

    body = gw.chat_body(alias, build_prompt(300), 800, "erin", True)
    res = gw.chat(key, body, timeout=120, on_chunk=kill_after)
    time.sleep(8)
    c1 = gw.wait_counters_stable()
    from ..metrics import delta_rows
    rows = delta_rows(c0, c1)
    i = sum(r.delta for r in rows if r.metric.endswith("input_tokens_metric_total")
            and r.labels.get(LABEL_TENANT) == "globex")
    o = sum(r.delta for r in rows if r.metric.endswith("output_tokens_metric_total")
            and r.labels.get(LABEL_TENANT) == "globex")
    before = e0[f"{pf.ns}/{pod}"]
    last = snap.get("engine_at_kill", {}).get(f"{pf.ns}/{pod}")
    ev = {"chunks_received": res.content_chunks, "stream_error": res.error, "stream_status": res.status,
          "client_usage": (res.prompt_tokens, res.completion_tokens), "litellm_in": i, "litellm_out": o,
          "engine_prompt_at_kill": (last.prompt - before.prompt) if last else None,
          "engine_generation_at_kill": (last.generation - before.generation) if last else None,
          "note": "engine counters reset when the pod died, so only the reading at the 20th chunk is "
                  "usable; this case also feeds O-11"}
    recovered = lab.cluster.wait_pods(pf.ns, pf.selector, 1, lab.cfg.recover_timeout_s)
    if recovered:
        pf.sync()
    ev["engine_recovered"] = bool(recovered)
    if not snap:
        return R.blocked("G-09", "Engine killed mid-stream",
                         f"the stream finished before the kill (only {res.content_chunks} chunks)", **ev)
    return R.info("G-09", "Engine killed mid-stream",
                  f"LiteLLM recorded {i:.0f} input and {o:.0f} output tokens after {res.content_chunks} "
                  f"chunks; engine had processed {ev['engine_prompt_at_kill']}/"
                  f"{ev['engine_generation_at_kill']} at the kill. Discovery case: record, then pin", **ev)


# --------------------------------------------------------------------------------------- G-16b
@case("G-16b", "LiteLLM replica restart: counter reset does not break the next window", disruptive=True)
def g16b(lab: Lab):
    pf = lab.litellm_pods
    if pf is None:
        return R.blocked("G-16b", "LiteLLM replica restart", "lab.prepare() was not called")
    names = sorted(pf.forwards)
    if len(names) < 2:
        return R.blocked("G-16b", "LiteLLM replica restart", f"need 2 replicas, have {names}")
    victim = names[0]
    gw = lab.gateways()[0]
    lab.traffic("gemma-4-31B", "rag", n=2, prompt_tokens=100, out_tokens=8, user="alice", gw=gw)
    total_before = sum(gw.counters().values())
    lab.cluster.delete_pod(lab.cfg.litellm_ns, victim)
    ready = lab.cluster.wait_pods(lab.cfg.litellm_ns, lab.cfg.litellm_selector, 2, 300)
    if len(ready) < 2:
        return R.failed("G-16b", "LiteLLM replica restart", "replica did not come back inside 300 s")
    pf.sync()
    restarted = [g for g in lab.gateways() if g.name not in names]
    if not restarted:
        return R.blocked("G-16b", "LiteLLM replica restart", "could not identify the replacement pod")
    new = restarted[0]
    total_after = sum(new.counters().values())
    out = lab.traffic("gemma-4-31B", "rag", n=2, prompt_tokens=100, out_tokens=8, user="alice", gw=new)
    ok, i, o = exact(out, **ident("acme-corp/engineering", "rag", "alice"))
    ev = {**summarize(out), "old_replica_total": total_before, "new_replica_total_at_start": total_after,
          "counter_reset_observed": total_after < total_before, "litellm_in": i, "litellm_out": o}
    if not ok:
        return R.failed("G-16b", "LiteLLM replica restart",
                        f"post-restart delta {i:.0f}/{o:.0f} vs engine {out.e_in:.0f}/{out.e_out:.0f}", **ev)
    return R.passed("G-16b", "LiteLLM replica restart",
                    f"replacement replica counters restarted at {total_after:.0f} (was {total_before:.0f}) "
                    "and its post-restart delta equals the engine. Prometheus reset handling is O-11", **ev)
