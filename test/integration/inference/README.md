# Inference attribution integration tests

Automates Phase 0 prerequisites, Phase 1 (gateway conformance), Phase 2 (OpenCost attribution
correctness) and Phase 3 (API, metrics and authorization) of
[`docs/testing/inference-attribution-litellm-integration-test-plan.md`](../../../docs/testing/inference-attribution-litellm-integration-test-plan.md)
against the lab built by
[`docs/testing/inference-attribution-litellm-openshift-install.md`](../../../docs/testing/inference-attribution-litellm-openshift-install.md).

Phase 1 asserts LiteLLM's token counters against the **engine oracle** (vLLM and SGLang
`/metrics`, read directly from each pod). It does not use OpenCost output, so a failure points at
the gateway or the configuration, not at OpenCost.

Phase 2 asserts OpenCost itself: the `/inferenceCost` API and the `llm_dimension_*` gauges against
the same oracle and OpenCost's own model-level rates. Each case runs inside an explicit
`start,end` window that closes only after two Prometheus scrapes followed the last request, so
the exact-delta attribution (not `increase()` extrapolation) is what gets tested.

Phase 3 covers the API surface, the exported metrics and the `user_id` authorization rules.

Python 3.11 or later, standard library only. The `oc` CLI must be logged in.

## Run

From this directory:

```bash
python3 -m unittest discover -s tests          # pure logic, no cluster needed
python3 -m inference_it list
python3 -m inference_it run --wait 900         # Phase 0 + non-disruptive Phases 1, 2 and 3
python3 -m inference_it run --cases G-01 O-03  # selected cases
python3 -m inference_it run --disruptive       # also G-09, G-16b and O-11, which restart pods
python3 -m inference_it run --cases O-19 O-20 O-21   # the matrix and its variants alone
```

Phase 2 needs the OpenCost install from the install guide (Phase 3 there): the API is reached
through its own port-forward, started on first use, and the admin token is read from the
`opencost-admin` secret. `OPENCOST_SVC`, `OPENCOST_PORT`, `OPENCOST_DEPLOY` and
`OPENCOST_SELECTOR` override the names (defaults `opencost2`, `9003`, `opencost2`,
`app.kubernetes.io/name=opencost`).

The runner starts its own `oc port-forward` processes, bound to `127.0.0.1` on free local ports.
It follows pod replacements, restarts a tunnel that stops relaying, and retries `oc` calls that
fail with the cluster's transient TLS timeouts. No manual port-forward is needed.

Exit code: 0 when every case is PASS, XFAIL-GAP or INFO; 1 on any FAIL or BLOCKED; 2 when the
lab was not ready.

## Configuration

Names default to the install guide and follow the same environment variables: `OC_NS`,
`LITELLM_NS`, `ENGINE_NS_A`, `ENGINE_NS_B`. Also `PROMETHEUS_SVC`, `PROMETHEUS_PORT`, `OC_BIN`
and `INFERENCE_IT_OUT` (report directory, default `runs/`).

Secrets (the LiteLLM master key and the virtual keys) are read from the cluster secrets
`litellm-secrets` and `litellm-virtual-keys` at run time and kept in memory. They are never written
to a report or a file.

## Cases

| Case | Checks | Needs |
|---|---|---|
| P-02 P-03 P-04 P-08 P-09 | label census, scrape path and auth, served-model equality, workload and user labels, LiteLLM settings | |
| G-01 | exact attributed tokens per key, no leakage to other series | |
| G-02 | streaming, with and without the client's `include_usage` | |
| G-03 | missing team, key alias and user: one sentinel | creates and deletes a team and key |
| G-04 | router rewrite: `model` is the engine's `model_name` | |
| G-06 | gateway timeout after prefill: asserted as XFAIL-GAP | |
| G-07 | rejected requests record nothing | |
| G-08 | client disconnect mid-stream: XFAIL-GAP | |
| G-10 | forged headers and metadata; body `user` is caller-asserted | |
| G-11 | anonymous scrape, Prometheus write endpoints, pushgateway | |
| G-14 | prefix-cache reuse | |
| G-15 | SaaS stub through the same LiteLLM | |
| G-16 | two replicas: per-replica deltas sum to the engine delta | 2 LiteLLM replicas |
| G-17 | new identity: XFAIL-GAP while the series is not pre-exported | creates and deletes a team and key |
| G-09, G-16b | engine killed mid-stream; LiteLLM replica restart | `--disruptive` |
| G-05, G-12, G-13 | reported BLOCKED with the exact change they need | engine fault hook, `cache: true`, a fallback entry |

States follow plan Section 5.4: PASS, FAIL, XFAIL-GAP, BLOCKED, INFO. An XFAIL-GAP case asserts
today's wrong behaviour. When upstream fixes it, the case turns FAIL until the expectation is
updated.

## Phase 2 cases (plan Section 8)

| Case | Checks | Notes |
|---|---|---|
| O-01 | token coverage gauges and API coverage within [0.9, 1.1] | polls the gauges while light traffic keeps the last collection cycle non-empty |
| O-02 | row cost equals in x inRate + out x outRate, both bases, tokens exact | |
| O-03 | release gates: tenant rows + `__unmetered__` equal the model total, both bases | fails the suite |
| O-04 | bypass traffic in `__unmetered__`, G-17 burst attributed, coverage drops | sends requests directly to the engine pods |
| O-05 | sentinel traffic under `__unidentified__`, identity coverage gauge 0 | creates a team without a key alias |
| O-06 | workload rows split by key alias, forged header ignored | |
| O-07 | user rows split, no-user rows carry an empty `user_id` | needs the admin token |
| O-08 | namespace resolution: single ns exact, two ns proportional, V2 map | O-08b BLOCKED (env change) |
| O-09 | same model in two namespaces (#4163) | |
| O-10 | baseline: no over-attribution, remainder non-negative | V6 and G-12 halves BLOCKED (config) |
| O-11 | LiteLLM replica restart mid-window: totals equal the oracle | `--disruptive`; engine half BLOCKED (reload takes 20-60 min) |
| O-12 | window exactness: two back-to-back windows keep their teams | |
| O-13 | SaaS series dropped by scope, not unmatched | V4/V5 halves BLOCKED (config) |
| O-14 | SGLang decode exclusion | BLOCKED: the lab has no disaggregated SGLang |
| O-15 | model-name join, no unmatched keys | the qualified-name variant needs a config change; noted |
| O-16 | idle on the allocation basis; zero-token window fully in `__unmetered__` | waits ~2 min quiet |
| O-17 | cached tokens priced at the full input rate [B] | |
| O-18 | estimated token share 0 | |
| O-19 | 3 tenants x 2 workloads x (3 users + none), every cell and aggregation exact | ~10 min; stores its totals for O-20/O-21 |
| O-20 | V3 (user label off): user_id empty, nothing else moves | patches the attribution ConfigMap, restarts OpenCost, restores in `finally` |
| O-21 | V7 (workload label off): tenant-only attribution | same mechanism as O-20 |

Aggregation lists always name `model_name` (and `namespace` where a case filters by it): OpenCost
clears properties that are not part of the aggregation key, so the plan's `aggregate=tenant_id`
is run as `model_name,tenant_id`.

O-20 and O-21 re-run the O-19 traffic under the same harness-made identities, so their per-tenant
totals are comparable with O-19's. Run them together: `--cases O-19 O-20 O-21`.

## Phase 3 cases (plan Section 9)

| Case | Checks | Notes |
|---|---|---|
| A-01 | tenant rows plus `__unmetered__` and `__unidentified__`, combined with model/namespace/workload aggregates | |
| A-02 | the `(acme-corp/engineering, rag)` / `(acme-corp, engineering/rag)` pairs stay separate, per user, nothing overwritten | two runs with different users |
| A-03 | `aggregate=namespace`, `model_name` and the natural key identical with attribution on and off | toggles `INFERENCE_ATTRIBUTION_ENABLED` via `oc set env`, restores in `finally`; queries the same closed window both times |
| A-04 | `user_id` aggregate/filter needs the flag AND admin (400 otherwise); filter-only unauthorized returns rows merged across users with `user_id` stripped; per-user rows sum to the merged row; tenant/workload queries unrestricted | toggles `INFERENCE_ATTRIBUTION_USER_AGGREGATION_ENABLED` off for the flag-off half |
| A-05 | `llm_dimension_hourly_cost` label set incl. `workload_id`/`user_id`; empty `user_id` as an empty label; under V3 empty everywhere with unchanged totals | patches the attribution ConfigMap like O-20; records how the lab limits Prometheus access |
| A-06 | `/inferenceCost/timeseries?accumulate=hour`: steps sum to `/total`, reconciliation per step | a sub-hour window yields one step; multi-step needs a >1h window |
| A-07 | runner-vs-API parity for the same burst, and 6 identical concurrent queries under load (shared-collector race) | parity within 5%: the runner cycle and the API window can differ by one poll interval of model cost at the edges |
| A-08 | raw JSON: empty dimensions omitted, `approximate` only on split rows; single-row tenant response with non-zero `TotalTokens` and self-consistent rates | the over-attribution-scaling half needs V6; noted |

A-03 and A-04 restart OpenCost twice each (toggle and restore); A-05 restarts it twice more for
the V3 variant. All restores run in `finally`, and every query they compare runs against the same
already-closed window, so the restarts cannot change the compared data.

## Known limits

- Span checks (`gen_ai.usage.*`, `gen_ai.request.model`) are not run: the lab has no OTel exporter.
  G-04 and G-06 say so in their evidence.
- Counter assertions are exact, so a concurrent client sending traffic through the same engines
  breaks them. Run the suite alone.
- Counter reads wait until each LiteLLM replica's counters stop moving (the callback is
  asynchronous), and until the engines are idle, so aborted requests are counted.
- The sentinel `"None"` and the label names are pinned in `inference_it/config.py`. G-03 and P-08
  fail if LiteLLM changes them.

## Output

`runs/phase1-<run id>.json` (evidence per case) and `.md` (results table and the requirements
Section 10 selection checklist). `runs/` is git-ignored.
