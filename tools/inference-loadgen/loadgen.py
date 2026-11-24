#!/usr/bin/env python3
"""Client-side load generator for the inference attribution integration test.

Implements the traffic side of docs/testing/inference-attribution-litellm-integration-test-plan.md:

  * Section 5.1  deterministic traffic: fixed prompt corpus, forced completion length
  * Section 5.2  oracle: engine counter deltas (E_in, E_out) read directly from engine /metrics
  * Section 5.3  explicit window: T0 before the first request, T1 after the last request
  * Section 12   a machine-readable run record

It sends requests through LiteLLM with a virtual key (tenant and workload are key-derived)
and a body `user` (caller-asserted). It does not query Prometheus or OpenCost; the assertion
runner consumes the JSON record this tool writes.

Standard library only, so it runs without a virtualenv.

Secrets come from environment variables, never from arguments or files:
  LITELLM_KEY_<WORKLOAD>   virtual key for a workload, e.g. LITELLM_KEY_RAG
  (or pass --key-env NAME to choose the variable)

Usage:
  python3 loadgen.py calibrate --gateway http://127.0.0.1:4000 --engine http://127.0.0.1:18000 \
      --model gemma-4-31B --key-env LITELLM_KEY_RAG

  python3 loadgen.py case --case-id O-02-rag --gateway http://127.0.0.1:4000 \
      --engine http://127.0.0.1:18000 --model gemma-4-31B --key-env LITELLM_KEY_RAG \
      --user alice --requests 20 --prompt-tokens 512 --output-tokens 128 \
      --out runs/O-02-rag.json

  python3 loadgen.py case ... --stream        # streaming variant (G-02)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Engine counters named in plan Section 5.2. vLLM uses vllm:*, SGLang uses sglang:*.
PROMPT_COUNTERS = ("vllm:prompt_tokens_total", "sglang:prompt_tokens_total")
GENERATION_COUNTERS = ("vllm:generation_tokens_total", "sglang:generation_tokens_total")

METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+\-.]+)")

# A fixed corpus so prompts are reproducible across runs (plan Section 5.1).
CORPUS = (
    "The quarterly review covers throughput, latency, and cost per token across "
    "every serving pool. Engineers compare cache hit rates, batch sizes, and queue "
    "depth before deciding whether to add replicas or move workloads between nodes. "
)


@dataclass
class RequestResult:
    index: int
    ok: bool
    status: int | None
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    latency_s: float
    error: str | None = None


@dataclass
class CaseRecord:
    case_id: str
    gateway: str
    engine: str
    model: str
    key_env: str
    user: str | None
    stream: bool
    requests_sent: int
    requests_ok: int
    prompt_tokens_per_request: int
    output_tokens_per_request: int
    t0: str
    t1: str
    engine_before: dict
    engine_after: dict
    engine_delta_prompt: float
    engine_delta_generation: float
    client_usage_prompt_total: int
    client_usage_completion_total: int
    results: list[RequestResult] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_key(key_env: str) -> str:
    key = os.environ.get(key_env, "")
    if not key:
        sys.exit(f"error: environment variable {key_env} is not set")
    return key


def build_prompt(target_tokens: int) -> str:
    """Return a reproducible prompt of approximately `target_tokens` tokens.

    The corpus is repeated, so the prompt is the same for every run with the same target.
    The real token count is whatever the engine reports; the oracle uses the engine's
    counter delta, never this estimate (plan Section 5.1).
    """
    repeats = max(1, target_tokens // 40)
    return CORPUS * repeats


def http_json(url: str, method: str = "GET", body: dict | None = None,
              headers: dict | None = None, timeout: float = 300.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            payload = {}
        return e.code, payload


def read_engine_counters(engine_url: str) -> dict:
    """Return the four oracle counters summed across series (for example, per engine_type)."""
    req = urllib.request.Request(engine_url.rstrip("/") + "/metrics")
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode()
    totals = {name: 0.0 for name in PROMPT_COUNTERS + GENERATION_COUNTERS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name in totals:
            totals[name] += float(m.group(3))
    return totals


def engine_delta(before: dict, after: dict, names: tuple[str, ...]) -> float:
    """Sum the delta of the counter that exists on this engine (vLLM or SGLang)."""
    present = [n for n in names if after.get(n, 0.0) or before.get(n, 0.0)]
    if not present:
        return 0.0
    name = present[0]
    return after[name] - before[name]


def chat_body(model: str, prompt: str, out_tokens: int, user: str | None, stream: bool) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Forced completion length (plan Section 5.1). min_tokens and ignore_eos are
        # vLLM extensions; LiteLLM passes unknown fields through to the engine.
        "max_tokens": out_tokens,
        "min_tokens": out_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": stream,
    }
    if user is not None:
        body["user"] = user
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def send_request(gateway: str, key: str, body: dict, index: int) -> RequestResult:
    start = time.monotonic()
    headers = {"Authorization": f"Bearer {key}"}
    if body.get("stream"):
        return _send_stream(gateway, headers, body, index, start)
    try:
        status, payload = http_json(gateway.rstrip("/") + "/v1/chat/completions",
                                    method="POST", body=body, headers=headers)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return RequestResult(index, False, None, None, None, time.monotonic() - start, str(e))
    latency = time.monotonic() - start
    if status != 200:
        return RequestResult(index, False, status, None, None, latency,
                             json.dumps(payload.get("error", payload))[:300])
    usage = payload.get("usage") or {}
    return RequestResult(index, True, status,
                         usage.get("prompt_tokens"), usage.get("completion_tokens"), latency)


def _send_stream(gateway: str, headers: dict, body: dict, index: int, start: float) -> RequestResult:
    data = json.dumps(body).encode()
    req = urllib.request.Request(gateway.rstrip("/") + "/v1/chat/completions", data=data,
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
    except urllib.error.HTTPError as e:
        return RequestResult(index, False, e.code, None, None, time.monotonic() - start,
                             e.read().decode(errors="replace")[:300])
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return RequestResult(index, False, None, None, None, time.monotonic() - start, str(e))
    return RequestResult(index, True, 200, usage.get("prompt_tokens"),
                         usage.get("completion_tokens"), time.monotonic() - start)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Send one request per target size and report the engine's own prompt-token delta.

    Plan Section 5.1: the prompt length is calibrated against the engine, not a client tokenizer.
    """
    key = read_key(args.key_env)
    results = {}
    for target in args.targets:
        before = read_engine_counters(args.engine)
        res = send_request(args.gateway, key,
                           chat_body(args.model, build_prompt(target), args.output_tokens,
                                     args.user, stream=False), index=0)
        after = read_engine_counters(args.engine)
        results[str(target)] = {
            "ok": res.ok,
            "engine_delta_prompt": engine_delta(before, after, PROMPT_COUNTERS),
            "engine_delta_generation": engine_delta(before, after, GENERATION_COUNTERS),
            "client_usage_prompt_tokens": res.usage_prompt_tokens,
            "error": res.error,
        }
        print(f"target={target} ok={res.ok} engine_prompt_delta="
              f"{results[str(target)]['engine_delta_prompt']}", file=sys.stderr)
    print(json.dumps({"calibrated_at": utc_now(), "model": args.model, "results": results},
                     indent=2))
    return 0


def cmd_case(args: argparse.Namespace) -> int:
    key = read_key(args.key_env)
    prompt = build_prompt(args.prompt_tokens)

    # Engine oracle before the first request (plan Section 5.3: T0 before any traffic)
    engine_before = read_engine_counters(args.engine)
    t0 = utc_now()

    results: list[RequestResult] = []
    for i in range(args.requests):
        body = chat_body(args.model, prompt, args.output_tokens, args.user, args.stream)
        results.append(send_request(args.gateway, key, body, index=i))
        if args.interval > 0:
            time.sleep(args.interval)

    # Hold until the caller has waited for scrapes (plan Section 5.1). The record notes T1;
    # the assertion runner queries Prometheus and OpenCost over [T0, T1] after the wait.
    if args.settle > 0:
        time.sleep(args.settle)
    t1 = utc_now()
    engine_after = read_engine_counters(args.engine)

    ok = [r for r in results if r.ok]
    record = CaseRecord(
        case_id=args.case_id,
        gateway=args.gateway,
        engine=args.engine,
        model=args.model,
        key_env=args.key_env,
        user=args.user,
        stream=args.stream,
        requests_sent=len(results),
        requests_ok=len(ok),
        prompt_tokens_per_request=args.prompt_tokens,
        output_tokens_per_request=args.output_tokens,
        t0=t0,
        t1=t1,
        engine_before=engine_before,
        engine_after=engine_after,
        engine_delta_prompt=engine_delta(engine_before, engine_after, PROMPT_COUNTERS),
        engine_delta_generation=engine_delta(engine_before, engine_after, GENERATION_COUNTERS),
        client_usage_prompt_total=sum(r.usage_prompt_tokens or 0 for r in ok),
        client_usage_completion_total=sum(r.usage_completion_tokens or 0 for r in ok),
        results=results,
    )
    out = json.dumps(asdict(record), indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"wrote {args.out}: {len(ok)}/{len(results)} ok, "
              f"engine prompt delta {record.engine_delta_prompt}, "
              f"generation delta {record.engine_delta_generation}", file=sys.stderr)
    else:
        print(out)
    return 0 if len(ok) == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--gateway", required=True, help="LiteLLM base URL, e.g. http://127.0.0.1:4000")
        sp.add_argument("--engine", required=True,
                        help="Engine /metrics base URL via port-forward, e.g. http://127.0.0.1:18000")
        sp.add_argument("--model", required=True, help="Client model alias sent to LiteLLM")
        sp.add_argument("--key-env", required=True, help="Env var holding the virtual key")
        sp.add_argument("--user", default=None, help="Body user field (caller-asserted end user)")

    cal = sub.add_parser("calibrate", help="Calibrate prompt length against the engine counters")
    common(cal)
    cal.add_argument("--targets", type=int, nargs="+", default=[256, 512, 1024, 1450])
    cal.add_argument("--output-tokens", type=int, default=16)
    cal.set_defaults(func=cmd_calibrate)

    case = sub.add_parser("case", help="Run one deterministic case and write its record")
    common(case)
    case.add_argument("--case-id", required=True, help="Plan case ID, e.g. O-02-rag")
    case.add_argument("--requests", type=int, default=10)
    case.add_argument("--prompt-tokens", type=int, default=512)
    case.add_argument("--output-tokens", type=int, default=128)
    case.add_argument("--stream", action="store_true", help="Use stream=true with include_usage")
    case.add_argument("--interval", type=float, default=0.0, help="Seconds between requests")
    case.add_argument("--settle", type=float, default=0.0,
                      help="Seconds to wait after the last request before T1 (at least two scrape intervals)")
    case.add_argument("--out", default=None, help="Write the JSON record to this path")
    case.set_defaults(func=cmd_case)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
