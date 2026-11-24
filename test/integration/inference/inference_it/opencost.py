"""OpenCost client for Phase 2: the /inferenceCost API and the llm_dimension_* metrics.

The API is read through a loopback port-forward. The admin token is read from the cluster secret
at run time and kept in memory only (never written to a report).
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone

UNMETERED = "__unmetered__"
UNIDENTIFIED = "__unidentified__"
BASES = ("allocation", "usage")


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message = status, message


def rfc3339(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window(t0: float, t1: float) -> str:
    """Explicit `start,end` window (plan Section 5.3); never a relative one like 24h."""
    return f"{rfc3339(t0)},{rfc3339(t1)}"


@dataclass
class Entry:
    """One row of /inferenceCost/total, with the JSON field names of InferenceCostResponse."""
    key: str
    model: str
    namespace: str
    tenant: str
    workload: str
    user: str
    approximate: bool
    cost: float
    input_cost: float
    output_cost: float
    prompt: float
    generation: float
    total_tokens: float
    in_rate: float          # inputCostPerMillionTokens
    out_rate: float         # outputCostPerMillionTokens
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_json(key: str, d: dict) -> "Entry":
        p = d.get("properties", {})
        return Entry(key, p.get("modelName", ""), p.get("namespace", ""), p.get("tenant_id", ""),
                     p.get("workload_id", ""), p.get("user_id", ""), bool(p.get("approximate")),
                     d.get("totalCost", 0.0), d.get("inputCost", 0.0), d.get("outputCost", 0.0),
                     d.get("promptTokens", 0.0), d.get("generationTokens", 0.0),
                     d.get("totalTokens", 0.0), d.get("inputCostPerMillionTokens", 0.0),
                     d.get("outputCostPerMillionTokens", 0.0), d)


class OpenCost:
    def __init__(self, port: int, admin_token: str | None = None):
        self.port, self.admin_token = port, admin_token

    def _get(self, path: str, params: dict, admin: bool, timeout: float = 120) -> tuple[int, bytes]:
        headers = {}
        if admin and self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"
        url = path + "?" + urllib.parse.urlencode(params, safe=",:\"+")
        last: Exception | None = None
        for _ in range(4):   # the loopback tunnel can die while the supervisor loop re-attaches
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
            try:
                conn.request("GET", url, headers=headers)
                resp = conn.getresponse()
                return resp.status, resp.read()
            except (OSError, http.client.HTTPException) as e:
                last = e
                time.sleep(2)
            finally:
                conn.close()
        raise RuntimeError(f"OpenCost API unreachable through the tunnel after retries: {last}")

    def raw(self, endpoint: str, params: dict, admin: bool = False) -> tuple[int, dict]:
        status, data = self._get(endpoint, params, admin)
        try:
            return status, json.loads(data or b"{}")
        except json.JSONDecodeError:
            return status, {"message": data.decode(errors="replace")[:300]}

    def total(self, win: str, basis: str = "allocation", aggregate: str | None = None,
              filter: str | None = None, admin: bool = False, accumulate: str | None = None
              ) -> list[Entry]:
        """GET /inferenceCost/total. Raises ApiError on a non-200 status."""
        params = {"window": win, "costBasis": basis}
        if aggregate:
            params["aggregate"] = aggregate
        if filter:
            params["filter"] = filter
        if accumulate:
            params["accumulate"] = accumulate
        status, body = self.raw("/inferenceCost/total", params, admin)
        if status != 200:
            raise ApiError(status, str(body.get("message") or body)[:300])
        costs = (body.get("data") or {}).get("inferenceCosts") or {}
        return [Entry.from_json(k, v) for k, v in sorted(costs.items())]

    def timeseries(self, win: str, accumulate: str, basis: str = "allocation",
                   aggregate: str | None = None, admin: bool = False) -> list[list[Entry]]:
        params = {"window": win, "costBasis": basis, "accumulate": accumulate}
        if aggregate:
            params["aggregate"] = aggregate
        status, body = self.raw("/inferenceCost/timeseries", params, admin)
        if status != 200:
            raise ApiError(status, str(body.get("message") or body)[:300])
        sets = (body.get("data") or {}).get("inferenceCostSets") or []
        return [[Entry.from_json(k, v) for k, v in sorted((s.get("inferenceCosts") or {}).items())]
                for s in sets]

    def status(self, endpoint: str = "/inferenceCost/total") -> int:
        """HTTP status of an endpoint with no window (501 when the feature is off, 400 when on)."""
        return self._get(endpoint, {}, False)[0]

    def metrics_text(self) -> str:
        """GET /metrics: the llm_dimension_* gauges OpenCost exports (no auth required)."""
        status, data = self._get("/metrics", {}, False, timeout=30)
        if status != 200:
            raise ApiError(status, "metrics endpoint")
        return data.decode(errors="replace")


# --- pure helpers over Entry lists (unit tested without a cluster) ---------------------------

def by_key(entries: list[Entry], *fields: str) -> dict[tuple, Entry]:
    """Index rows by the given Entry attributes, e.g. by_key(rows, 'tenant', 'workload')."""
    return {tuple(getattr(e, f) for f in fields): e for e in entries}


def total_cost(entries: list[Entry]) -> float:
    return sum(e.cost for e in entries)


def total_tokens(entries: list[Entry]) -> tuple[float, float]:
    return sum(e.prompt for e in entries), sum(e.generation for e in entries)


def close(a: float, b: float, rel: float = 1e-6, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def expected_cost(prompt: float, generation: float, in_rate: float, out_rate: float) -> float:
    """in x inRate/1e6 + out x outRate/1e6 (plan Section 5.2). Rates come from OpenCost itself."""
    return prompt * in_rate / 1e6 + generation * out_rate / 1e6
