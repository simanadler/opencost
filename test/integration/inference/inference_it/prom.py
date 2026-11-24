"""Minimal Prometheus HTTP API client (read-only queries, plus the write-path probe for G-11)."""
from __future__ import annotations

import http.client
import json
import math
import re
import urllib.parse


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str) -> float:
    """Prometheus duration ("30s", "1m30s", "500ms") in seconds."""
    parts = _DURATION.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        raise ValueError(f"not a duration: {text!r}")
    return sum(float(n) * _UNIT_S[u] for n, u in parts)


class Prom:
    def __init__(self, port: int):
        self.port = port

    def _call(self, method: str, path: str, body: bytes | None = None, timeout: float = 30,
              headers: dict | None = None) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request(method, path, body, headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def query(self, expr: str) -> list[dict]:
        status, data = self._call("GET", "/api/v1/query?" + urllib.parse.urlencode({"query": expr}))
        if status != 200:
            raise RuntimeError(f"prometheus query -> {status}: {data[:300]!r} (expr: {expr[:200]})")
        payload = json.loads(data)
        if payload.get("status") != "success":
            raise RuntimeError("prometheus query failed")
        return payload["data"]["result"]

    def targets(self) -> list[dict]:
        status, data = self._call("GET", "/api/v1/targets?state=active")
        if status != 200:
            raise RuntimeError(f"prometheus targets -> {status}")
        return json.loads(data)["data"]["activeTargets"]

    def server_time(self) -> float:
        """Prometheus's own clock (the `time()` scalar). Scrape timestamps are in this clock."""
        status, data = self._call("GET", "/api/v1/query?" + urllib.parse.urlencode({"query": "time()"}))
        if status != 200:
            raise RuntimeError(f"prometheus time() -> {status}")
        return float(json.loads(data)["data"]["result"][1])

    def scrape_interval(self, jobs: tuple[str, ...], default: float = 30.0) -> float:
        """Longest scrape interval among the active targets of `jobs`, in seconds."""
        found = [parse_duration(t["scrapeInterval"]) for t in self.targets()
                 if t.get("labels", {}).get("job") in jobs and t.get("scrapeInterval")]
        return max(found) if found else default

    def active_targets(self, jobs: tuple[str, ...]) -> list[tuple[str, str]]:
        """(job, instance) of every healthy target of `jobs`."""
        return sorted((t["labels"]["job"], t["labels"].get("instance", "")) for t in self.targets()
                      if t.get("labels", {}).get("job") in jobs and t.get("health") == "up")

    def scrape_counts(self, since_server: float, jobs: tuple[str, ...]) -> dict[tuple[str, str], int]:
        """Samples of `up` per (job, instance) with timestamp after `since_server` (server clock).

        One `up` sample is written per scrape, so this counts scrapes. A target with none is
        absent from the Prometheus answer and is reported as 0 by wait logic in the caller.
        """
        span = max(1, int(math.ceil(self.server_time() - since_server)))
        # Escape for a PromQL double-quoted string literal (RE2), not Python regex: re.escape
        # would emit `\-` for hyphens, which PromQL rejects as an unknown escape sequence.
        job_re = "|".join(j.replace("\\", "\\\\").replace('"', '\\"') for j in jobs)
        out = {}
        for r in self.query(f'count_over_time(up{{job=~"{job_re}"}}[{span}s])'):
            m = r["metric"]
            out[(m.get("job", ""), m.get("instance", ""))] = int(float(r["value"][1]))
        return out

    def probe_write(self, path: str, content_type: str = "application/x-protobuf") -> int:
        """POST an empty body to a write endpoint with no credentials. Returns the HTTP status."""
        status, _ = self._call("POST", path, b"", headers={"Content-Type": content_type})
        return status
