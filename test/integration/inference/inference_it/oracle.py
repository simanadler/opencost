"""Engine oracle: tokens the engines actually processed (plan Sections 3 and 5.2).

Read directly from each engine pod's /metrics, independent of Prometheus scrape timing.
SGLang prefill/decode disaggregation reports the full prompt on both workers, so series with
engine_type="decode" are excluded from the prompt reference (plan O-14).
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field

from .cluster import PodForwards
from .metrics import parse

PROMPT = {"vllm:prompt_tokens_total", "sglang:prompt_tokens_total"}
GENERATION = {"vllm:generation_tokens_total", "sglang:generation_tokens_total"}
CACHED = {"vllm:prompt_tokens_cached_total"}


@dataclass(frozen=True)
class Reading:
    prompt: float
    generation: float
    cached: float
    models: frozenset


@dataclass
class OracleDelta:
    prompt: float = 0.0
    generation: float = 0.0
    cached: float = 0.0
    per_pod: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def reading_from_text(text: str) -> Reading:
    prompt = generation = cached = 0.0
    models = set()
    for s in parse(text, PROMPT | GENERATION | CACHED):
        if s.label("engine_type") == "decode" and s.name in PROMPT:
            continue
        if s.name in PROMPT:
            prompt += s.value
        elif s.name in GENERATION:
            generation += s.value
        else:
            cached += s.value
        if s.label("model_name"):
            models.add(s.label("model_name"))
    return Reading(prompt, generation, cached, frozenset(models))


def diff(before: dict, after: dict) -> OracleDelta:
    """Delta per pod. A lower value means the pod restarted: the new value is the delta."""
    d = OracleDelta()
    for pod, a in after.items():
        b = before.get(pod)
        if b is None:
            d.warnings.append(f"pod {pod} appeared during the case; its whole counter is counted")
            b = Reading(0, 0, 0, a.models)
        dp = a.prompt - b.prompt if a.prompt >= b.prompt else a.prompt
        dg = a.generation - b.generation if a.generation >= b.generation else a.generation
        dc = a.cached - b.cached if a.cached >= b.cached else a.cached
        if a.prompt < b.prompt or a.generation < b.generation:
            d.warnings.append(f"pod {pod} counter reset during the case")
        d.per_pod[pod] = (dp, dg)
        d.prompt += dp
        d.generation += dg
        d.cached += dc
    for pod in before:
        if pod not in after:
            d.warnings.append(f"pod {pod} disappeared during the case; its delta is unknown")
    return d


class EngineOracle:
    """Sums the counters of every pod in one or more PodForwards groups."""

    def __init__(self, *groups: PodForwards):
        self.groups = groups

    def _read(self, fw) -> Reading:
        last: Exception | None = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{fw.port}/metrics")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return reading_from_text(resp.read().decode())
            except (OSError, ValueError) as e:
                last = e
                fw.bounce()
                time.sleep(3)
        raise RuntimeError(f"engine {fw.ns}/{fw.target} metrics unreadable: {last}")

    def snapshot(self) -> dict:
        out = {}
        for g in self.groups:
            for pod in sorted(g.forwards):
                out[f"{g.ns}/{pod}"] = self._read(g.forwards[pod])
        return out

    def wait_idle(self, quiet_s: float = 10, timeout: float = 240) -> dict:
        """Wait until no engine counter has moved for `quiet_s` (aborted requests finish)."""
        deadline = time.monotonic() + timeout
        last = self.snapshot()
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(2)
            cur = self.snapshot()
            if cur != last:
                last, quiet_since = cur, time.monotonic()
            elif time.monotonic() - quiet_since >= quiet_s:
                return cur
        return last
