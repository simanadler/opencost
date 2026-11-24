"""Helpers shared by the case modules."""
from __future__ import annotations

from ..lab import Lab, Outcome
from ..config import LABEL_TENANT, LABEL_USER, LABEL_WORKLOAD, LABEL_MODEL, SENTINEL


def exact(out: Outcome, **labels) -> tuple[bool, float, float]:
    """Attributed tokens for the label set equal the engine oracle exactly (plan Section 5.2)."""
    i, o = out.attributed(**labels)
    return (i == out.e_in and o == out.e_out), i, o


def outside(out: Outcome, **labels) -> float:
    """Tokens counted under any series that does NOT match the labels (cross-attribution)."""
    total = 0.0
    for r in out.rows:
        if not all(r.labels.get(k) == v for k, v in labels.items()):
            total += r.delta
    return total


def ident(tenant, workload, user=None, model=None) -> dict:
    d = {LABEL_TENANT: tenant, LABEL_WORKLOAD: workload}
    if user is not None:
        d[LABEL_USER] = user
    if model is not None:
        d[LABEL_MODEL] = model
    return d


def traffic_resilient(lab: Lab, alias: str, key: str, **kw) -> tuple[Outcome, list[str]]:
    """Run traffic; if the engine rejects the vLLM-only length-forcing fields (HTTP 400), repeat
    without them. The oracle, not the forced length, is the authority for E_out."""
    notes: list[str] = []
    out = lab.traffic(alias, key, **kw)
    if any(s == 400 for s in out.statuses) and kw.get("force_length", True):
        notes.append("engine rejected min_tokens/ignore_eos; repeated without forced length")
        kw["force_length"] = False
        out = lab.traffic(alias, key, **kw)
    return out, notes


def summarize(out: Outcome) -> dict:
    return {"E_in": out.e_in, "E_out": out.e_out, "client_in": out.client_in,
            "client_out": out.client_out, "statuses": out.statuses,
            "series": out.series(), "oracle_warnings": out.oracle.warnings,
            "per_pod": out.oracle.per_pod}


__all__ = ["exact", "outside", "ident", "traffic_resilient", "summarize", "SENTINEL", "Lab"]
