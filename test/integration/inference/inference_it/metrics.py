"""Prometheus text-format parsing and counter deltas."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]   # sorted, hashable
    value: float

    def label(self, key: str, default: str | None = None) -> str | None:
        return dict(self.labels).get(key, default)


def parse(text: str, names: set[str] | None = None) -> list[Sample]:
    out = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE.match(line)
        if not m or (names is not None and m.group(1) not in names):
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        labels = tuple(sorted(_LABEL.findall(m.group(2) or "")))
        out.append(Sample(m.group(1), labels, value))
    return out


Series = dict  # {(name, labels_tuple): value}


def to_series(samples: list[Sample]) -> Series:
    """Collapse to a dict. Duplicate keys (several replicas) are summed."""
    out: Series = {}
    for s in samples:
        k = (s.name, s.labels)
        out[k] = out.get(k, 0.0) + s.value
    return out


@dataclass(frozen=True)
class Row:
    """A counter series whose value changed between two scrapes."""
    metric: str
    labels: dict
    delta: float


def delta_rows(before: Series, after: Series) -> list[Row]:
    """Per-series increase. A series that went down is a counter reset: its new value counts."""
    rows = []
    for key, value in after.items():
        prev = before.get(key, 0.0)
        d = value - prev if value >= prev else value
        if d:
            rows.append(Row(key[0], dict(key[1]), d))
    return rows
