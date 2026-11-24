"""Case registry. A case is a function `fn(lab) -> CaseResult | list[CaseResult]`."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

CASES: dict[str, "Case"] = {}


@dataclass
class Case:
    id: str
    title: str
    fn: Callable
    phase: int
    requires: tuple[str, ...]
    disruptive: bool


def case(id: str, title: str, *, phase: int = 1, requires: tuple[str, ...] = (),
         disruptive: bool = False):
    def deco(fn):
        CASES[id] = Case(id, title, fn, phase, requires, disruptive)
        return fn
    return deco


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def ordered(selected: list[Case]) -> list[Case]:
    """Phase 0 first, then Phase 1; disruptive cases last so a pod restart cannot starve the rest."""
    return sorted(selected, key=lambda c: (c.phase, c.disruptive, _natural(c.id)))
