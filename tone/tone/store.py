"""One on-disk format for windows, so every command reads the same thing.

JSONL, one window per line. The export parser writes it, the simulator writes
it, and `score` / `weights` / `validate` / `fixtures` read it. That means the
500 MB XML is parsed exactly once and every later command starts from a file you
can open in a text editor and check by eye.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .score import Window


def to_json(w: Window) -> dict:
    return {
        "start": w.start.isoformat(),
        "source": w.source,
        "on_demand": w.on_demand,
        "quantized": w.quantized,
        "apple_sdnn": w.apple_sdnn,
        "rr_ms": [round(float(v), 4) for v in np.asarray(w.rr)],
    }


def from_json(data: dict) -> Window:
    return Window(
        start=datetime.fromisoformat(data["start"]),
        rr=np.asarray(data.get("rr_ms", []), dtype=float),
        source=data.get("source", "unknown"),
        on_demand=bool(data.get("on_demand", False)),
        quantized=bool(data.get("quantized", False)),
        apple_sdnn=data.get("apple_sdnn"),
    )


def save(windows: list[Window], path: str | Path) -> None:
    with Path(path).open("w") as fh:
        for w in sorted(windows, key=lambda w: w.start):
            fh.write(json.dumps(to_json(w)) + "\n")


def load(path: str | Path) -> list[Window]:
    out: list[Window] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(from_json(json.loads(line)))
    out.sort(key=lambda w: w.start)
    return out
