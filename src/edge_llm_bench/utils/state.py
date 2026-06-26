"""Resume state — what configs are done, what failed, what to skip."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunState:
    """Tracks per-config completion so ``--resume`` can skip finished work."""

    run_id: str
    started_at: str
    completed: dict[str, str] = field(default_factory=dict)  # config_id → status
    notes: dict[str, str] = field(default_factory=dict)  # config_id → free-form
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> RunState:
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def load_or_new(cls, path: Path, run_id: str, started_at: str) -> RunState:
        if path.exists():
            return cls.load(path)
        return cls(run_id=run_id, started_at=started_at)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX; close-enough on Windows

    def is_done(self, config_id: str) -> bool:
        return self.completed.get(config_id) == "ok"

    def mark(self, config_id: str, status: str, note: str = "") -> None:
        self.completed[config_id] = status
        if note:
            self.notes[config_id] = note
