from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InteractionTarget:
    usd: str
    kind: str
    prim_path: str | None = None
    prim_name: str | None = None


@dataclass(frozen=True)
class InteractionSpec:
    id: str
    batch: str
    module: str
    target: InteractionTarget
    physics: dict[str, Any]
    usd: dict[str, Any]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "InteractionSpec":
        target = data.get("target", {})
        return InteractionSpec(
            id=str(data["id"]),
            batch=str(data["batch"]),
            module=str(data["module"]),
            target=InteractionTarget(
                usd=str(target["usd"]),
                kind=str(target.get("kind", "joint")),
                prim_path=target.get("prim_path"),
                prim_name=target.get("prim_name"),
            ),
            physics=dict(data.get("physics", {})),
            usd=dict(data.get("usd", {})),
        )


@dataclass
class ApplyResult:
    id: str
    batch: str
    usd: str
    module: str
    status: str
    prim_path: str | None = None
    changed_attrs: list[str] | None = None
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "batch": self.batch,
            "usd": self.usd,
            "module": self.module,
            "status": self.status,
            "prim_path": self.prim_path,
            "changed_attrs": self.changed_attrs or [],
            "warnings": self.warnings or [],
            "errors": self.errors or [],
        }


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return workspace_root().parent
