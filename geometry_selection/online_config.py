"""Strict, committed configuration for online geometry selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .cache import canonical_hash
from .graph import PoseGraphConfig
from .graph_scorer import GraphScoreConfig
from .online import OnlineBranchConfig
from .scorer import ScorerConfig
from .selection import SelectionConfig
from .window_bundle import validate_geometry_extraction_config
from .window_graph import WindowGraphConfig


ONLINE_SELECTION_CONFIG_SCHEMA_VERSION = 1


def _strict_dataclass(cls, payload: dict[str, Any]):
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} config must be a mapping")
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return cls(**payload)


@dataclass(frozen=True)
class OnlineVGGTOmegaConfig:
    source_root: str
    checkpoint: str
    device: str = "cuda"
    image_resolution: int = 512
    preprocessing_mode: str = "balanced"

    def validate(self) -> None:
        if not self.source_root or not self.checkpoint:
            raise ValueError("VGGT-Omega source_root and checkpoint must be non-empty")
        if not self.device:
            raise ValueError("VGGT-Omega device must be non-empty")
        if self.image_resolution <= 0 or self.image_resolution % 16:
            raise ValueError("VGGT-Omega image_resolution must be a positive multiple of 16")
        if self.preprocessing_mode not in {"balanced", "max_size"}:
            raise ValueError("VGGT-Omega preprocessing_mode is invalid")


@dataclass(frozen=True)
class OnlineProvisionalConfig:
    retain_frames: bool = False

    def validate(self) -> None:
        if not isinstance(self.retain_frames, bool):
            raise TypeError("retain_frames must be boolean")


@dataclass(frozen=True)
class OnlineSelectionRunConfig:
    schema_version: int
    method_version: str
    branch: OnlineBranchConfig
    geometry_extraction: dict[str, Any]
    scorer: ScorerConfig
    graph_score: GraphScoreConfig
    selection: SelectionConfig
    vggt_omega: OnlineVGGTOmegaConfig
    provisional: OnlineProvisionalConfig

    def validate(self, *, num_steps: int | None = None) -> None:
        if self.schema_version != ONLINE_SELECTION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported online selection config schema "
                f"{self.schema_version}; expected {ONLINE_SELECTION_CONFIG_SCHEMA_VERSION}"
            )
        if not self.method_version:
            raise ValueError("method_version must be non-empty")
        self.branch.validate()
        validate_geometry_extraction_config(self.geometry_extraction)
        self.scorer.validate()
        self.graph_score.validate()
        self.selection.validate()
        self.vggt_omega.validate()
        self.provisional.validate()
        if self.geometry_extraction["num_keyframes"] < 4:
            raise ValueError("online geometry requires at least four keyframes")
        if num_steps is not None:
            if num_steps < 3:
                raise ValueError("online selection requires at least three denoising steps")
            if self.branch.selection_steps[-1] >= num_steps - 1:
                raise ValueError(
                    "online selection cannot run on the final denoising step"
                )

    def semantic_dict(self) -> dict[str, Any]:
        graph = asdict(self.graph_score)
        graph["window"] = asdict(self.graph_score.window)
        graph["optimizer"] = asdict(self.graph_score.optimizer)
        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "branch": {
                **asdict(self.branch),
                "selection_steps": list(self.branch.selection_steps),
            },
            "geometry_extraction": dict(self.geometry_extraction),
            "score": {**asdict(self.scorer), "local_offsets": list(self.scorer.local_offsets)},
            "graph_score": graph,
            "selection": asdict(self.selection),
            "vggt_omega": {
                "image_resolution": self.vggt_omega.image_resolution,
                "preprocessing_mode": self.vggt_omega.preprocessing_mode,
            },
            "provisional": asdict(self.provisional),
        }

    def resolved_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "vggt_omega_paths": asdict(self.vggt_omega),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.semantic_dict())


def load_online_selection_config(path: Path) -> OnlineSelectionRunConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("top-level online selection config must be a mapping")
    expected = {
        "schema_version",
        "method_version",
        "branch",
        "geometry_extraction",
        "score",
        "graph_score",
        "selection",
        "vggt_omega",
        "provisional",
    }
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        raise ValueError(f"online config keys mismatch; missing={missing}, unknown={unknown}")
    branch_payload = dict(payload["branch"])
    branch_payload["selection_steps"] = tuple(branch_payload.get("selection_steps", ()))
    branch = _strict_dataclass(OnlineBranchConfig, branch_payload)

    scorer_payload = dict(payload["score"])
    scorer_payload["local_offsets"] = tuple(scorer_payload.get("local_offsets", ()))
    scorer = _strict_dataclass(ScorerConfig, scorer_payload)
    graph_payload = dict(payload["graph_score"])
    window = _strict_dataclass(WindowGraphConfig, dict(graph_payload.pop("window")))
    optimizer = _strict_dataclass(PoseGraphConfig, dict(graph_payload.pop("optimizer")))
    graph = _strict_dataclass(
        GraphScoreConfig,
        {**graph_payload, "window": window, "optimizer": optimizer},
    )
    config = OnlineSelectionRunConfig(
        schema_version=payload["schema_version"],
        method_version=payload["method_version"],
        branch=branch,
        geometry_extraction=dict(payload["geometry_extraction"]),
        scorer=scorer,
        graph_score=graph,
        selection=_strict_dataclass(SelectionConfig, dict(payload["selection"])),
        vggt_omega=_strict_dataclass(OnlineVGGTOmegaConfig, dict(payload["vggt_omega"])),
        provisional=_strict_dataclass(OnlineProvisionalConfig, dict(payload["provisional"])),
    )
    config.validate()
    return config
