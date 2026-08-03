"""Strict YAML configuration for reproducible offline candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .cache import canonical_hash
from .scorer import ScorerConfig
from .selection import SelectionConfig


CONFIG_SCHEMA_VERSION = 1


def _construct_strict(cls, payload: dict[str, Any]):
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} config must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return cls(**payload)


@dataclass(frozen=True)
class OfflineRankingConfig:
    schema_version: int
    method_version: str
    experiment_name: str
    candidate_manifest: str
    protocol_manifest: str
    dataset_root: str
    geometry_cache_root: str
    geometry_checkpoint_sha256: str
    geometry_source_tree_sha256: str
    geometry_source_commit: str
    output_root: str
    expected_split: str
    scorer: ScorerConfig
    selection: SelectionConfig

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config schema {self.schema_version}; expected {CONFIG_SCHEMA_VERSION}"
            )
        if not self.method_version or not self.experiment_name:
            raise ValueError("method_version and experiment_name must be non-empty")
        if self.expected_split not in {"debug", "validation", "test"}:
            raise ValueError("expected_split must be debug, validation, or test")
        for name in (
            "geometry_checkpoint_sha256",
            "geometry_source_tree_sha256",
            "geometry_source_commit",
        ):
            value = getattr(self, name)
            if len(value) != 64 and name != "geometry_source_commit":
                raise ValueError(f"{name} must be a SHA-256 digest")
            if name == "geometry_source_commit" and len(value) != 40:
                raise ValueError("geometry_source_commit must be a full Git commit")
        self.scorer.validate()
        self.selection.validate()

    def to_dict(self) -> dict[str, Any]:
        score = asdict(self.scorer)
        score["local_offsets"] = list(self.scorer.local_offsets)
        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "experiment_name": self.experiment_name,
            "candidate_manifest": self.candidate_manifest,
            "protocol_manifest": self.protocol_manifest,
            "dataset_root": self.dataset_root,
            "geometry_cache_root": self.geometry_cache_root,
            "geometry_checkpoint_sha256": self.geometry_checkpoint_sha256,
            "geometry_source_tree_sha256": self.geometry_source_tree_sha256,
            "geometry_source_commit": self.geometry_source_commit,
            "output_root": self.output_root,
            "expected_split": self.expected_split,
            "score": score,
            "selection": asdict(self.selection),
        }

    def semantic_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        for key in (
            "candidate_manifest",
            "protocol_manifest",
            "dataset_root",
            "geometry_cache_root",
            "output_root",
        ):
            payload.pop(key)
        return payload

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.semantic_dict())


def load_offline_ranking_config(path: Path) -> OfflineRankingConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("top-level YAML value must be a mapping")
    expected = {
        "schema_version",
        "method_version",
        "experiment_name",
        "candidate_manifest",
        "protocol_manifest",
        "dataset_root",
        "geometry_cache_root",
        "geometry_checkpoint_sha256",
        "geometry_source_tree_sha256",
        "geometry_source_commit",
        "output_root",
        "expected_split",
        "score",
        "selection",
    }
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        raise ValueError(f"config keys mismatch; missing={missing}, unknown={unknown}")

    score_payload = dict(payload.pop("score"))
    if "local_offsets" in score_payload:
        score_payload["local_offsets"] = tuple(score_payload["local_offsets"])
    scorer = _construct_strict(ScorerConfig, score_payload)
    selection = _construct_strict(SelectionConfig, dict(payload.pop("selection")))
    config = OfflineRankingConfig(scorer=scorer, selection=selection, **payload)
    config.validate()
    return config


def write_resolved_config(config: OfflineRankingConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8")
    temporary.replace(path)
