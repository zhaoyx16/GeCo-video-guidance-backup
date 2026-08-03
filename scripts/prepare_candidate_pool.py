#!/usr/bin/env python3
"""Extract VGGT-Omega geometry and freeze a candidate-pool manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_selection.backbones.vggt_omega import VGGTOmegaAdapter, file_sha256
from geometry_selection.appearance import score_appearance_pair
from geometry_selection.cache import canonical_hash, load_geometry_cache, save_geometry_cache
from geometry_selection.selection import (
    load_and_validate_candidate_pool,
    materialize_candidate_pool,
    validate_candidate_spec,
)
from geometry_selection.protocol import (
    FORMAL_SPLIT_COUNTS,
    file_sha256 as protocol_file_sha256,
    validate_candidate_spec_against_protocol,
    validate_committed_file,
    validate_experiment_lock,
    validate_formal_protocol,
)
from geometry_selection.model_lock import load_model_lock
from geometry_selection.window_bundle import (
    independent_run_id,
    make_window_bundle,
)
from geometry_selection.window_graph import make_window_schedule
from scripts.extract_vggt_omega_geometry import (
    decode_video_frames,
    select_keyframes,
    video_frame_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "external/vggt_omega",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-mode",
        choices=("formal", "legacy-debug"),
        required=True,
    )
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--experiment-lock", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--num-keyframes", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--preprocessing-mode",
        choices=("balanced", "max_size"),
        default="balanced",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidate pool: {args.output}")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    validate_candidate_spec(spec)
    code_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    formal = args.artifact_mode == "formal"
    model_lock = None
    experiment_lock_sha256 = None
    implementation_sha256 = None
    if formal:
        if args.expected_git_commit is None:
            parser.error("formal mode requires --expected-git-commit")
        if dirty or code_commit != args.expected_git_commit:
            raise RuntimeError(
                f"geometry producer is not the expected clean commit: "
                f"commit={code_commit}, dirty={dirty}"
            )
        validate_committed_file(args.protocol_manifest, REPO_ROOT, code_commit)
        if args.model_lock is None:
            parser.error("formal mode requires --model-lock")
        validate_committed_file(args.model_lock, REPO_ROOT, code_commit)
        protocol = validate_formal_protocol(args.protocol_manifest)
        if protocol_file_sha256(args.model_lock) != protocol["_meta"]["model_lock_sha256"]:
            raise ValueError("model lock digest differs from frozen protocol")
        model_lock = load_model_lock(args.model_lock)
        if args.experiment_lock is None:
            parser.error("formal mode requires --experiment-lock")
        if args.artifact_root is None:
            parser.error("formal mode requires --artifact-root")
        experiment_lock = validate_experiment_lock(
            args.experiment_lock,
            REPO_ROOT,
            code_commit,
            protocol_manifest_sha256=protocol_file_sha256(args.protocol_manifest),
            model_lock_sha256=protocol_file_sha256(args.model_lock),
            split=spec["split"],
            backbone=spec["backbone"],
            candidate_spec_sha256=protocol_file_sha256(args.spec),
            artifact_root=args.artifact_root,
        )
        experiment_lock_sha256 = protocol_file_sha256(args.experiment_lock)
        implementation_sha256 = experiment_lock["implementation_sha256"]
        artifact_root = args.artifact_root.resolve()
        for path in (args.output.resolve(), args.cache_root.resolve()):
            try:
                path.relative_to(artifact_root)
            except ValueError as error:
                raise ValueError(f"formal output/cache path escapes artifact root: {path}") from error
        for case in spec["cases"]:
            for candidate in case["candidates"]:
                try:
                    Path(candidate["video"]).resolve().relative_to(artifact_root)
                except ValueError as error:
                    raise ValueError("formal candidate video escapes artifact root") from error
    validate_candidate_spec_against_protocol(
        spec,
        args.protocol_manifest,
        args.dataset_root,
        formal=formal,
        expected_git_commit=code_commit if formal else None,
        model_lock=model_lock,
        experiment_lock_sha256=experiment_lock_sha256,
        expected_implementation_sha256=implementation_sha256,
        candidate_spec_sha256=protocol_file_sha256(args.spec) if formal else None,
    )
    adapter = VGGTOmegaAdapter(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        device=args.device,
        image_resolution=args.image_resolution,
        preprocessing_mode=args.preprocessing_mode,
    )
    backbone_identity = adapter.cache_identity()
    if formal:
        expected_geometry = model_lock["geometry_backbone"]
        geometry_mismatches = {
            key: (backbone_identity.get(key), expected)
            for key, expected in expected_geometry.items()
            if backbone_identity.get(key) != expected
        }
        if geometry_mismatches:
            raise ValueError(f"geometry backbone identity mismatch: {geometry_mismatches}")
    producer_identity = {
        "commit": code_commit,
        "dirty": dirty,
        "adapter_sha256": file_sha256(REPO_ROOT / "geometry_selection/backbones/vggt_omega.py"),
        "extractor_sha256": file_sha256(REPO_ROOT / "scripts/extract_vggt_omega_geometry.py"),
        "preparer_sha256": file_sha256(Path(__file__).resolve()),
        "appearance_sha256": file_sha256(REPO_ROOT / "geometry_selection/appearance.py"),
        "window_graph_sha256": file_sha256(REPO_ROOT / "geometry_selection/window_graph.py"),
        "window_bundle_sha256": file_sha256(REPO_ROOT / "geometry_selection/window_bundle.py"),
        "experiment_lock_sha256": experiment_lock_sha256,
        "implementation_sha256": implementation_sha256,
    }
    geometry_keys: dict[tuple[str, str], str] = {}
    geometry_window_bundles: dict[tuple[str, str], dict] = {}
    extraction_config = spec.get("geometry_extraction")
    for case in spec.get("cases", []):
        for candidate in case.get("candidates", []):
            video = Path(candidate["video"]).resolve()
            video_hash = file_sha256(video)
            num_keyframes = (
                extraction_config["num_keyframes"]
                if extraction_config is not None
                else args.num_keyframes
            )
            indices = select_keyframes(video_frame_count(video), num_keyframes)
            with tempfile.TemporaryDirectory(prefix="vggt_omega_frames_") as directory:
                image_paths, decode_identity = decode_video_frames(
                    video,
                    indices,
                    Path(directory),
                )
                provenance = {
                    "video_sha256": video_hash,
                    "keyframe_indices": indices,
                    "decoded_frames": decode_identity,
                    "geometry_backbone": backbone_identity,
                    "producer": producer_identity,
                    "extractor_schema_version": 2,
                }
                cache_key = canonical_hash(provenance)
                try:
                    load_geometry_cache(
                        args.cache_root,
                        cache_key,
                        expected_provenance=provenance,
                    )
                    status = "cache_hit"
                except FileNotFoundError:
                    prediction = adapter.predict_image_paths(
                        image_paths,
                        keyframe_indices=indices,
                        hash_checkpoint=True,
                    )
                    save_geometry_cache(
                        args.cache_root,
                        cache_key,
                        prediction,
                        provenance,
                    )
                    status = "computed"
                if extraction_config is not None:
                    paths_by_frame = dict(zip(indices, image_paths, strict=True))
                    pixels_by_frame = {
                        int(record["index"]): record["pixels_sha256"]
                        for record in decode_identity["frames"]
                    }
                    files_by_frame = {
                        index: file_sha256(path)
                        for index, path in paths_by_frame.items()
                    }
                    schedule = make_window_schedule(
                        indices,
                        local_window_size=extraction_config["local_window_size"],
                        local_stride=extraction_config["local_stride"],
                        loop_context=extraction_config["loop_context"],
                        min_loop_node_gap=extraction_config["min_loop_node_gap"],
                        max_loop_windows=extraction_config["max_loop_windows"],
                    )
                    window_records = []
                    for window_id, window_kind, frame_indices in schedule:
                        frame_pixels = [pixels_by_frame[index] for index in frame_indices]
                        frame_files = [files_by_frame[index] for index in frame_indices]
                        appearance_evidence = None
                        if window_kind == "loop":
                            appearance_evidence = score_appearance_pair(
                                paths_by_frame[frame_indices[0]],
                                paths_by_frame[frame_indices[-1]],
                                source_frame=frame_indices[0],
                                target_frame=frame_indices[-1],
                            ).to_dict()
                        run_id = independent_run_id(
                            video_sha256=video_hash,
                            window_id=window_id,
                            kind=window_kind,
                            frame_indices=frame_indices,
                            frame_pixels_sha256=frame_pixels,
                            frame_file_sha256=frame_files,
                            geometry_backbone=backbone_identity,
                            producer=producer_identity,
                        )
                        window_provenance = {
                            "video_sha256": video_hash,
                            "window_id": window_id,
                            "window_kind": window_kind,
                            "keyframe_indices": list(frame_indices),
                            "frame_pixels_sha256": frame_pixels,
                            "frame_file_sha256": frame_files,
                            "independent_run_id": run_id,
                            "decoder": {
                                "decoder": decode_identity["decoder"],
                                "opencv_version": decode_identity["opencv_version"],
                                "backend": decode_identity["backend"],
                            },
                            "geometry_backbone": backbone_identity,
                            "producer": producer_identity,
                            "extractor_schema_version": 4,
                        }
                        window_cache_key = canonical_hash(window_provenance)
                        try:
                            load_geometry_cache(
                                args.cache_root,
                                window_cache_key,
                                expected_provenance=window_provenance,
                            )
                            window_status = "cache_hit"
                        except FileNotFoundError:
                            window_prediction = adapter.predict_image_paths(
                                [paths_by_frame[index] for index in frame_indices],
                                keyframe_indices=frame_indices,
                                hash_checkpoint=True,
                            )
                            save_geometry_cache(
                                args.cache_root,
                                window_cache_key,
                                window_prediction,
                                window_provenance,
                            )
                            window_status = "computed"
                        window_records.append(
                            {
                                "window_id": window_id,
                                "kind": window_kind,
                                "frame_indices": list(frame_indices),
                                "frame_pixels_sha256": frame_pixels,
                                "frame_file_sha256": frame_files,
                                "appearance_evidence": appearance_evidence,
                                "geometry_cache_key": window_cache_key,
                                "independent_run_id": run_id,
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "case": case["case_id"],
                                    "candidate": candidate["candidate_id"],
                                    "window": window_id,
                                    "status": window_status,
                                }
                            )
                        )
            key = (case["case_id"], candidate["candidate_id"])
            geometry_keys[key] = cache_key
            if extraction_config is not None:
                geometry_window_bundles[key] = make_window_bundle(
                    video_sha256=video_hash,
                    global_geometry_cache_key=cache_key,
                    global_keyframe_indices=indices,
                    extraction_config=extraction_config,
                    window_records=window_records,
                )
            print(json.dumps({"case": key[0], "candidate": key[1], "status": status}))

    pool = materialize_candidate_pool(
        spec,
        geometry_keys,
        artifact_mode=args.artifact_mode,
        producer_identity=producer_identity,
        candidate_spec_sha256=file_sha256(args.spec.resolve()),
        geometry_window_bundles=(
            geometry_window_bundles if extraction_config is not None else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(pool, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_and_validate_candidate_pool(
        temporary,
        expected_split=spec["split"],
        verify_video_hashes=True,
        expected_case_count=FORMAL_SPLIT_COUNTS[spec["split"]] if formal else None,
    )
    try:
        os.link(temporary, args.output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite frozen candidate pool: {args.output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "cases": len(pool["cases"])}, indent=2))


if __name__ == "__main__":
    main()
