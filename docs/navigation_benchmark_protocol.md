# Controlled Navigation Benchmark Protocol

This package defines the experiment record used for future comparisons of:

- Frame Guidance only;
- Frame Guidance plus the current RGB-GeCo variant; and
- Frame Guidance plus a future latent-geometry method.

It is deliberately protocol-only. It does not modify a generation pipeline,
Frame Guidance, or a latent critic.

## What Is Frozen

Every benchmark manifest starts with one or more frozen split_manifest records.
Each assignment contains dataset_id, scene_id, sequence_id, and split. The
validator rejects a scene or sequence that appears in more than one frozen
manifest, and rejects a scene or sequence assigned to different splits. A split
must also name an immutable external registry artifact (`uri`, `version`,
`registry_id`, `artifact_path`, and `expected_sha256`). The artifact must be a
JSON file packaged with the final benchmark manifest; the validator checks its
bytes, registry metadata, and exact assignments. This external pin is
deliberately not derived from the run manifest itself. A generation condition must name the
split-manifest ID, its SHA-256, and the pinned external split-artifact hash.
The referenced manifest must be present in the same run manifest and must
assign that exact dataset/scene/sequence tuple to the claimed split.

The frozen split manifest is therefore created before method tuning. Do not
create a new split to accommodate a method result. The validator does not fetch
registries over the network: it resolves a relative `artifact_path` against the
manifest directory, so the checked JSON artifact must travel with the manifest.

## Paired Condition

The condition mapping is hashed as condition_hash. It includes:

- source clip interval, native FPS, source hash, intrinsics hash, and pose hash;
- first, middle, and last image-anchor paths, source timestamps, and hashes;
- the frozen split-manifest reference;
- static-scene eligibility and a deterministic scene/sequence statistical unit;
- prompt, seed, model ID/revision/config, and complete sampler config;
- Frame Guidance settings; and
- source-to-generated frame and timestamp mapping.

For any pair_id every expected method must have exactly the same condition
hash. Only method, output, and execution provenance may differ. This means a
same-seed RGB-GeCo comparison is a controlled ablation, but trajectory control
still has to be reported separately: the generated motion must not get smaller
to obtain a better geometry score.

## Anchor Contract

This v1 protocol fixes the anchor rule rather than accepting any three frames:

1. first is source_clip.start_frame;
2. middle is start_frame + floor((end_frame - start_frame) / 2);
3. last is source_clip.end_frame.

Anchors must be stored in that order. Source timestamps are checked against the
declared native FPS and time origin. The generated video must contain at least
three frames, and frame_guidance.generated_timing then maps the three source
anchors to generated frame 0, generated floor midpoint, and generated final
frame. It also records both source and generated timestamps. The three generated
anchors must be distinct.

This is the contract used by a future Frame Guidance implementation and by
trajectory evaluation; it avoids a result being evaluated on a different set
of frames from the ones used as trajectory anchors.

## Completed Runs and Metrics

A completed generation_run must include:

- immutable git commit, full normalized condition hash, method parameters, and
  mechanism/version;
- output video URI and SHA-256;
- device-role mapping, runtime, and peak allocated/reserved VRAM per device;
- a record_hash calculated over the completed record; and
- source/anchor/model/checkpoint provenance already held in the condition.

A metric_result is bound to a completed run via both run_record_hash and
evaluated_output_sha256. It must also have a SHA-256 for the metric output
artifact, plus an evaluator descriptor with name, version, model/checkpoint,
full config, independence policy, and evaluator fingerprint.

GeCo-Eval is useful, but it is guidance_aligned when RGB-GeCo itself uses
VGGT/UFM. It must not be described as an independent evaluator. The validator
therefore only accepts `guidance_aligned` for the guidance-aligned metric role;
`independent_geometry` and `trajectory_adherence` require
`independence_policy=independent`. Motion preservation and visual quality may
use an independent evaluator or a human-annotation protocol.

Trajectory evaluation accepts only bound pose artifacts. A reference artifact
must hash-match source_clip.poses_ref. A predicted artifact must name the
generated video SHA-256. Both must expose the contractual anchors, explicit
W2C or C2W convention, and translation units. Final trajectory metric records
are checked again against the completed run's condition hash, source pose hash,
output hash, anchor-map hash, convention, and units. Raw arbitrary pose matrices
are rejected by the public trajectory interface.

## Current RGB Reference Identity

The current RGB path is a flow-matching RGB-GeCo variant, not an assertion of
exact original GeCo equivalence. Its method.mechanism must record at least:

    {
      "id": "rgb_geco_flow_matching_x0",
      "version": "...",
      "time_travel": "absent",
      "temporal_vae_context": "past_only_approximate",
      "guidance_schedule_state": "active_guidance"
    }

guidance_schedule_state distinguishes a true all-zero schedule from a positive
schedule with zero learning rate. They can be visually identical under a
correct implementation, but they exercise different sampling paths and must
not be merged in provenance.

## Fail-Closed Aggregation

Final aggregation requires all expected arms, all completed and hash-valid,
with a result for the requested metric. It rejects missing, failed, duplicate,
or unbound arms rather than skipping them. All selected metric results must use
the same evaluator fingerprint and config.

The declared independent unit is condition.scene.statistical_unit. Typical
navigation data should use a sequence cluster when multiple clips or seeds
share one source sequence. Confidence intervals use a cluster bootstrap, not a
frame-level or seed-level bootstrap. At least two independent clusters are
required; otherwise the protocol rejects the confidence interval instead of
reporting a misleading one.

## Commands

Create and publish the external split-assignment JSON artifact first. It must
contain `registry_id`, `version`, and `assignments`. Then create a frozen split
record with its URI/version/registry ID/local artifact path/SHA-256 and call
with_split_manifest_hash. Calculate a condition hash with with_condition_hash.
After a generation completes, add the execution and output hash and call
with_record_hash.

Validate a final three-arm manifest:

    python -m navigation_benchmark validate \
      --manifest runs.jsonl \
      --expected-method fg_only \
      --expected-method fg_rgb_geco \
      --expected-method fg_latent_geometry \
      --require-static-scene \
      --require-completed

Evaluate externally estimated poses only after writing bound source and output
pose artifacts:

    python -m navigation_benchmark trajectory \
      --manifest runs.jsonl \
      --run-id example-run \
      --reference-pose-artifact reference_pose_artifact.json \
      --predicted-pose-artifact predicted_pose_artifact.json \
      --metric-artifact-uri artifacts://pose-eval/example-run.json \
      --metric-artifact-sha256 <sha256> \
      --metric-output trajectory_metrics.jsonl \
      --evaluator-name external-pose-estimator \
      --evaluator-version v1 \
      --evaluator-model-id model-id \
      --evaluator-checkpoint-revision revision \
      --evaluator-config-json '{"config_key":"value"}'

Aggregate one candidate against the reference while requiring every planned
arm in the pair:

    python -m navigation_benchmark aggregate \
      --manifest runs.jsonl \
      --metrics metrics.jsonl \
      --baseline-method fg_only \
      --candidate-method fg_rgb_geco \
      --expected-method fg_only \
      --expected-method fg_rgb_geco \
      --expected-method fg_latent_geometry \
      --metric independent_pose_rmse

## External Requirements

The protocol cannot manufacture dataset ground truth. Before a final study it
still needs frozen source clips, intrinsics, poses, frame artifacts, static
eligibility labels, an independent evaluator, and a content-addressed artifact
store or stable URI scheme. The JSON template is illustrative only; replace
all placeholder hashes and compute the real fingerprints before validation.

## Independent Epipolar Pilot

The package also provides a CPU-only classical diagnostic:

    python -m navigation_benchmark epipolar \
      --video fg_only=/absolute/path/fg_only.mp4 \
      --video candidate=/absolute/path/candidate.mp4 \
      --lags-sec 0.5,1.0 \
      --output epipolar_report.json

It uses OpenCV SIFT correspondences and a MAGSAC fundamental matrix. It does
not use VGGT, UFM, generator features, or a learned depth model. The report
keeps geometry and motion separate:

- feature and reciprocal-match coverage are reported before geometry;
- `parallax_observable_fraction` excludes low-motion and
  homography-degenerate pairs;
- the fundamental matrix is fit on one subset of reciprocal matches and
  scored on held-out matches;
- a capped, failure-aware Sampson error prevents failed F estimates from
  disappearing from the aggregate; and
- low-motion pairs explicitly abstain instead of receiving a good geometry
  score.

This diagnostic is an independent metric family, but it is not sufficient on
its own. Textureless regions, repeated patterns, disocclusion, and non-rigid
objects can make two-view estimation fail. Final claims still require the
GT-bound trajectory metric, an additional temporal/track diagnostic, visual
quality, and the predeclared static-scene eligibility rule.

Before using it in a method table, calibrate it on a real source clip plus
known controls:

    python scripts/calibrate_epipolar_metric.py \
      --reference_frames '/dataset/sequence/image_2/0000*.png' \
      --reference_fps 10 \
      --reference_output_width 1280 \
      --reference_output_height 704 \
      --reference_resize_mode stretch \
      --video fg_only=/absolute/path/fg_only.mp4 \
      --output_json calibration.json \
      --output_csv calibration.csv \
      --control_video_dir calibration_controls

The calibration first encodes every control with H.264 and then evaluates the
decoded video, matching the generated-video path. It includes a frozen-video
control and a time-varying non-rigid warp. A frozen control must lose parallax
observability rather than appearing geometrically superior. The non-rigid
control should worsen robust correspondence statistics relative to the
unmodified source clip; otherwise the configuration is not sensitive enough
for the target failure mode. The reference resize mode must match the
generator's actual input/anchor preprocessing; the current Wan
`VideoProcessor` pilot uses `stretch`, not a center crop. Requested time lags
must be exactly representable at every compared FPS within
`max_lag_error_sec`; use 0.5 and 1.0 seconds for the current 10/16/24 FPS
protocol.
