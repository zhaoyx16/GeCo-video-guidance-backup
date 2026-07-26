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
manifest, and rejects a scene or sequence assigned to different splits. A
generation condition must name the split-manifest ID and its SHA-256. The
referenced manifest must be present in the same run manifest and must assign
that exact dataset/scene/sequence tuple to the claimed split.

The frozen split manifest is therefore created before method tuning. Do not
create a new split to accommodate a method result.

## Paired Condition

The condition mapping is hashed as condition_hash. It includes:

- source clip interval, native FPS, source hash, intrinsics hash, and pose hash;
- first, middle, and last image-anchor paths, source timestamps, and hashes;
- the frozen split-manifest reference;
- static-scene eligibility and an explicit scene/sequence statistical unit;
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
declared native FPS and time origin. frame_guidance.generated_timing then maps
the three source anchors to generated frame 0, generated floor midpoint, and
generated final frame. It also records both source and generated timestamps.

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
VGGT/UFM. It must not be described as an independent evaluator. Independent
geometry or trajectory evaluators must state independence_policy=independent.

Trajectory evaluation accepts only bound pose artifacts. A reference artifact
must hash-match source_clip.poses_ref. A predicted artifact must name the
generated video SHA-256. Both must expose the contractual anchors, explicit
W2C or C2W convention, and translation units. Raw arbitrary pose matrices are
rejected by the public trajectory interface.

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

Create a frozen split record with with_split_manifest_hash, then calculate a
condition hash with with_condition_hash. After a generation completes, add the
execution and output hash and call with_record_hash.

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
