# Controlled Navigation Benchmark Protocol

This package records and validates the protocol for future comparisons between
Frame Guidance only, Frame Guidance plus RGB GeCo, and Frame Guidance plus a
latent geometry method. It does not generate videos or change a guided
pipeline.

## Core rule

Every arm in a paired comparison must have the same condition mapping. The
mapping is the normalized configuration: scene identity and split, source clip
interval and FPS, first/middle/last anchor paths and hashes, prompt, seed,
model revision, sampler, resolution, frame count, and Frame Guidance settings.
The only fields that may differ across arms are method, output, and execution.

The tooling calculates a SHA-256 condition hash over the full condition
mapping. A changed anchor, prompt, seed, scheduler, model revision or sampling
parameter therefore fails paired-condition validation.

source_clip.time_origin must state whether anchor timestamps are relative to
the selected clip or to the source sequence. This matters whenever start_frame
is nonzero.

Use paths as configurable URIs rather than machine-specific absolute paths.
Examples include dataset://DL3DV/..., relative paths, object-storage URIs, or
an experiment artifact URI.

## Required provenance

A completed run must record:

- the git commit;
- the method mechanism and version;
- the full normalized condition and matching condition hash;
- source, pose, intrinsics, and anchor hashes;
- model ID and checkpoint revision;
- seed;
- device-role mapping;
- per-device peak allocated and reserved VRAM;
- runtime in seconds; and
- output URI and optional output hash.

A completed record must also carry record_hash, a SHA-256 fingerprint of the
entire completed record apart from record_hash itself. This is a lightweight
tamper-evident immutable-record check; it does not replace artifact storage.

The metric record separately records evaluator name/version, evaluator
model/checkpoint, and the full evaluator configuration. GeCo-Eval should be
recorded with metric_role guidance_aligned: it is useful, but not independent
of a VGGT/UFM-based RGB-GeCo objective. The benchmark should also include an
independent geometry metric and a trajectory-adherence metric.

## Method provenance

method.mechanism distinguishes mechanism variants explicitly. For the current
RGB reference, use a value such as:

    {
      "id": "rgb_geco_flow_matching_x0",
      "version": "wan-rgb-geco-v1",
      "time_travel": "absent",
      "temporal_vae_context": "past_only_approximate",
      "guidance_schedule_state": "active_guidance"
    }

This avoids calling a flow-matching x0 implementation exact original GeCo when
it lacks GeCo time travel and uses an approximate temporal VAE slice.

guidance_schedule_state is mandatory and must be one of:

- baseline_all_zero_schedule: no guidance step is scheduled;
- positive_schedule_zero_lr: guidance steps are scheduled, but their update
  scale is zero; and
- active_guidance: at least one scheduled update has a non-zero scale.

The first two are deliberately distinct. They may be expected to produce the
same video under a correct no-op implementation, but they do not exercise the
same code path and must not be silently merged in a reproducibility table.

## Commands

Create records in Python and stamp the canonical condition hash:

    from navigation_benchmark.manifest import with_condition_hash, write_records
    record = with_condition_hash(record)
    write_records("runs.jsonl", [record])

After a run completes and execution provenance is filled:

    from navigation_benchmark.manifest import with_record_hash
    completed_record = with_record_hash(completed_record)

Validate a final three-arm pair:

    python -m navigation_benchmark validate \
      --manifest runs.jsonl \
      --expected-method fg_only \
      --expected-method fg_rgb_geco \
      --expected-method fg_latent_geometry \
      --require-static-scene

Compute trajectory adherence from externally estimated pose series and source
GT poses:

    python -m navigation_benchmark trajectory \
      --gt-poses gt_anchor_poses.json \
      --predicted-poses predicted_anchor_poses.json \
      --run-id example-run \
      --metric-output trajectory_metrics.jsonl \
      --evaluator-name pose-estimator \
      --evaluator-version v1 \
      --evaluator-model-id model-id \
      --evaluator-checkpoint-revision revision \
      --evaluator-config-json '{"pose_convention":"world_from_camera"}'

Aggregate a lower-is-better metric as a paired comparison:

    python -m navigation_benchmark aggregate \
      --manifest runs.jsonl \
      --metrics metrics.jsonl \
      --baseline-method fg_only \
      --candidate-method fg_rgb_geco \
      --metric geco_fused

The resulting improvement is positive when the candidate is better. The
summary includes paired count, mean and median improvement, fraction improved,
and a deterministic bootstrap confidence interval.

## External data still required

The protocol cannot manufacture the following source data:

- sequence-disjoint train/dev/test splits;
- source clip intervals and native FPS;
- first/middle/last image anchors;
- source intrinsics and GT camera poses;
- static-scene eligibility labels and rationale; and
- an external pose estimator or reconstruction method for trajectory evaluation.

Those inputs must be frozen in a dataset manifest before hyperparameter tuning
or critic training. The template in examples is intentionally not directly
valid: it contains replace-with placeholders and must be hashed after filling
real data.
