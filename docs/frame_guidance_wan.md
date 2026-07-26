# Controlled Wan x0 Frame-MSE / RGB-GeCo Flow-Matching Variants

`run_wan_frame_guidance_case.py` adds two controlled arms to the existing Wan
full-RGB-GeCo pipeline:

* `fg_only`: `controlled_wan_x0_frame_mse_variant`.
* `fg_geco`: `controlled_wan_x0_frame_mse_rgb_geco_flow_matching_variant`.
  It is the exact same sparse anchor arm plus the existing RGB GeCo
  motion-residual term.

The runner is intentionally separate from `run_wan_geco_case_full.py`.  The
audited image-and-text and RGB-GeCo reference path therefore stays unchanged.

## Manifest

The manifest is the source of truth for a paired experiment.  It stores the
prompt, the first-frame condition, all frame anchors, and the source/generated
timestamp contract.  Supply the same required `--pair_id` to both modes.  The
runner emits this id plus a `pair_config_hash`, anchor image hashes, source FPS,
generated FPS, and every anchor's source/generated timestamp.  See
`examples/frame_guidance_wan_manifest_example.json` for a complete schema.

Anchors use zero-based **generated-video frame indices**.  They must contain
index `0` plus at least two later frames.  The runner hashes the manifest and
every anchor image, records them in a per-run JSON file, and refuses to run if
the `image_prompt` condition differs from the frame-0 anchor.

The frame-0 anchor is a provenance and I2V-conditioning record.  It does not
receive a latent gradient because Wan holds the initial condition fixed.  The
middle and final anchors use the frame objective

`L_frame = mean_i MSE(decoded_x0[i], GT_anchor[i])`.

For an anchor `f > 0`, Wan's causal VAE uses the predecessor/target latent pair
`[(f - 1) // 4, (f - 1) // 4 + 1]` and local decoded slot
`(f - 1) % 4 + 1`.  This avoids supervising a future local slot.  The pair has
the correct local index algebra, but later frames can still differ from a full
causal decode because earlier VAE cache state is absent.  Use
`verify_wan_frame_guidance_temporal_mapping.py` when model assets and a GPU are
available to quantify that approximation for the anchors you will use.

For `fg_geco`, the update objective is

`L = frame_loss_weight * L_frame + geco_loss_weight * L_GeCo`.

`L_GeCo` is the existing full-resolution VGGT/UFM residual term.  `x0`
conversion, the full Transformer Jacobian, VAE decode path, loss sign, and the
Wan scheduler update are unchanged from the audited RGB-GeCo pipeline.  The
runner persists raw `frame_loss_raw`, `geco_loss_raw`, and `combined_loss` per
guidance update.  `frame_loss_weight=1.0` and `geco_loss_weight=1.0` are raw,
tunable multipliers, **not** normalized or automatically balanced weights.

These are deliberately controlled Wan variants, not faithful reproductions of
either official Frame Guidance/Video Latent Optimization or original GeCo:
time-travel/re-noising is absent, and selected VAE-frame decoding is causal and
approximate.  Every run manifest records those facts as well as the selected
frame indices, scheduler, decode scale, checkpointing, and cross-device
settings.

## Schedule

The runner requires an explicit `--guidance_schedule`, for example:

```bash
--guidance_schedule '25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1,34:1'
```

The example is a conservative initial **diagnostic** window for 50 Wan steps,
not a fixed paper claim.  Choose the actual window from per-step decoded-x0
visualizations: it should be late enough that the predicted scene is legible,
but early enough to retain trajectory control.  Use the same schedule, learning
rate, seed, prompt, anchors, resolution, FPS, and sampler settings for
`fg_only` and `fg_geco`.

## Deliberate divergence from the official Frame Guidance code

The official Wan implementation conditions on the first image and computes RGB
MSE at selected target-frame indices.  This implementation preserves that core
frame-level objective.  It does **not** port the official Video Latent
Optimization time-travel/re-noising update, because changing the Wan FlowMatch
trajectory would confound the controlled comparison with the already-audited
RGB-GeCo sampler.  Instead, it applies the existing full-Jacobian update before
the unchanged `FlowMatchEulerDiscreteScheduler.step`, then recomputes the model
prediction for the normal scheduler step.

Unlike the official notebook's low-resolution `latent_downscale_factor=4`
example, this arm requires `decode_spatial_scale=1.0` so `fg_geco` uses the
same full-resolution differentiable VAE path as the RGB-GeCo reference.

## Optional VAE parity probe

When the Wan VAE weights and a GPU are available, run:

```bash
python verify_wan_frame_guidance_temporal_mapping.py \
  --frames 121 --anchor_frames 0,60,120 --height 704 --width 1280
```

It compares each predecessor/target selected slice with the corresponding full
decode frame and reports mean/max absolute differences.  It is intentionally
not part of the default unit suite because loading the VAE is expensive.  The
algebraic mapping is unit-tested regardless; pixel equality for later frames is
not assumed because causal decoder history before the predecessor is absent.
