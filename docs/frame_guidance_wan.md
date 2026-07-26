# Wan Frame Guidance Arm

`run_wan_frame_guidance_case.py` adds two controlled arms to the existing Wan
full-RGB-GeCo pipeline:

* `fg_only`: sparse first/middle/last RGB anchor guidance.
* `fg_geco`: the exact same sparse anchor guidance plus the existing RGB GeCo
  motion-residual term.  The provenance label for this arm is
  `rgb_geco_flow_matching_variant`.

The runner is intentionally separate from `run_wan_geco_case_full.py`.  The
audited image-and-text and RGB-GeCo reference path therefore stays unchanged.

## Manifest

The manifest is the source of truth for a paired experiment.  It stores the
prompt, the first-frame condition, and all frame anchors.  See
`examples/frame_guidance_wan_manifest_example.json` for a complete schema.

Anchors use zero-based **generated-video frame indices**.  They must contain
index `0` plus at least two later frames.  The runner hashes the manifest and
every anchor image, records them in a per-run JSON file, and refuses to run if
the `image_prompt` condition differs from the frame-0 anchor.

The frame-0 anchor is a provenance and I2V-conditioning record.  It does not
receive a latent gradient because Wan holds the initial condition fixed.  The
middle and final anchors use the frame objective

`L_frame = mean_i MSE(decoded_x0[i], GT_anchor[i])`.

For `fg_geco`, the update objective is

`L = frame_loss_weight * L_frame + geco_loss_weight * L_GeCo`.

`L_GeCo` is the existing full-resolution VGGT/UFM residual term.  `x0`
conversion, the full Transformer Jacobian, VAE decode path, loss sign, and the
Wan scheduler update are unchanged from the audited RGB-GeCo pipeline.

This is deliberately called an **RGB GeCo FlowMatch variant**, not an exact
reproduction of original GeCo: it omits time-travel/re-noising and relies on a
causal-VAE selected-frame slice.  Every run manifest records those facts as
well as the selected frame indices, scheduler, decode scale, checkpointing,
and cross-device settings.

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
