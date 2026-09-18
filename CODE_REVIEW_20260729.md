# Independent Geometry-Attention Code Review

Date: 2026-07-29

Scope:

- `external/guidance_wan/draft_geometry_map.py`
- `external/guidance_wan/geometry_transport.py`
- `external/guidance_wan/pipeline_wan_i2v_geometry_transport.py`
- `scripts/build_draft_geometry_map.py`
- `run_wan_geometry_transport.py`

The reviewer received the method equations, tensor conventions, source code,
and required tests. The reviewer did not receive the preliminary experiment
results.

## Confirmed Correct

- VGGT extrinsics are treated as world-to-camera transforms.
- VGGT depth is treated as camera-space Z.
- The target-to-source relative transform and free-space conflict sign are
  correct.
- Wan's causal frame mapping is correct, including frames 13-16 mapping to
  temporal token 4.
- Wan patch tokens flatten in temporal-height-width order.
- The block 0 pre-hook modifies the residual-stream input after patch
  embedding.
- Conditional and unconditional CFG forwards are tracked separately.
- Suppression preserves temporal conditioning token 0.

## Findings And Resolution

### High

1. The runner and pipeline did not enforce the reviewed experiment contract.
   Resolved by strict offline suppression mode, which requires a precomputed
   map, block 0, conditional-only `hidden_input_suppress`, no online geometry,
   and no attention-averaging method.
2. Geometry maps had no generation provenance. Resolved by hashing the
   conditioning image, draft video, model and scheduler configuration files,
   and validating prompt, negative prompt, seed, steps, frames, resolution,
   FPS, guidance scale, and model path before guided sampling.
3. Bilinear source-depth sampling could create false conflicts at depth
   boundaries. Resolved by conservative local-min depth sampling and
   abstention when local relative depth spread exceeds a configured threshold.

### Medium

1. Non-finite or out-of-range confidence could poison hidden states, and an
   empty map could be marked active. Resolved by confidence validation and
   explicit empty-map abstention.
2. Multiple RGB frames mapping to one Wan temporal token overwrote one
   another. Resolved by order-independent temporal support aggregation.
3. Pixel/token conversion mixed centre conventions. Resolved by using
   `(token + 0.5) * scale - 0.5` for pixel centres and
   `floor((pixel + 0.5) / scale)` for containing-token lookup.
4. Hooks and attention processors were restored only after a successful
   denoising loop. Resolved with `try/finally` cleanup.

### Low

1. The correspondence visualizer used `frame // 4`. Resolved by importing the
   shared causal frame-to-latent mapper.

## Additional Bugs Found By Tests

- Zero VGGT confidence was accepted when the configured confidence floor was
  zero. Confidence masks now require finite, strictly positive values.
- Token centres were computed with the correct half-pixel formula but then
  quantized before camera backprojection. Backprojection now uses continuous
  pixel-centre coordinates.

## Executable Evidence

CPU suite:

```text
tests/test_geometry_attention_correctness.py
11 tests passed
```

It covers projection identity, free-space sign, legal surfaces, low-confidence
abstention, depth-edge abstention, causal frame mapping, order-independent
temporal aggregation, alpha-zero and zero-mask no-op behavior, single-token
and all-token masks, invalid confidence, strict contract rejection,
provenance mismatch, and exception-safe cleanup structure.

Tiny Wan integration artifacts:

```text
/vol/dissolve/yz10325/outputs/geometry_attention_20260728/
code_correctness_tests_20260729_v2/
```

- Reference vs `alpha=0` with an active single-token map: exact decoded-frame
  `framemd5` match.
- Reference vs `alpha=0.05` with an all-zero map: exact decoded-frame
  `framemd5` match, with explicit `status=abstained`.
- Same seed repeated: exact decoded-frame `framemd5` match.

## Remaining Limitations

- The review and tests increase confidence but cannot prove the absence of all
  defects.
- VGGT geometry can still be inaccurate even when the implementation is
  correct. Geometry-map visualization and abstention remain required before
  every guided experiment.
- Conflict suppression identifies where the draft is suspect; it does not yet
  provide a verified reference surface as corrective K/V evidence.
- No large parameter sweep should resume until the rebuilt format-v2 map is
  visually inspected on the fixed corridor case.
