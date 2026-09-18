# Geometry Evidence Contract for Future Branch Integration

## Purpose

Free-space hallucination and persistent-surface deformation should use one
sampling pipeline and one attention-injection interface. Detectors may be
developed in parallel, but they must not install independent hooks or invent
incompatible tensor layouts.

This document is an interface proposal only. It does not change either
experimental branch.

## Shared Token Layout

All tensors refer to the Wan transformer token grid:

```text
target state: [T-1, H, W]
source slots: [T-1, H*W, S]
source confidence: [T-1, H, W, S]
flatten order: T-major, then H-major, then W
```

The conditioning token `t0` is immutable. A target token at model time `t`
may only reference source times `< t`.

## GeometryEvidence Fields

```text
format_version
token_grid
temporal_scale

surface_source_time
surface_source_index
surface_source_confidence
visible_surface_confidence

background_source_time
background_source_index
background_source_confidence
free_space_conflict_confidence

occlusion_confidence
unknown_confidence
static_confidence

generation_contract
source_artifact
evidence_semantics
detector_metadata
```

Meanings:

- `surface_*`: the same persistent static surface visible in source and
  target; used by the deformation/disappearance branch;
- `background_*`: an observed source-ray background behind an illegal
  foreground conflict; used by the free-space branch;
- `occlusion_confidence`: evidence that a known surface is legitimately
  hidden;
- `unknown_confidence`: newly revealed or insufficiently observed region;
- `static_confidence`: confidence that the target belongs to the static
  world rather than a dynamic object.

## State Arbitration

Raw detector evidence may overlap. Before attention injection, one shared
arbiter must produce mutually exclusive actions:

```text
unknown or low confidence
-> abstain

legitimate occlusion
-> abstain

known visible static surface
-> surface reference correction

known free-space conflict
-> observed-background correction
```

The arbiter, not each detector, owns thresholds and state priority.

## Unified Intervention

A first integration can keep the current output-space residual:

```text
O' = O
   + alpha_surface * c_visible * (V_surface - O)
   + alpha_free * c_free * (V_background - O)
```

Both terms are zero for unknown, occluded, dynamic, or low-confidence tokens.
The original self-attention remains responsible for new content.

## Branch Rules

Detector branches:

- may only produce `GeometryEvidence`;
- must not register transformer hooks;
- must not modify the sampling loop;
- must not choose attention layer, denoising window, or alpha;
- must preserve the shared frame-token mapping;
- must include provenance and confidence semantics;
- must pass synthetic geometry tests.

The integration branch:

- owns attention hooks/processors;
- owns state arbitration and abstention gates;
- owns injection strengths and denoising schedule;
- runs cross-family regression tests.

## Required Regression Matrix

```text
free-space canonical positive:
board repaired, approximately 50-degree turn retained

free-space seed0 negative:
abstain

free-space seed2 negative:
abstain

persistent-surface canonical positive:
deformation repaired

alpha=0:
exact baseline

all-zero evidence:
exact baseline

random correspondence:
must not satisfy the full visual + motion criterion
```

Do not merge the two method branches until both positive cases pass alone
under this matrix. After merging, rerun every row before tuning joint alpha
values.
