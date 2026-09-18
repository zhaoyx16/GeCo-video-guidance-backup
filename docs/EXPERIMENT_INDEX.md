# Experiment Index

This document maps research questions to code branches. It distinguishes a backed-up implementation from a frozen experiment and from an exploratory result.

## 1. RGB GeCo Adaptation

**Question:** Can GeCo's differentiable RGB-space geometry guidance be adapted to Wan/Cosmos?

| Asset | Branch | Status |
|---|---|---|
| Adapted RGB GeCo candidate | `codex/rgb-geco-reference-candidate` | Reference baseline implementation |
| Wan frame-guidance work | `codex/frame-guidance-wan` | Separate backed-up experiment line |

Use this line as the adapted-GeCo comparison, not as the implementation base for Source-aware C2F.

## 2. Latent and Attention Exploration

**Question:** Can frozen Wan sampling be modified inside latent/transformer feature space without differentiable VAE decoding?

| Asset | Branch | Status |
|---|---|---|
| Early latent/attention snapshot | `codex/experiment-snapshot-0726` | Historical base |
| DL3DV latent geometry probe | `codex/latent-geometry-probe-wan` | Diagnostic probe |
| C2F memory variants | `codex/c2f-memory-geometry-wip-0918` | WIP snapshot |

The WIP branch preserves alternatives such as fine mutual matching, flow smoothing, margin, Q/K, and step consensus. It does not establish that those variants outperform the frozen method.

## 3. C2F Development and Validation

**Question:** Does low-cost coarse-to-fine temporal value memory improve video consistency, and why does it help some scenes but hurt others?

| Stage | Branch | What it contains |
|---|---|---|
| Dev validation | `codex/c2f-validation-0914` | Frozen Dev25 protocol, visual review, paired metrics, decision gates |
| P0 forensics | `codex/c2f-p0-forensics-0915` | Preregistered winner/loser diagnostics and audit |
| Geometry validation | `codex/c2f-geometry-validation-0915` | External-geometry sanity checks and geometry-gated Stage B experiments |
| Source-aware method | `codex/c2f-source-rerank-val100-0916` | Frozen Veto/Pairwise/Source-aware Val100 implementation |
| Host-matched extension | `codex/c2f-source-hostmatched-baseline-0916` | Adds host-matched Wan baseline support |

For reproducibility, use these immutable references:

```text
source-aware-val100-v1
source-aware-hostmatched-baseline-v1
```

## 4. Explicit Geometry and Attention Transport

**Question:** Can explicit geometric evidence localize an error and transport a trusted value feature to repair it?

| Asset | Branch | Status |
|---|---|---|
| Draft-assisted geometry attention | `codex/draft-geometry-attn-0728` | Exploratory snapshot |
| Geometry-aligned attention/value transport | `codex/geometry-attention-0728` | Exploratory snapshot with correctness tests |
| Geometry transport prototypes mixed with C2F variants | `codex/c2f-memory-geometry-wip-0918` | WIP snapshot |

These branches provide mechanism evidence and implementation ideas. They should not be described as the final integrated method unless a later frozen benchmark establishes that claim.

## 5. Dataset and Case Screening

**Question:** Which DL3DV/navigation cases provide valid, interpretable failures and fair paired comparisons?

| Asset | Branch | Status |
|---|---|---|
| Navigation manifests and tooling | `codex/experiment-snapshot-0726` | Historical snapshot |
| Baseline screening tooling | `codex/baseline-screen-0729` | Backed-up screening branch |
| DL3DV benchmark/port utilities | `codex/c2f-memory-geometry-wip-0918` | WIP utilities, not final method code |

## 6. Evaluation and Audit Infrastructure

Evaluation support is intentionally separated from method branches where practical:

- `codex/evaluation-navigation`
- `feature/geco-evaluation-runtime-hippasus`
- `codex/geco-evaluation-runtime-hippasus-v5`
- `feature/geometry-selection`
- `codex/online-branch-audit-v1-20260806`
- `online-heldout-v1`
- `fullgraph-dpo-vggt-loader-v1`
- `codex/fullgraph-dpo-traindev-v1`

These branches contain pilots, runtime fixes, candidate auditing, held-out support, or evaluator setup. They do not by themselves define the paper's proposed method.

## 7. Dissertation Use

When documenting an experiment, record all of the following together:

1. research question and hypothesis;
2. immutable Git tag or full commit hash;
3. frozen scene manifest and split;
4. conditioning image, prompt, seed, frames, resolution, and sampling settings;
5. method configuration;
6. visual audit outcome before metric calculation;
7. metric protocol and paired statistical analysis;
8. runtime and peak VRAM;
9. result interpretation and failure cases.

Use the frozen Source-aware tags for the current formal experiment. Treat all WIP branches as idea archives until they pass the same paired protocol. Do not use sealed Test100 results during method selection.
