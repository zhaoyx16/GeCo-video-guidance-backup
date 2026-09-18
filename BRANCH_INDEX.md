# Repository Branch Index

Snapshot date: 2026-09-18

This repository contains several independent experiment lines. A branch being newer does not mean that it supersedes every other branch. Use the frozen tags below for reproducible paper experiments, and use this index to locate exploratory or supporting work.

## Start Here

1. **Frozen Source-aware C2F Val100 code**
   - Tag: `source-aware-val100-v1`
   - Commit: `cd3719d`
   - Branch: `codex/c2f-source-rerank-val100-0916`
2. **Host-matched baseline extension of the same lineage**
   - Tag: `source-aware-hostmatched-baseline-v1`
   - Commit: `ad6edb4`
   - Branch: `codex/c2f-source-hostmatched-baseline-0916`
3. **Repository navigation and backup record**
   - Branch: `codex/repository-inventory-0918`

`main` is an older July backup at `b239735`. It is not the current experiment entry point and has intentionally not been rewritten or force-updated.

## Authoritative C2F Lineage

The branches below form one direct ancestry chain:

```text
codex/c2f-validation-0914
  -> codex/c2f-p0-forensics-0915
  -> codex/c2f-geometry-validation-0915
  -> codex/c2f-source-rerank-val100-0916
  -> codex/c2f-source-hostmatched-baseline-0916
```

| Branch | Commit | Role | Status |
|---|---:|---|---|
| `codex/c2f-validation-0914` | `4ceefe3` | Frozen Dev25 validation and paired evaluation protocol | Historical protocol stage |
| `codex/c2f-p0-forensics-0915` | `4eb059d` | Preregistered winner/loser forensic analysis | Diagnostic stage |
| `codex/c2f-geometry-validation-0915` | `76267b9` | Stage A/B external-geometry sanity and gated-C2F experiments | Completed development stage |
| `codex/c2f-source-rerank-val100-0916` | `cd3719d` | Source-aware V/P/S Val100 experiment implementation | **Frozen method code** |
| `codex/c2f-source-hostmatched-baseline-0916` | `ad6edb4` | Adds host-matched Wan validation baseline support | **Latest extension of frozen lineage** |

The host-matched branch is newer because it adds baseline support. The method configuration itself remains pinned by `source-aware-val100-v1`.

## Experimental Method Branches

| Branch | Commit | Contents | Use |
|---|---:|---|---|
| `codex/rgb-geco-reference-candidate` | `80ae5e2` | Adapted RGB GeCo guidance reference | Baseline/reference implementation |
| `codex/experiment-snapshot-0726` | `71041f9` | Navigation manifests plus early Wan latent/attention exploration | Historical common base |
| `codex/baseline-screen-0729` | `3537505` | Navigation baseline screening tools and cases | Dataset/case screening |
| `codex/frame-guidance-wan` | `40a53a2` | Latest backed-up Wan frame-guidance experiments | Separate experiment line |
| `codex/latent-geometry-probe-wan` | `49c2364` | DL3DV Wan latent geometry probe | Probe/analysis |
| `codex/draft-geometry-attn-0728` | `6371fa1` | Draft-conditioned geometry-attention experiments | Exploratory snapshot |
| `codex/geometry-attention-0728` | `5289245` | Geometry-aligned attention/value transport experiments | Exploratory snapshot |
| `codex/c2f-memory-geometry-wip-0918` | `bd4111d` | C2F memory variants and geometry transport prototypes | **WIP, not a frozen result** |

## Evaluation and Infrastructure Branches

| Branch | Commit | Role |
|---|---:|---|
| `codex/evaluation-navigation` | `6762fe0` | Independent epipolar/navigation evaluation pilot |
| `feature/geometry-selection` | `9445e7e` | Audited online geometry-selection tooling |
| `codex/online-branch-audit-v1-20260806` | `3507394` | Auditable all-candidate online rollouts |
| `online-heldout-v1` | `78729ae` | Held-out/local window extraction support |
| `feature/geco-evaluation-runtime-hippasus` | `d57d920` | Hippasus evaluation runtime fixes |
| `codex/geco-evaluation-runtime-hippasus-v5` | `c02ae73` | Split-VAE geometry metadata runtime fix |
| `fullgraph-dpo-vggt-loader-v1` | `b1d4a1e` | VGGT safetensors loader support |
| `codex/fullgraph-dpo-traindev-v1` | `d8d89ab` | Train/dev evaluator environment certification |

## Interpretation Rules

- A GitHub branch is the latest committed state of that branch, not necessarily the latest state of the whole project.
- Frozen tags identify reproducible paper configurations. Prefer tags over moving branch names when citing a result.
- `WIP` and `exploratory snapshot` branches preserve research history but are not validated final methods.
- Do not merge all branches mechanically. Several branches intentionally isolate incompatible experiments.
- No Test100 result is indexed or exposed here.

See `docs/EXPERIMENT_INDEX.md` for the research-purpose map and `docs/REPOSITORY_BACKUP_PLAN.md` for the backup audit.
