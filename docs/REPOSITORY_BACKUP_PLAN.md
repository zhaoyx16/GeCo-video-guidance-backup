# Repository Backup Record

Snapshot date: 2026-09-18

## Scope

The GitHub backup remote is:

```text
git@github.com:zhaoyx16/GeCo-video-guidance-backup.git
```

This pass preserved committed branches and captured previously dirty experiment worktrees without merging, rebasing, force-pushing, or deleting history.

## Actions Completed

- Pushed all clean experiment and infrastructure branches that were previously local-only.
- Created branch names for two useful detached commits and pushed them:
  - `codex/fullgraph-dpo-traindev-v1` at `d8d89ab`
  - `codex/geco-evaluation-runtime-hippasus-v5` at `c02ae73`
- Added annotated frozen tags:
  - `source-aware-val100-v1` -> `cd3719d`
  - `source-aware-hostmatched-baseline-v1` -> `ad6edb4`
- Captured dirty experiment worktrees as explicit snapshot commits:

| Branch | Commit | Verification before push |
|---|---:|---|
| `codex/baseline-screen-0729` | `3537505` | Python syntax check passed |
| `codex/draft-geometry-attn-0728` | `6371fa1` | Python syntax check passed |
| `codex/frame-guidance-wan` | `40a53a2` | `11 passed` in frame-guidance helper tests |
| `codex/geometry-attention-0728` | `5289245` | `16 passed` in geometry-attention correctness tests |
| `codex/c2f-memory-geometry-wip-0918` | `bd4111d` | Python syntax check passed; manifest tests `3 passed` |

## WIP Copy Handling

The original July experiment worktree contained untracked C2F-memory and geometry-transport files. They were copied into an isolated branch instead of being committed onto the historical `codex/experiment-snapshot-0726` branch.

Fourteen `._*` AppleDouble metadata files were found in the copied paths. They were excluded from the new branch because they are macOS filesystem metadata, not source code. Their real source counterparts passed syntax checks. The original worktree was not cleaned or deleted.

## Safety Checks

- No force push was used.
- No branch or tag was deleted.
- No experiment branch was merged into another.
- No model weights, private keys, SSH certificates, tokens, cookies, or `.env` files were included.
- A high-risk credential-pattern scan found no private-key headers, Hugging Face tokens, GitHub tokens, or AWS access-key patterns in the new WIP snapshot.
- Sealed Test100 outputs and per-case results were not accessed or added.

## Recovery

To recover the frozen method exactly:

```bash
git fetch backup --tags
git switch --detach source-aware-val100-v1
```

To recover the host-matched extension:

```bash
git switch --detach source-aware-hostmatched-baseline-v1
```

To continue development without changing a frozen tag, create a new branch from the desired tag or commit.

## Deferred Repository Cleanup

The following actions were intentionally not performed:

- updating or replacing `main`;
- deleting historical branches;
- combining experiment lines;
- renaming existing branches;
- choosing a final public release branch.

A later reviewed integration branch such as `paper/code-release-v1` should contain only the selected method, reproducible configs, evaluation entry points, and documentation required by the dissertation.
