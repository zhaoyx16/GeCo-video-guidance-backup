# C2F External Geometry Validation

This directory implements the preregistered Stage A sanity check in
`STAGE_A_PROTOCOL.md`.

The workflow is intentionally split into four auditable steps:

1. Create `STAGE_A_LOCK.json` before geometry inference.
2. Replay only the five locked cases through step 29 with dense read-only C2F
   diagnostics (`sample_size=8192`). No full video is generated.
3. Run pinned VGGT-Omega on eight existing baseline frames per case and compare
   its projection with the frozen C2F matches at step 20/layer 10.
4. Summarize the five case records and inspect every visualization before any
   Stage B generation.

The scripts refuse stale or mismatched artifacts rather than silently reusing
them.
