# C2F P0 Forensics

This directory contains the locked protocol, replay tooling, and analysis for diagnosing the heterogeneous Dev25 result of frozen `C2F K=3, alpha=0.025`.

The forensic pipeline is a separate copy at `external/guidance_wan/pipeline_wan_i2v_c2f_forensics.py`. The frozen C2F implementation and completed Dev25 outputs are never modified.

Workflow:

1. Build and inspect `P0_LOCK.json` with `make_p0_lock.py`.
2. Verify frozen-versus-forensic step-29 latent equivalence for one case in both replay modes.
3. Replay all 25 scenes in `baseline_observe` and `c2f` modes.
4. Run the preregistered scene-level and intervention-level analysis.
5. Only then implement the locked one-variable counterfactual.
