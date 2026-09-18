#!/bin/bash
#SBATCH --job-name=geco_models
#SBATCH --account=brics.u6ph
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT="${HOME}/projects/GeCo-71041f9"
CONTAINER="/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif"
VENV="${HOME}/venvs/geco-e4s"
HF_HOME="/projects/u6ph/${USER}/huggingface"
MODEL_DIR="/projects/u6ph/${USER}/geco_models"

mkdir -p "${HF_HOME}" "${MODEL_DIR}" "${PROJECT}/logs"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_HIGH_PERFORMANCE=1
export MODEL_DIR

apptainer exec --bind "${HOME}:${HOME},/projects:/projects" "${CONTAINER}" \
    "${VENV}/bin/python" - <<'PY'
import json
import os
import shlex
from pathlib import Path
from huggingface_hub import snapshot_download

downloads = {
    "wan": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
    "cosmos": ("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"),
}
auxiliary = [
    ("facebook/VGGT-1B", None),
    ("infinity1096/UFM-Base", None),
]

resolved = {}
for key, (repo_id, revision) in downloads.items():
    print(f"downloading {repo_id} revision={revision}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=os.environ["HF_HUB_CACHE"],
        max_workers=4,
    )
    print(f"ready {repo_id}: {path}", flush=True)
    resolved[key] = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "path": str(Path(path).resolve()),
        "snapshot_commit": Path(path).name,
    }

for repo_id, revision in auxiliary:
    print(f"downloading {repo_id} revision={revision}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=os.environ["HF_HUB_CACHE"],
        max_workers=4,
    )
    print(f"ready {repo_id}: {path}", flush=True)

model_dir = Path(os.environ["MODEL_DIR"])
(model_dir / "model_refs.json").write_text(json.dumps(resolved, indent=2) + "\n")
(model_dir / "model_refs.sh").write_text(
    "\n".join(
        [
            f"WAN_MODEL={shlex.quote(resolved['wan']['path'])}",
            f"COSMOS_MODEL={shlex.quote(resolved['cosmos']['path'])}",
        ]
    )
    + "\n"
)
print(json.dumps(resolved, indent=2))
PY

du -sh "${HF_HOME}"
