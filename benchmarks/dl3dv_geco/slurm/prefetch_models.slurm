#!/bin/bash
#SBATCH --job-name=geco-models
#SBATCH --account=brics.u6ph
#SBATCH --output=/projects/u6ph/yanxi.u6ph/logs/geco_models_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT=/projects/u6ph/yanxi.u6ph
CONTAINER_PATH=${PROJECT_ROOT}/containers/pytorch_25.05-py3.sif
REPO_INNER=/vol/dissolve/yz10325/repos/GeCo-geometry-selection
VENV_INNER=/vol/dissolve/yz10325/envs/geco-geometry-selection
HF_HOME_INNER=/vol/dissolve/yz10325/cache/huggingface
HF_HUB_CACHE_INNER=${HF_HOME_INNER}/hub
MODEL_ROOT_INNER=/vol/dissolve/yz10325/geco_models
MODEL_KEYS=${MODEL_KEYS:-wan}

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/cache/huggingface" \
    "${PROJECT_ROOT}/geco_models/frozen" "${PROJECT_ROOT}/geco_models/refs"

apptainer exec \
    --cleanenv \
    --bind "${PROJECT_ROOT}:/vol/dissolve/yz10325" \
    --env "HF_HOME=${HF_HOME_INNER}" \
    --env "HF_HUB_CACHE=${HF_HUB_CACHE_INNER}" \
    --env "HF_XET_HIGH_PERFORMANCE=1" \
    --env "MODEL_KEYS=${MODEL_KEYS}" \
    "${CONTAINER_PATH}" \
    /bin/bash -lc '
        set -euo pipefail
        source "$1/bin/activate"
        cd "$2"
        python - <<"PY"
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from geometry_selection.model_lock import validate_frozen_model_snapshot
from scripts.freeze_model_snapshot import freeze_snapshot


SPECS = {
    "wan": {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "revision": "b8fff7315c768468a5333511427288870b2e9635",
    },
    "cosmos": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "revision": "0d37c7498f54cee3c599d438d895a0a4a8608064",
    },
}

keys = [key.strip() for key in os.environ["MODEL_KEYS"].split(",") if key.strip()]
if not keys or any(key not in SPECS for key in keys):
    raise ValueError(f"MODEL_KEYS must be a comma-separated subset of {sorted(SPECS)}")

model_root = Path("/vol/dissolve/yz10325/geco_models")
for key in keys:
    spec = SPECS[key]
    print(f"downloading {key}: {spec['repo_id']}@{spec['revision']}", flush=True)
    snapshot = Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            cache_dir=os.environ["HF_HUB_CACHE"],
            max_workers=4,
        )
    ).resolve()
    frozen = model_root / "frozen" / f"{key}-{spec['revision']}"
    if frozen.exists():
        validate_frozen_model_snapshot(frozen)
    else:
        freeze_snapshot(snapshot, frozen)
        validate_frozen_model_snapshot(frozen)

    receipt = {
        "schema": "geometry-prefetched-model-v1",
        "key": key,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "snapshot_path": str(snapshot),
        "frozen_path": str(frozen),
    }
    receipt_path = model_root / "refs" / f"{key}.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=receipt_path.parent,
        prefix=f".{receipt_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    print(json.dumps(receipt, indent=2), flush=True)
PY
    ' bash "${VENV_INNER}" "${REPO_INNER}"

echo "MODEL_PREFETCH_READY ${MODEL_KEYS}"
