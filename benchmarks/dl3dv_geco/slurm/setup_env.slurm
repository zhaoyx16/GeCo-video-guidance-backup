#!/bin/bash
#SBATCH --job-name=geco_env
#SBATCH --account=brics.u6ph
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT="${HOME}/projects/GeCo-71041f9"
CONTAINER="/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif"
VENV="${HOME}/venvs/geco-e4s"

mkdir -p "${HOME}/venvs" "${PROJECT}/logs"

apptainer exec --bind "${HOME}:${HOME}" "${CONTAINER}" bash -lc "
set -euo pipefail
python3 -m venv --system-site-packages '${VENV}'
'${VENV}/bin/python' -m pip install --upgrade pip
'${VENV}/bin/python' -m pip install --no-deps \
    diffusers==0.38.0 \
    flow-vis \
    imageio-ffmpeg
'${VENV}/bin/python' -m pip install --no-deps -e '${PROJECT}/external/vggt'
'${VENV}/bin/python' -m pip install --no-deps -e '${PROJECT}/external/UFM'
'${VENV}/bin/python' - <<'PY'
import platform
import importlib.util
from pathlib import Path
import torch
import diffusers
from diffusers import WanImageToVideoPipeline, Cosmos2_5_PredictBasePipeline
from vggt.models.vggt import VGGT
from uniflowmatch.models.ufm import UniFlowMatchConfidence

project = Path('${PROJECT}')
for relative, class_name in [
    ('external/guidance_wan/pipeline_wan_i2v_full_guided.py', 'WanImageToVideoPipeline'),
    ('external/guidance_cosmos/pipeline_cosmos2_5_predict_guided.py', 'Cosmos2_5_PredictBasePipeline'),
]:
    path = project / relative
    spec = importlib.util.spec_from_file_location(f'_geco_setup_{path.stem}', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('custom_pipeline', relative, getattr(module, class_name).__name__)

print('machine', platform.machine())
print('torch', torch.__version__)
print('diffusers', diffusers.__version__)
print('wan_pipeline', WanImageToVideoPipeline.__name__)
print('cosmos_pipeline', Cosmos2_5_PredictBasePipeline.__name__)
print('vggt', VGGT.__name__)
print('ufm', UniFlowMatchConfidence.__name__)
PY
"
