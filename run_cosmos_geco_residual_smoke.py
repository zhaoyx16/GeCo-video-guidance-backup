import sys
import torch
from pathlib import Path
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_cosmos")
from pipeline_cosmos2_5_predict_guided import Cosmos2_5_PredictBasePipeline

from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
from vggt.models.vggt import VGGT
from uniflowmatch.models.ufm import UniFlowMatchConfidence


def _cosmos_execution_device(self):
    try:
        return next(self.transformer.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

Cosmos2_5_PredictBasePipeline._execution_device = property(_cosmos_execution_device)

model = "/vol/dissolve/yz10325/checkpoints/Cosmos-Predict2.5-2B-diffusers-base-post-trained"
out = Path("/vol/dissolve/yz10325/outputs/cosmos_geco_residual_smoke/red_cube_guided.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

prompt = "A realistic fixed camera video of a red cube sliding smoothly from the left side of a wooden table to the right side. The camera is fixed."

device = torch.device("cuda")
compute_dtype = _get_compute_dtype_for_vggt(device)

print("loading VGGT/UFM...", flush=True)
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
ufm_model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base").to(dtype=torch.float32, device=device).eval()
for p in vggt_model.parameters():
    p.requires_grad_(False)
for p in ufm_model.parameters():
    p.requires_grad_(False)

residual_motion_metric = make_motion_metric(
    vggt_model,
    ufm_model,
    device,
    compute_dtype,
    vggt_strategy="once",
    pair_mode="adjacent",
    ufm_scale=0.0625,
    cov_thresh=0.5,
    percentile_val=20,
    min_threshold=0.2,
    grad_through_vggt=False,
    debug_autograd=False,
)

print("loading Cosmos diffusers...", flush=True)
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

generator = torch.Generator(device="cuda").manual_seed(42)

print("generating Cosmos + GeCo residual smoke...", flush=True)
output = pipe(
    image=None,
    video=None,
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted geometry, flickering, object deformation",
    height=256,
    width=448,
    num_frames=17,
    num_inference_steps=3,
    guidance_scale=7.0,
    generator=generator,
    fixed_frames=[0, 8, 16],
    guidance_step=[0, 1, 0],
    guidance_lr=[0.0, 0.05, 0.0],
    loss_fn="residual_motion",
    additional_inputs={"residual_motion_metric": residual_motion_metric},
)

export_to_video(output.frames[0], str(out), fps=16)
print("saved:", out, flush=True)
