import sys
import torch
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_guided import WanImageToVideoPipeline

from demo_guidance_fast_failure import _get_compute_dtype_for_vggt, make_motion_metric
from vggt.models.vggt import VGGT
from uniflowmatch.models.ufm import UniFlowMatchConfidence

model = "/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers"
image_path = "/vol/dissolve/yz10325/experiments/experiments_videogpa/data_manipulation/extracted_frames/open_drawer/frame_first.png"
out = Path("/vol/dissolve/yz10325/outputs/wan_geco_residual_smoke/open_drawer_guided.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

prompt = "A robot gripper pulls open the low drawer by the black handle. The camera is fixed."

device = torch.device("cuda")
compute_dtype = _get_compute_dtype_for_vggt(device)

print("loading VGGT/UFM...")
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
    ufm_scale=0.125,
    cov_thresh=0.5,
    percentile_val=20,
    min_threshold=0.2,
    grad_through_vggt=False,
    debug_autograd=False,
)

print("loading Wan...")
vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(42)

print("generating Wan + GeCo residual smoke...")
output = pipe(
    prompt=prompt,
    image=image,
    height=256,
    width=448,
    num_frames=9,
    num_inference_steps=5,
    guidance_scale=5.0,
    generator=generator,
    fixed_frames=[0, 8],
    guidance_step=[0, 1, 1, 0, 0],
    guidance_lr=[0.0, 0.1, 0.1, 0.0, 0.0],
    loss_fn="residual_motion",
    additional_inputs={"residual_motion_metric": residual_motion_metric},
)

export_to_video(output.frames[0], str(out), fps=16)
print("saved:", out)
