import sys
import torch
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_wan")
from pipeline_wan_i2v_guided import WanImageToVideoPipeline

model = "/vol/dissolve/yz10325/checkpoints/Wan2.2-TI2V-5B-Diffusers"
image_path = "/vol/dissolve/yz10325/experiments/experiments_videogpa/data_manipulation/extracted_frames/open_drawer/frame_first.png"
out = Path("/vol/dissolve/yz10325/outputs/wan_custom_pipeline_smoke/open_drawer_baseline.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

prompt = "A robot gripper pulls open the low drawer by the black handle. The camera is fixed."

vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = Image.open(image_path).convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(42)

output = pipe(
    prompt=prompt,
    image=image,
    height=256,
    width=448,
    num_frames=9,
    num_inference_steps=5,
    guidance_scale=5.0,
    generator=generator,
    guidance_step=[0, 1, 1, 0, 0],
    guidance_lr=[0.0, 0.1, 0.1, 0.0, 0.0],
    loss_fn="latent_l2",
)

export_to_video(output.frames[0], str(out), fps=16)
print("saved:", out)
