import sys
import torch
from pathlib import Path
from diffusers.utils import export_to_video

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_cosmos")
from pipeline_cosmos2_5_predict_guided import Cosmos2_5_PredictBasePipeline


def _cosmos_execution_device(self):
    try:
        return next(self.transformer.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

Cosmos2_5_PredictBasePipeline._execution_device = property(_cosmos_execution_device)

model = "/vol/dissolve/yz10325/checkpoints/Cosmos-Predict2.5-2B-diffusers-base-post-trained"
out = Path("/vol/dissolve/yz10325/outputs/cosmos_geco_residual_smoke/red_cube_baseline.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

prompt = "A realistic fixed camera video of a red cube sliding smoothly from the left side of a wooden table to the right side. The camera is fixed."

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

generator = torch.Generator(device="cuda").manual_seed(42)
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
)

export_to_video(output.frames[0], str(out), fps=16)
print("saved:", out, flush=True)
