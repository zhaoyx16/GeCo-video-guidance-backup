import sys
import torch

sys.path.insert(0, "/vol/dissolve/yz10325/repos/GeCo/external/guidance_cosmos")
from pipeline_cosmos2_5_predict_guided import Cosmos2_5_PredictBasePipeline


def _cosmos_execution_device(self):
    try:
        return next(self.transformer.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

Cosmos2_5_PredictBasePipeline._execution_device = property(_cosmos_execution_device)

model_id = "/vol/dissolve/yz10325/checkpoints/Cosmos-Predict2.5-2B-diffusers-base-post-trained"
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)
out = pipe(
    image=None,
    video=None,
    prompt="A realistic fixed camera video of a red cube on a wooden table.",
    negative_prompt="low quality, blurry, distorted geometry, flickering, object deformation",
    height=256,
    width=448,
    num_frames=17,
    num_inference_steps=3,
    guidance_scale=7.0,
    generator=generator,
    output_type="latent",
    guidance_step=[0, 1, 0],
    guidance_lr=[0.0, 0.1, 0.0],
    loss_fn="latent_l2",
)
latents = out.frames
print("latents", tuple(latents.shape), latents.dtype, latents.device, float(latents.float().norm()))
