# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models import AutoencoderKLWan, CosmosTransformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import (
    is_cosmos_guardrail_available,
    is_torch_xla_available,
    is_torchvision_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput


if is_torchvision_available():
    import torchvision.transforms.functional


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _geco_move_tensor(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
    *,
    via_cpu_for_grad: bool = False,
    force_via_cpu: bool = False,
) -> torch.Tensor:
    """Move a tensor without relying on unreliable direct multi-GPU copies on this host.

    CPU staging is required for guidance tensors that carry gradients. It is also
    used for the VAE conditioning/final-decode forward transfers when the VAE is
    placed on another GPU: the direct transfer path produced constant gray videos
    despite a valid single-GPU VAE decode. Same-device paths are unchanged.
    """
    target = torch.device(device)
    if tensor.device == target:
        return tensor if dtype is None else tensor.to(dtype=dtype)
    if force_via_cpu or (via_cpu_for_grad and torch.is_grad_enabled() and tensor.requires_grad):
        staged = tensor.to("cpu")
        return staged.to(device=target, dtype=dtype) if dtype is not None else staged.to(target)
    return tensor.to(device=target, dtype=dtype) if dtype is not None else tensor.to(target)


def _geco_tiled_decode_with_per_tile_checkpoint(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    """Match Diffusers ``tiled_decode`` while checkpointing one spatial tile at a time.

    A whole-VAE checkpoint releases the initial forward activations, but its backward
    recomputation still retains the causal decoder graph for every spatial tile until
    the complete tiled video is assembled. Each tile in the upstream implementation
    starts with an empty causal cache, so keeping that cache local to a tile makes it
    safe to checkpoint tiles independently without changing the VAE's tile layout,
    blending, temporal order, decoded pixels, or guidance objective.
    """
    _, _, num_frames, height, width = z.shape
    sample_height = height * vae.spatial_compression_ratio
    sample_width = width * vae.spatial_compression_ratio

    tile_latent_min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    tile_latent_min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    tile_latent_stride_height = vae.tile_sample_stride_height // vae.spatial_compression_ratio
    tile_latent_stride_width = vae.tile_sample_stride_width // vae.spatial_compression_ratio
    tile_sample_stride_height = vae.tile_sample_stride_height
    tile_sample_stride_width = vae.tile_sample_stride_width

    if vae.config.patch_size is not None:
        sample_height = sample_height // vae.config.patch_size
        sample_width = sample_width // vae.config.patch_size
        tile_sample_stride_height = tile_sample_stride_height // vae.config.patch_size
        tile_sample_stride_width = tile_sample_stride_width // vae.config.patch_size
        blend_height = vae.tile_sample_min_height // vae.config.patch_size - tile_sample_stride_height
        blend_width = vae.tile_sample_min_width // vae.config.patch_size - tile_sample_stride_width
    else:
        blend_height = vae.tile_sample_min_height - tile_sample_stride_height
        blend_width = vae.tile_sample_min_width - tile_sample_stride_width

    vae.clear_cache()
    rows = []
    for top in range(0, height, tile_latent_stride_height):
        row = []
        for left in range(0, width, tile_latent_stride_width):
            z_tile = z[:, :, :, top : top + tile_latent_min_height, left : left + tile_latent_min_width]

            def _decode_one_tile(tile_latents: torch.Tensor) -> torch.Tensor:
                # This is a private cache for exactly one upstream tiled_decode tile.
                # It is deliberately not vae._feat_map, which would be mutated again
                # when checkpoint recomputes this tile during autograd backward.
                feat_cache = [None] * vae._cached_conv_counts["decoder"]
                decoded_frames = []
                for frame_idx in range(num_frames):
                    feat_idx = [0]
                    frame_latents = vae.post_quant_conv(tile_latents[:, :, frame_idx : frame_idx + 1])
                    decoded_frames.append(
                        vae.decoder(
                            frame_latents,
                            feat_cache=feat_cache,
                            feat_idx=feat_idx,
                            first_chunk=(frame_idx == 0),
                        )
                    )
                return torch.cat(decoded_frames, dim=2)

            row.append(checkpoint(_decode_one_tile, z_tile, use_reentrant=False))
        rows.append(row)

    result_rows = []
    for row_idx, row in enumerate(rows):
        result_row = []
        for column_idx, tile in enumerate(row):
            if row_idx > 0:
                tile = vae.blend_v(rows[row_idx - 1][column_idx], tile, blend_height)
            if column_idx > 0:
                tile = vae.blend_h(row[column_idx - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width])
        result_rows.append(torch.cat(result_row, dim=-1))

    decoded = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
    if vae.config.patch_size is not None:
        decoded = unpatchify(decoded, patch_size=vae.config.patch_size)
    vae.clear_cache()
    return torch.clamp(decoded, min=-1.0, max=1.0)

DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Cosmos2_5_PredictBasePipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video

        >>> model_id = "nvidia/Cosmos-Predict2.5-2B"
        >>> pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        ...     model_id, revision="diffusers/base/post-trained", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Common negative prompt reused across modes.
        >>> negative_prompt = (
        ...     "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
        ...     "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
        ...     "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
        ...     "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
        ...     "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
        ...     "Overall, the video is of poor quality."
        ... )

        >>> # Text2World: generate a 93-frame world video from text only.
        >>> prompt = (
        ...     "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights "
        ...     "cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh "
        ...     "lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet "
        ...     "reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. "
        ...     "The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow "
        ...     "advance of traffic through the frosty city corridor."
        ... )
        >>> video = pipe(
        ...     image=None,
        ...     video=None,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "text2world.mp4", fps=16)

        >>> # Image2World: condition on a single image and generate a 93-frame world video.
        >>> prompt = (
        ...     "A high-definition video captures the precision of robotic welding in an industrial setting. "
        ...     "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. "
        ...     "The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid "
        ...     "display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring "
        ...     "the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a "
        ...     "ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video "
        ...     "progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. "
        ...     "The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. "
        ...     "The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with "
        ...     "the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
        ... )
        >>> image = load_image(
        ...     "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"
        ... )
        >>> video = pipe(
        ...     image=image,
        ...     video=None,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "image2world.mp4", fps=16)

        >>> # Video2World: condition on an input clip and predict a 93-frame world video.
        >>> prompt = (
        ...     "The video opens with an aerial view of a large-scale sand mining construction operation, showcasing extensive piles "
        ...     "of brown sand meticulously arranged in parallel rows. A central water channel, fed by a water pipe, flows through the "
        ...     "middle of these sand heaps, creating ripples and movement as it cascades down. The surrounding area features dense green "
        ...     "vegetation on the left, contrasting with the sandy terrain, while a body of water is visible in the background on the right. "
        ...     "As the video progresses, a piece of heavy machinery, likely a bulldozer, enters the frame from the right, moving slowly along "
        ...     "the edge of the sand piles. This machinery's presence indicates ongoing construction work in the operation. The final frame "
        ...     "captures the same scene, with the water continuing its flow and the bulldozer still in motion, maintaining the dynamic yet "
        ...     "steady pace of the construction activity."
        ... )
        >>> input_video = load_video(
        ...     "https://github.com/nvidia-cosmos/cosmos-predict2.5/raw/refs/heads/main/assets/base/sand_mining.mp4"
        ... )
        >>> video = pipe(
        ...     image=None,
        ...     video=input_video,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "video2world.mp4", fps=16)

        >>> # To produce an image instead of a world (video) clip, set num_frames=1 and
        >>> # save the first frame: pipe(..., num_frames=1).frames[0][0].
        ```
"""


class Cosmos2_5_PredictBasePipeline(DiffusionPipeline):
    r"""
    Pipeline for [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) base model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`Qwen2_5_VLForConditionalGeneration`]):
            Frozen text-encoder. Cosmos Predict2.5 uses the [Qwen2.5
            VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) encoder.
        tokenizer (`AutoTokenizer`):
            Tokenizer associated with the Qwen2.5 VL encoder.
        transformer ([`CosmosTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # We mark safety_checker as optional here to get around some test failures, but it is not really optional
    _optional_components = ["safety_checker"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        safety_checker: CosmosSafetyChecker = None,
    ):
        super().__init__()

        if safety_checker is None:
            try:
                safety_checker = CosmosSafetyChecker()
            except ImportError:
                safety_checker = None

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_mean", None) is not None
            else None
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_std", None) is not None
            else None
        )
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

    def _get_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        input_ids_batch = []

        for sample_idx in range(len(prompt)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[sample_idx],
                        }
                    ],
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = (
                input_ids["input_ids"] if not isinstance(input_ids, list) and "input_ids" in input_ids else input_ids
            )
            input_ids = torch.LongTensor(input_ids)
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        outputs = self.text_encoder(
            input_ids_batch.to(device),
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)) / (
                hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8
            )
            normalized_hidden_states.append(normalized_state)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds

    # Modified from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    # Modified from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2VideoToWorldPipeline.prepare_latents and
    # diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2TextToImagePipeline.prepare_latents
    def prepare_latents(
        self,
        video: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 704,
        width: int = 1280,
        num_frames_in: int = 93,
        num_frames_out: int = 93,
        do_classifier_free_guidance: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        B = batch_size
        C = num_channels_latents
        T = (num_frames_out - 1) // self.vae_scale_factor_temporal + 1
        H = height // self.vae_scale_factor_spatial
        W = width // self.vae_scale_factor_spatial
        shape = (B, C, T, H, W)
        vae_device = next(self.vae.parameters()).device

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
            cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)

            cond_latents = torch.zeros_like(latents)

            return (
                latents,
                cond_latents,
                cond_mask,
                cond_indicator,
            )
        else:
            if video is None:
                raise ValueError("`video` must be provided when `num_frames_in` is greater than 0.")
            needs_preprocessing = not (isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[1] == 3)
            if needs_preprocessing:
                video = self.video_processor.preprocess_video(video, height, width)
            video = video.to(device=vae_device, dtype=self.vae.dtype)
            if isinstance(generator, list):
                cond_latents = [
                    retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator=generator[i])
                    for i in range(batch_size)
                ]
            else:
                cond_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)

            latents_mean = self.latents_mean.to(device=vae_device, dtype=dtype)
            latents_std = self.latents_std.to(device=vae_device, dtype=dtype)
            cond_latents = _geco_move_tensor(
                (cond_latents - latents_mean) / latents_std,
                device,
                dtype,
                force_via_cpu=vae_device != torch.device(device),
            )

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

            padding_shape = (B, 1, T, H, W)
            ones_padding = latents.new_ones(padding_shape)
            zeros_padding = latents.new_zeros(padding_shape)

            num_cond_latent_frames = (num_frames_in - 1) // self.vae_scale_factor_temporal + 1
            cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            cond_indicator[:, :, 0:num_cond_latent_frames] = 1.0
            cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

            return (
                latents,
                cond_latents,
                cond_mask,
                cond_indicator,
            )

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        video: list[PipelineImageInput] | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, None], PipelineCallback | MultiPipelineCallbacks] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        conditional_frame_timestep: float = 0.1,
        num_latent_conditional_frames: int = 2,
        fixed_frames: int | list[int] | None = None,
        guidance_step: int | list[int] = 0,
        guidance_lr: float | list[float] = 1e-2,
        loss_fn: str | None = None,
        additional_inputs: dict[str, Any] | None = None,
    ):
        r"""
        The call function to the pipeline for generation. Supports three modes:

        - **Text2World**: `image=None`, `video=None`, `prompt` provided. Generates a world clip.
        - **Image2World**: `image` provided, `video=None`, `prompt` provided. Conditions on a single frame.
        - **Video2World**: `video` provided, `image=None`, `prompt` provided. Conditions on an input clip.

        Set `num_frames=93` (default) to produce a world video, or `num_frames=1` to produce a single image frame (the
        above in "*2Image mode").

        Outputs follow `output_type` (e.g., `"pil"` returns a list of `num_frames` PIL images per prompt).

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                Optional single image for Image2World conditioning. Must be `None` when `video` is provided.
            video (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                Optional input video for Video2World conditioning. Must be `None` when `image` is provided.
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide generation. Required unless `prompt_embeds` is supplied.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                Number of output frames. Use `93` for world (video) generation; set to `1` to return a single frame.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            num_latent_conditional_frames (`int`, defaults to `2`):
                Number of latent conditional frames to use for Video2World conditioning. The number of pixel frames
                extracted from the input video is calculated as `4 * (num_latent_conditional_frames - 1) + 1`. Set to 1
                for Image2World-like behavior (single frame conditioning).

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if self.safety_checker is None:
            logger.warning("Cosmos safety checker is disabled in this experimental guided pipeline.")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        print("[cosmos] encode_prompt start", flush=True)
        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        print("[cosmos] encode_prompt done", flush=True)

        if loss_fn == "residual_motion" and self.text_encoder is not None:
            offload_text_encoder = True
            if additional_inputs is not None:
                offload_text_encoder = bool(additional_inputs.get("offload_text_encoder_after_encode", offload_text_encoder))
            if offload_text_encoder:
                print("[cosmos] offload text_encoder to CPU after prompt encoding", flush=True)
                self.text_encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        transformer_dtype = self.transformer.dtype

        num_frames_in = None
        if image is not None:
            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for image input (given {batch_size})")

            image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
            video = torch.cat([image, torch.zeros_like(image).repeat(num_frames - 1, 1, 1, 1)], dim=0)
            video = video.unsqueeze(0)
            num_frames_in = 1
        elif video is None:
            video = torch.zeros(batch_size, num_frames, 3, height, width, dtype=torch.uint8)
            num_frames_in = 0
        else:
            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for video input (given {batch_size})")

            if num_latent_conditional_frames not in [1, 2]:
                raise ValueError(
                    f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}"
                )

            frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1

            total_input_frames = len(video)

            if total_input_frames < frames_to_extract:
                raise ValueError(
                    f"Input video has only {total_input_frames} frames but Video2World requires at least "
                    f"{frames_to_extract} frames for conditioning."
                )

            num_frames_in = frames_to_extract

        assert video is not None
        video = self.video_processor.preprocess_video(video, height, width)

        # For Video2World: extract last frames_to_extract frames from input, then pad
        if image is None and num_frames_in > 0 and num_frames_in < video.shape[2]:
            video = video[:, :, -num_frames_in:, :, :]

        num_frames_out = num_frames

        if video.shape[2] < num_frames_out:
            n_pad_frames = num_frames_out - video.shape[2]
            last_frame = video[:, :, -1:, :, :]  # [B, C, T==1, H, W]
            pad_frames = last_frame.repeat(1, 1, n_pad_frames, 1, 1)  # [B, C, T, H, W]
            video = torch.cat((video, pad_frames), dim=2)

        assert num_frames_in <= num_frames_out, f"expected ({num_frames_in=}) <= ({num_frames_out=})"

        video = video.to(device=vae_device, dtype=vae_dtype)

        num_channels_latents = self.transformer.config.in_channels - 1
        print("[cosmos] prepare_latents start", flush=True)
        latents, cond_latent, cond_mask, cond_indicator = self.prepare_latents(
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        print("[cosmos] prepare_latents done", flush=True)
        cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
        cond_mask = cond_mask.to(transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        print("[cosmos] denoising setup", flush=True)
        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        allowed_losses = {None, "latent_l2", "residual_motion"}
        if loss_fn not in allowed_losses:
            raise ValueError(f"loss_fn must be one of {allowed_losses}")
        if isinstance(guidance_step, int):
            guidance_step = [guidance_step] * num_inference_steps
        else:
            assert len(guidance_step) == num_inference_steps, "guidance_step length mismatch"
        if isinstance(guidance_lr, (int, float)):
            guidance_lr = [float(guidance_lr)] * num_inference_steps
        else:
            assert len(guidance_lr) == num_inference_steps, "guidance_lr length mismatch"

        transformer_activation_checkpointing = bool(
            additional_inputs is not None and additional_inputs.get("transformer_activation_checkpointing", False)
        )
        vae_activation_checkpointing = bool(
            additional_inputs is not None and additional_inputs.get("vae_activation_checkpointing", False)
        )
        vae_checkpoint_mode = str(
            additional_inputs.get("vae_checkpoint_mode", "whole") if additional_inputs is not None else "whole"
        )
        if vae_checkpoint_mode not in {"whole", "per_tile"}:
            raise ValueError("vae_checkpoint_mode must be 'whole' or 'per_tile'")
        cross_device_grad_via_cpu = bool(
            additional_inputs is not None and additional_inputs.get("cross_device_grad_via_cpu", False)
        )
        debug_guidance_gradient = bool(
            additional_inputs is not None and additional_inputs.get("debug_guidance_gradient", False)
        )
        debug_guidance_stages = bool(
            additional_inputs is not None and additional_inputs.get("debug_guidance_stages", False)
        )
        debug_guidance_dump_dir = (
            additional_inputs.get("debug_guidance_dump_dir", None) if additional_inputs is not None else None
        )
        debug_x0_interval = int(additional_inputs.get("debug_x0_interval", 0) or 0) if additional_inputs else 0
        debug_x0_dir = additional_inputs.get("debug_x0_dir", None) if additional_inputs else None
        debug_x0_frames = additional_inputs.get("debug_x0_frames", None) if additional_inputs else None

        # 这行是在处理 conditioning 区域，也就是 Cosmos I2V / V2W 里“已经给定的首帧或输入视频帧”。
        # latents
        # 当前 noisy latent，包含整段视频 latent。
        # cond_latent
        # 由输入 image/video encode 得到的 conditioning latent，比如第一帧对应的 latent。
        # cond_mask
        # mask，标记哪些 latent token 是 conditioning 区域：
        # cond_mask = 1 -> 给定的 conditioning 帧，不能自由生成
        # cond_mask = 0 -> 需要模型生成的未来帧
        # 所以：
        # (latents - cond_latent)
        # 表示在 conditioning 区域，要从当前 noisy latent 回到给定 condition latent 的“速度/velocity”。
        # 再乘：
        # * cond_mask
        # 只保留 conditioning 区域的 velocity，非 conditioning 区域变成 0。
        gt_velocity = (latents - cond_latent) * cond_mask
        if debug_guidance_gradient:
            condition_fraction_by_latent_t = (
                cond_mask.float().mean(dim=(0, 1, 3, 4)).detach().cpu().tolist()
            )
            print(
                f"[cosmos] condition_fraction_by_latent_t={condition_fraction_by_latent_t}",
                flush=True,
            )

        def _save_x0_debug(step_idx: int, x0_debug: torch.Tensor):
            """Save selected x0 predictions without changing sampling or guidance."""
            if debug_x0_interval <= 0 or debug_x0_dir is None:
                return
            frames_to_save = debug_x0_frames
            if frames_to_save is None:
                frames_to_save = fixed_frames if fixed_frames is not None else [0, num_frames - 1]
            if isinstance(frames_to_save, int):
                frames_to_save = [frames_to_save]
            frames_to_save = [min(max(int(f), 0), num_frames - 1) for f in frames_to_save]

            with torch.no_grad():
                latents_mean = self.latents_mean.to(x0_debug.device, x0_debug.dtype)
                latents_std = self.latents_std.to(x0_debug.device, x0_debug.dtype)
                decode_latents = x0_debug * latents_std + latents_mean
                latent_t = decode_latents.shape[2]
                frames_per_latent = max(int(getattr(self, "vae_scale_factor_temporal", 4) or 4), 1)
                step_dir = os.path.join(str(debug_x0_dir), f"step_{step_idx:03d}")
                os.makedirs(step_dir, exist_ok=True)

                for frame_idx in frames_to_save:
                    latent_id = 0 if frame_idx == 0 else min(latent_t - 1, (frame_idx - 1) // frames_per_latent + 1)
                    # Cosmos uses a causal temporal VAE: retain the target token and past context.
                    lo = max(0, latent_id - 2)
                    hi = latent_id
                    decode_chunk = _geco_move_tensor(
                        decode_latents[:, :, lo : hi + 1].contiguous(),
                        vae_device,
                        self.vae.dtype,
                        force_via_cpu=x0_debug.device != vae_device,
                    )
                    decoded_video = self.vae.decode(decode_chunk, return_dict=False)[0]
                    chunk_num_frames = max(1, (hi - lo) * frames_per_latent + 1)
                    decoded_video = self._match_num_frames(decoded_video, chunk_num_frames)
                    frames_01 = ((decoded_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)
                    local_frame = min(max(frame_idx - lo * frames_per_latent, 0), frames_01.shape[1] - 1)
                    frame_u8 = (frames_01[0, local_frame] * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    Image.fromarray(frame_u8).save(os.path.join(step_dir, f"frame_{frame_idx:03d}.png"))

        print("[cosmos] denoising loop start", flush=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t.cpu().item()

                # NOTE: assumes sigma(t) \in [0, 1]
                sigma_t = (
                    torch.tensor(self.scheduler.sigmas[i].item())
                    .unsqueeze(0)
                    .to(device=device, dtype=transformer_dtype)
                )
                
                def _predict_noise_for_latents(current_latents):
                    # 把 condition latent 和当前 noisy latent 拼成真正送进 transformer 的输入。
                    # cond_mask = 1 的地方，用 cond_latent
                    # cond_mask = 0 的地方，用当前生成 latent current_latents
                    # 也就是：
                    # condition frames 固定
                    # generated frames 用当前 denoising latent
                    # 这个是 Cosmos I2W/V2W 特有的。GeCo/CogVideoX 原来主要是 T2V/I2V，但这个 mask 逻辑是 Cosmos baseline 自带的，必须保留。                    
                    current_in_latents = cond_mask * cond_latent + (1 - cond_mask) * current_latents
                    # 把输入转成 transformer 的 dtype，通常是 bfloat16。这和 baseline pipeline 一致。
                    current_in_latents = current_in_latents.to(transformer_dtype)
                    # 这是 Cosmos conditioning 机制。
                    # cond_indicator = 1 的 latent token，用固定的 conditional_frame_timestep
                    # 其他生成区域，用当前 denoising timestep/sigma
                    # 也就是说 condition 部分告诉模型：“这些是低噪声/条件帧”；生成部分告诉模型：“这些是当前要 denoise 的 noisy latent”。
                    # 这个也是 Cosmos baseline 逻辑，不是我们加的 guidance。                    
                    current_in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t
                    # 这是正向 prompt 的 transformer prediction。Cosmos 是 flow matching，所以这个 current_noise_pred 更准确地说是 velocity / flow prediction，不是 DDPM 里的 epsilon noise。                    
                    def _run_transformer(encoder_hidden_states):
                        def _transformer_forward(hidden_states):
                            return self.transformer(
                                hidden_states=hidden_states,
                                condition_mask=cond_mask,
                                timestep=current_in_timestep,
                                encoder_hidden_states=encoder_hidden_states,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                        if (
                            transformer_activation_checkpointing
                            and torch.is_grad_enabled()
                            and current_in_latents.requires_grad
                        ):
                            return checkpoint(_transformer_forward, current_in_latents, use_reentrant=False)
                        return _transformer_forward(current_in_latents)

                    current_noise_pred = _run_transformer(prompt_embeds)
                    # NOTE: replace velocity with gt_velocity for conditioning inputs only.
                    
                    # IMPORTANT：意思是：
                    # conditioning 区域：不用 transformer 预测，直接用 gt_velocity = (latents - cond_latent) * cond_mask
                    # generated 区域：用 transformer 预测的 velocity
                    # 为什么要这样做：
                    # Cosmos 是 world prediction / I2V 模型，输入第一帧必须保持固定。对 conditioning latent，
                    # 模型不应该“重新生成”它，而是强制 scheduler 把这部分 latent 拉回已知的 cond_latent。
                    # 这样 denoising 过程中第一帧/condition 部分稳定，不会被 guidance 或 transformer 改坏。  
                    # 目的：condition frames 的轨迹被固定，不让模型/ guidance 改坏第一帧或已知输入帧。
                    # 这和后面 guidance 的：
                    # grad = grad * (1 - cond_mask)
                    # 是一致的，都是保护 condition latent。                                                          
                    current_noise_pred = gt_velocity + current_noise_pred * (1 - cond_mask)

                    # 这是 negative prompt 的 transformer prediction
                    if self.do_classifier_free_guidance:
                        current_noise_pred_neg = _run_transformer(negative_prompt_embeds)
                        current_noise_pred_neg = gt_velocity + current_noise_pred_neg * (1 - cond_mask)
                        # 最终预测 = prompt预测 + guidance_scale * (prompt预测 - negative预测)
                        # 直觉上就是：
                        # 往 prompt 想要的方向推远一点，
                        # 远离 negative prompt / 无条件生成方向。                        
                        current_noise_pred = current_noise_pred + self.guidance_scale * (current_noise_pred - current_noise_pred_neg)

                    # 所以 _predict_noise_for_latents() 返回的是：
                    # 当前 latents + prompt + condition frame 下的 CFG flow prediction                        
                    return current_noise_pred

                do_metric_guidance = guidance_step[i] > 0 and loss_fn in {"latent_l2", "residual_motion"}
                if not do_metric_guidance:
                    noise_pred = _predict_noise_for_latents(latents)
                    if debug_x0_interval > 0 and (i % debug_x0_interval == 0 or i == len(timesteps) - 1):
                        x0_debug = latents - self.scheduler.sigmas[i].to(
                            device=latents.device, dtype=latents.dtype
                        ) * noise_pred
                        _save_x0_debug(i, x0_debug)

                if do_metric_guidance:
                    if loss_fn == "residual_motion" and (
                        additional_inputs is None or not callable(additional_inputs.get("residual_motion_metric", None))
                    ):
                        raise ValueError("Pass additional_inputs={residual_motion_metric: callable(frames_01)->scalar_score}")

                    # Guidance is applied to the current noisy latent x_t before the scheduler step.
                    # For Cosmos flow matching, decode the clean estimate x0 ~= x_t - sigma_t * v_theta(x_t),
                    # not x_t itself. Recompute v_theta after each latent update so repeated guidance
                    # iterations do not use stale model predictions.
                    sigma_for_guidance = self.scheduler.sigmas[i].to(device=latents.device, dtype=latents.dtype)

                    for rep in range(guidance_step[i]):
                        with torch.enable_grad():
                            latents_req = latents.detach().requires_grad_(True)
                            noise_pred_for_guidance = _predict_noise_for_latents(latents_req)
                            x0_pred = latents_req - sigma_for_guidance * noise_pred_for_guidance
                            if loss_fn == "latent_l2":
                                loss = x0_pred.float().pow(2).mean()
                            else:
                                frames = fixed_frames
                                if frames is None:
                                    frames = [0, num_frames // 2, num_frames - 1]
                                elif isinstance(frames, int):
                                    frames = [frames]
                                frames = [min(max(int(x), 0), num_frames - 1) for x in frames]

                                latents_mean = self.latents_mean.to(latents_req.device, latents_req.dtype)
                                latents_std = self.latents_std.to(latents_req.device, latents_req.dtype)
                                decode_latents = x0_pred * latents_std + latents_mean

                                # Decode only the selected metric frames. For long videos, using one
                                # min/max chunk over fixed_frames can still span the whole sequence. Instead,
                                # decode a tiny latent neighborhood per selected frame and keep only that frame.
                                latent_t = decode_latents.shape[2]
                                frames_per_latent = max(int(getattr(self, "vae_scale_factor_temporal", 4) or 4), 1)
                                selected_frames_01 = []
                                decoded_chunk_stats = []
                                decode_input_stats = []
                                for f in frames:
                                    if f == 0:
                                        latent_id = 0
                                    else:
                                        latent_id = min(latent_t - 1, (f - 1) // frames_per_latent + 1)
                                    # The temporal VAE is causal: use the target token and its past context.
                                    lo = max(0, latent_id - 2)
                                    hi = latent_id
                                    decode_chunk = decode_latents[:, :, lo : hi + 1].contiguous()
                                    decode_spatial_scale = 1.0
                                    if additional_inputs is not None:
                                        decode_spatial_scale = float(additional_inputs.get("decode_spatial_scale", decode_spatial_scale))
                                    if 0.0 < decode_spatial_scale < 1.0:
                                        _, _, tt, hh, ww = decode_chunk.shape
                                        hh2 = max(1, int(round(hh * decode_spatial_scale)))
                                        ww2 = max(1, int(round(ww * decode_spatial_scale)))
                                        decode_chunk = F.interpolate(
                                            decode_chunk.float(),
                                            size=(tt, hh2, ww2),
                                            mode="trilinear",
                                            align_corners=False,
                                        ).to(dtype=decode_chunk.dtype)
                                    decode_chunk = _geco_move_tensor(
                                        decode_chunk,
                                        vae_device,
                                        self.vae.dtype,
                                        via_cpu_for_grad=cross_device_grad_via_cpu,
                                    )
                                    if debug_guidance_stages:
                                        decode_input_float = decode_chunk.detach().float()
                                        decode_input_stats.append(
                                            (
                                                decode_input_float.abs().mean().item(),
                                                decode_input_float.abs().max().item(),
                                            )
                                        )
                                    if debug_guidance_dump_dir is not None:
                                        # Diagnostic only: preserve the exact VAE input after device transfer.
                                        # This lets native and checkpointed tiled decode be compared offline.
                                        os.makedirs(str(debug_guidance_dump_dir), exist_ok=True)
                                        torch.save(
                                            {
                                                "frame_index": f,
                                                "latent_id": latent_id,
                                                "lo": lo,
                                                "hi": hi,
                                                "decode_chunk": decode_chunk.detach().cpu(),
                                            },
                                            os.path.join(
                                                str(debug_guidance_dump_dir),
                                                f"step_{i:03d}_frame_{f:03d}.pt",
                                            ),
                                        )
                                    if vae_activation_checkpointing:
                                        tile_latent_height = (
                                            self.vae.tile_sample_min_height // self.vae.spatial_compression_ratio
                                        )
                                        tile_latent_width = (
                                            self.vae.tile_sample_min_width // self.vae.spatial_compression_ratio
                                        )
                                        use_per_tile_checkpoint = self.vae.use_tiling and (
                                            decode_chunk.shape[-2] > tile_latent_height
                                            or decode_chunk.shape[-1] > tile_latent_width
                                        )
                                        if vae_checkpoint_mode == "per_tile" and use_per_tile_checkpoint:
                                            # Experimental memory fallback. It mirrors upstream tiling tile-by-tile,
                                            # but the exact native VAE decode below is the default for formal runs.
                                            decoded_video = _geco_tiled_decode_with_per_tile_checkpoint(
                                                self.vae, decode_chunk
                                            )
                                        else:
                                            # Formal guidance uses Diffusers' exact native tiled decode. Checkpointing
                                            # only releases its activations; it does not alter pixels, the VAE cache
                                            # semantics, or the gradient of the GeCo loss with respect to the latent.
                                            def _decode_chunk_for_checkpoint(z):
                                                return self.vae.decode(z, return_dict=False)[0]

                                            decoded_video = checkpoint(
                                                _decode_chunk_for_checkpoint, decode_chunk, use_reentrant=False
                                            )
                                    else:
                                        decoded_video = self.vae.decode(decode_chunk, return_dict=False)[0]
                                    chunk_num_frames = max(1, (hi - lo) * frames_per_latent + 1)
                                    decoded_video = self._match_num_frames(decoded_video, chunk_num_frames)
                                    if debug_guidance_stages:
                                        decoded_float = decoded_video.detach().float()
                                        decoded_chunk_stats.append(
                                            (
                                                decoded_float.abs().mean().item(),
                                                decoded_float.abs().max().item(),
                                                (decoded_float.abs() >= 0.99).float().mean().item(),
                                                (decoded_float.abs() >= 0.9999).float().mean().item(),
                                            )
                                        )
                                    frames_01_all = ((decoded_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)
                                    local_f = min(max(f - lo * frames_per_latent, 0), frames_01_all.shape[1] - 1)
                                    selected_frames_01.append(frames_01_all[:, local_f : local_f + 1])
                                frames_01 = torch.cat(selected_frames_01, dim=1)
                                score = additional_inputs["residual_motion_metric"](frames_01)
                                if not score.requires_grad:
                                    raise RuntimeError("Residual motion metric returned a detached score.")
                                loss = -score

                            if debug_guidance_stages:
                                # This is deliberately opt-in: each VJP traverses the full guided graph.
                                # It separates a Transformer/x0 Jacobian issue from a VAE or RGB-metric issue.
                                def _stage_norm(value):
                                    return "None" if value is None else f"{value.float().norm().item():.8e}"

                                if loss_fn == "residual_motion":
                                    x0_float = x0_pred.detach().float()
                                    z_float = decode_latents.detach().float()
                                    frame_float = frames_01.detach().float()
                                    chunk_abs_mean = sum(item[0] for item in decoded_chunk_stats) / len(decoded_chunk_stats)
                                    chunk_near_sat = sum(item[2] for item in decoded_chunk_stats) / len(decoded_chunk_stats)
                                    chunk_at_sat = sum(item[3] for item in decoded_chunk_stats) / len(decoded_chunk_stats)
                                    chunk_z_abs_mean = sum(item[0] for item in decode_input_stats) / len(decode_input_stats)
                                    chunk_z_abs_max = max(item[1] for item in decode_input_stats)
                                    print(
                                        f"[cosmos] guidance_values step={i} "
                                        f"x0_mean_abs={x0_float.abs().mean().item():.8e} "
                                        f"x0_max_abs={x0_float.abs().max().item():.8e} "
                                        f"vae_z_mean_abs={z_float.abs().mean().item():.8e} "
                                        f"vae_z_max_abs={z_float.abs().max().item():.8e} "
                                        f"chunk_z_mean_abs={chunk_z_abs_mean:.8e} "
                                        f"chunk_z_max_abs={chunk_z_abs_max:.8e} "
                                        f"decoded_abs_mean={chunk_abs_mean:.8e} "
                                        f"decoded_near_sat={chunk_near_sat:.6%} "
                                        f"decoded_at_sat={chunk_at_sat:.6%} "
                                        f"frames_min={frame_float.min().item():.8e} "
                                        f"frames_max={frame_float.max().item():.8e}",
                                        flush=True,
                                    )

                                loss_stage_grads = torch.autograd.grad(
                                    loss,
                                    [frames_01, decode_latents, x0_pred, latents_req],
                                    retain_graph=True,
                                    allow_unused=True,
                                )
                                x0_probe_grad = torch.autograd.grad(
                                    x0_pred.float().square().mean(),
                                    latents_req,
                                    retain_graph=True,
                                    allow_unused=True,
                                )[0]
                                print(
                                    f"[cosmos] guidance_stages step={i} "
                                    f"dL_dRGB={_stage_norm(loss_stage_grads[0])} "
                                    f"dL_dDecodeLatents={_stage_norm(loss_stage_grads[1])} "
                                    f"dL_dx0={_stage_norm(loss_stage_grads[2])} "
                                    f"dL_dxt={_stage_norm(loss_stage_grads[3])} "
                                    f"d_x0_l2_dxt={_stage_norm(x0_probe_grad)}",
                                    flush=True,
                                )
                            grad = torch.autograd.grad(loss, latents_req, retain_graph=False, create_graph=False)[0]
                        raw_grad_norm = grad.float().norm().item() if debug_guidance_gradient else None
                        grad = grad * (1 - cond_mask)
                        if debug_guidance_gradient:
                            print(
                                f"[cosmos] guidance_grad step={i} raw_norm={raw_grad_norm:.8e} "
                                f"generated_norm={grad.float().norm().item():.8e}",
                                flush=True,
                            )
                        latent_mean_abs = latents_req.detach().float().abs().mean().item()
                        grad_f = grad.float()
                        grad_norm = grad_f.norm() + 1e-8
                        grad_mean_abs = grad_f.abs().mean().item()
                        grad_max_abs = grad_f.abs().max().item()
                        latents_before_guidance = latents
                        latents = (latents - float(guidance_lr[i]) * grad / grad_norm).detach()
                        latent_delta = (latents - latents_before_guidance).float().abs().mean().item()
                        relative_delta = latent_delta / (latent_mean_abs + 1e-8) * 100.0
                        relative_grad = grad_mean_abs / (latent_mean_abs + 1e-8) * 100.0
                        print(
                            f"cosmos_guidance_loss({i}/{rep}): {loss.item():.6f} "
                            f"sigma={float(sigma_for_guidance):.6f} "
                            f"latent_mean_abs={latent_mean_abs:.6f} "
                            f"grad_norm={float(grad_norm):.6f} grad_mean_abs={grad_mean_abs:.10f} grad_max_abs={grad_max_abs:.8f} "
                            f"latent_delta={latent_delta:.8f} relative_delta={relative_delta:.6f}% relative_grad={relative_grad:.8f}%",
                            flush=True,
                        )

                if do_metric_guidance:
                    # Use the updated latent for the actual scheduler step as well.
                    with torch.no_grad():
                        noise_pred = _predict_noise_for_latents(latents)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents_mean = self.latents_mean.to(latents.device, latents.dtype)
            latents_std = self.latents_std.to(latents.device, latents.dtype)
            latents = latents * latents_std + latents_mean
            decode_latents = _geco_move_tensor(
                latents,
                vae_device,
                self.vae.dtype,
                force_via_cpu=latents.device != vae_device,
            )
            video = self.vae.decode(decode_latents, return_dict=False)[0]
            video = self._match_num_frames(video, num_frames)

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)

    def _match_num_frames(self, video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
        if target_num_frames <= 0 or video.shape[2] == target_num_frames:
            return video

        frames_per_latent = max(self.vae_scale_factor_temporal, 1)
        video = torch.repeat_interleave(video, repeats=frames_per_latent, dim=2)

        current_frames = video.shape[2]
        if current_frames < target_num_frames:
            pad = video[:, :, -1:, :, :].repeat(1, 1, target_num_frames - current_frames, 1, 1)
            video = torch.cat([video, pad], dim=2)
        elif current_frames > target_num_frames:
            video = video[:, :, :target_num_frames]

        return video
