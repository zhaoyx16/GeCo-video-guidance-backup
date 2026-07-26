# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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


'''
# pipeline 里真正插入 guidance
'''
import html
import os
from typing import Any, Callable, Mapping

import PIL
import regex as re
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _geco_move_tensor(
    tensor: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    *,
    via_cpu_for_grad: bool = False,
    via_cpu: bool = False,
) -> torch.Tensor:
    """Move a tensor while preserving valid multi-GPU values and VJPs."""
    target = torch.device(device)
    if tensor.device == target:
        return tensor if dtype is None else tensor.to(dtype=dtype)
    if via_cpu or (via_cpu_for_grad and torch.is_grad_enabled() and tensor.requires_grad):
        # Direct peer copies produced invalid values (conditioning) and invalid VJPs
        # (guidance) on this host. CPU staging preserves the same tensor values and,
        # when gradients are enabled, retains the cross-device autograd bridge.
        staged = tensor.to("cpu")
        return staged.to(device=target, dtype=dtype) if dtype is not None else staged.to(target)
    return tensor.to(device=target, dtype=dtype) if dtype is not None else tensor.to(target)


def _prepare_frame_guidance_targets(
    video_processor: VideoProcessor,
    raw_targets: Mapping[int | str, PipelineImageInput],
    fixed_frames: list[int],
    num_frames: int,
    height: int,
    width: int,
    vae_device: torch.device,
) -> dict[int, torch.Tensor]:
    """Preprocess immutable RGB anchors once for Frame Guidance MSE.

    Targets are stored as ``[B, H, W, C]`` in the VideoProcessor's native
    ``[-1, 1]`` range, matching the official Frame Guidance frame loss.  The
    target tensors have no gradient; only decoded x0 predictions are optimized.
    """
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise ValueError("Frame Guidance requires additional_inputs['frame_guidance_targets'].")

    fixed_set = set(fixed_frames)
    targets: dict[int, torch.Tensor] = {}
    for raw_index, raw_target in raw_targets.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Frame Guidance target index is not an integer: {raw_index!r}") from error
        if index < 0 or index >= num_frames:
            raise ValueError(f"Frame Guidance target index {index} is outside [0, {num_frames - 1}].")
        if index not in fixed_set:
            raise ValueError(
                f"Frame Guidance target index {index} is missing from fixed_frames={sorted(fixed_set)}."
            )
        with torch.no_grad():
            target = video_processor.preprocess(raw_target, height=height, width=width).to(
                vae_device, dtype=torch.float32
            )
        if target.ndim != 4 or target.shape[0] != 1 or target.shape[1] != 3:
            raise ValueError(
                f"Frame Guidance target {index} must preprocess to [1, 3, H, W], got {tuple(target.shape)}."
            )
        targets[index] = target.permute(0, 2, 3, 1).contiguous()

    if not any(index > 0 for index in targets):
        raise ValueError("Frame Guidance needs at least one target after frame zero.")
    return targets


def _wan_causal_frame_decode_plan(
    frame_index: int,
    *,
    num_frames: int,
    latent_frames: int,
    temporal_scale: int,
) -> tuple[int, int, int]:
    """Return a causal Wan latent slice and local decoded-frame index.

    ``AutoencoderKLWan._decode`` clears its cache, decodes latent token zero as
    a one-frame initial chunk, then decodes every later token as a four-frame
    chunk.  Consequently, RGB frame ``f > 0`` belongs to latent token
    ``(f - 1) // temporal_scale + 1``.  To decode that target after a cache
    reset, the Frame Guidance convention is to pass the predecessor/target
    pair and select local output ``(f - 1) % temporal_scale + 1``.  This is the
    same index algebra used by the official Wan Frame Guidance implementation.

    The pair provides the correct *local output slot*.  It does not promise
    exact equality with a full causal decode for later frames because the
    decoder cache before the predecessor is intentionally absent; the optional
    parity probe reports that approximation separately.
    """
    if temporal_scale <= 0:
        raise ValueError(f"Wan temporal_scale must be positive, got {temporal_scale}.")
    if latent_frames <= 0:
        raise ValueError(f"Wan latent_frames must be positive, got {latent_frames}.")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(f"Frame index {frame_index} is outside [0, {num_frames - 1}].")

    max_decodable_frame = temporal_scale * (latent_frames - 1)
    if frame_index > max_decodable_frame:
        raise ValueError(
            f"Frame {frame_index} cannot be represented by {latent_frames} Wan latent frames "
            f"at temporal_scale={temporal_scale}."
        )
    if frame_index == 0:
        return 0, 1, 0

    target_latent = (frame_index - 1) // temporal_scale + 1
    if target_latent >= latent_frames:
        raise ValueError(
            f"Frame {frame_index} maps to missing Wan latent token {target_latent}; "
            f"only [0, {latent_frames - 1}] are available."
        )
    return target_latent - 1, target_latent + 1, (frame_index - 1) % temporal_scale + 1


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

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import CLIPVisionModel

        >>> # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanImageToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))
        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


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


class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. In two-stage denoising,
            `transformer` handles high-noise stages and `transformer_2` handles low-noise stages. If not provided, only
            `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
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
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
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
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def _get_geco_vae_device(self):
        return getattr(self, "_geco_vae_device", self._execution_device)

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.config.expand_timesteps:
            video_condition = image

        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        vae_device = self._get_geco_vae_device()
        # The conditioning image originates on the transformer device. On this host,
        # direct GPU-to-GPU copies can make the I2V condition non-finite.
        video_condition = _geco_move_tensor(
            video_condition, vae_device, self.vae.dtype, via_cpu=True
        )

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(vae_device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            vae_device, self.vae.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(device=vae_device, dtype=self.vae.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std
        # Send the VAE-encoded condition back to the transformer through CPU for the
        # same peer-copy correctness reason; this is outside the guidance objective.
        latent_condition = _geco_move_tensor(latent_condition, device, dtype, via_cpu=True)

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        fixed_frames: int | list[int] | None = None,
        guidance_step: int | list[int] = 0,
        guidance_lr: float | list[float] = 1e-2,
        loss_fn: str | None = None,
        additional_inputs: dict[str, Any] | None = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
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
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) for proper patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # 检查 guidance
        # 作用：保证 loss_fn 合法，并且每个 denoising step 都有对应的 guidance_step 和 guidance_lr。
        allowed_losses = {None, "latent_l2", "residual_motion", "frame", "frame_residual_motion"}
        if loss_fn not in allowed_losses:
            raise ValueError(f"loss_fn must be one of {allowed_losses}")

        if isinstance(guidance_step, int):
            guidance_step = [guidance_step] * num_inference_steps
        else:
            assert len(guidance_step) == num_inference_steps, "guidance_step length mismatch"

        if isinstance(guidance_lr, float):
            guidance_lr = [guidance_lr] * num_inference_steps
        else:
            assert len(guidance_lr) == num_inference_steps, "guidance_lr length mismatch"

        # Frame Guidance uses the same predicted x0 and latent update as the audited
        # RGB-GeCo path.  Only the loss term differs.  Preprocess fixed target frames
        # once, after the pipeline's final height/width adjustment, so the MSE target
        # is exactly aligned with the decoded x0 resolution.
        uses_frame_guidance = loss_fn in {"frame", "frame_residual_motion"}
        uses_residual_motion = loss_fn in {"residual_motion", "frame_residual_motion"}
        frame_guidance_targets = None
        frame_loss_weight = 1.0
        geco_loss_weight = 1.0
        frame_guidance_fixed_frames = None
        if uses_frame_guidance:
            if additional_inputs is None:
                raise ValueError("Frame Guidance requires additional_inputs with frame_guidance_targets.")
            if fixed_frames is None:
                raise ValueError("Frame Guidance requires fixed_frames matching its target anchor indices.")
            if isinstance(fixed_frames, int):
                frame_guidance_fixed_frames = [int(fixed_frames)]
            else:
                frame_guidance_fixed_frames = [int(index) for index in fixed_frames]
            if len(set(frame_guidance_fixed_frames)) != len(frame_guidance_fixed_frames):
                raise ValueError("Frame Guidance fixed_frames must not contain duplicates.")
            if any(index < 0 or index >= num_frames for index in frame_guidance_fixed_frames):
                raise ValueError(
                    f"Frame Guidance fixed_frames must lie in [0, {num_frames - 1}], got {frame_guidance_fixed_frames}."
                )
            frame_loss_weight = float(additional_inputs.get("frame_loss_weight", 1.0))
            geco_loss_weight = float(additional_inputs.get("geco_loss_weight", 1.0))
            if frame_loss_weight <= 0:
                raise ValueError("frame_loss_weight must be positive.")
            if loss_fn == "frame_residual_motion" and geco_loss_weight <= 0:
                raise ValueError("geco_loss_weight must be positive for frame_residual_motion.")
            frame_guidance_targets = _prepare_frame_guidance_targets(
                self.video_processor,
                additional_inputs.get("frame_guidance_targets"),
                frame_guidance_fixed_frames,
                num_frames,
                height,
                width,
                self._get_geco_vae_device(),
            )
            if 0 not in frame_guidance_fixed_frames:
                raise ValueError("Frame Guidance fixed_frames must include the frame-0 conditioning anchor.")
            missing_frame_targets = [
                index for index in frame_guidance_fixed_frames if index not in frame_guidance_targets
            ]
            if missing_frame_targets:
                raise ValueError(
                    "Frame Guidance is missing targets for fixed_frames="
                    f"{missing_frame_targets}."
                )

        # 冻结模型参数： freeze transformer 和 VAE 权重，只更新 latent
        # 作用：不训练 transformer / VAE 权重。后面只更新当前 latents。
        # 注意：VAE 参数冻结，但梯度仍然可以通过 VAE decode 传回 latent。
        self.transformer.requires_grad_(False)
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.requires_grad_(False)
        self.vae.requires_grad_(False)

        # 这段是 Wan 原本的 denoising step，也就是每个 timestep 怎么从当前 latent 预测下一步要用的 noise_pred
        # 在当前 denoising step，Wan 根据当前 noisy latent、condition image、text prompt 和 timestep，
        # 选择合适的 transformer，预测当前 step 的 guided model output noise_pred。这个 noise_pred 后面会被 scheduler 用来更新 latents。
        # 和我们 guidance 的关系：
        # 这段是原始 Wan 的正常 denoising。后来插入的 GeCo guidance，就是在这个 noise_pred 算完之后、scheduler.step 之前，对 latents 做一次额外梯度更新。
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # 进入每个 denoising step，并选择当前用哪个 transformer
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                # 准备 transformer 输入
                if self.config.expand_timesteps:
                    # 这行是在把 condition 部分 和 要生成的 noisy latent 部分 合成 transformer 输入。
                    # first_frame_mask = 0 的位置：用 condition
                    # first_frame_mask = 1 的位置：用 current latents
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    # 转成 transformer 的 dtype，通常是 bfloat16
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    # 构造每个 latent token 对应的 timestep
                    # 为什么乘 first_frame_mask：
                    # condition frame 的 timestep 应该接近 0 或特殊值
                    # generated latent 的 timestep 是当前 t
                    # [:, ::2, ::2] 是因为 transformer patch/token 下采样，timestep map 要和 transformer token grid 对齐。
                    # 它在给 condition 区域和 generated 区域分配不同 timestep。
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # temp_ts:   [seq_len]
                    # timestep:  [batch_size, seq_len]
                    # 把一条 token-level timestep sequence 复制到 batch 维度上，
                    # 让每个 batch sample 都有同样的 per-token timestep。
                    # 在这个 Wan I2V 分支里，timestep 不是一个单独数字，而是每个 token 都可以有自己的 timestep：
                    # condition token: 0
                    # generated token: t
                    # 这就是为什么它需要 [batch, seq_len]
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # expand_timesteps=False:
                    # condition 和 generated latent 在 channel 维拼接；
                    # timestep 是 sample-level，整个输入都用同一个 t。
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])
                # conditional prediction
                # 这里真正 forward transformer
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                # 如果启用 classifier-free guidance，就还要跑一次 unconditional prediction branch
                if self.do_classifier_free_guidance:
                    # 这次 forward 用的是 negative/unconditional prompt embedding
                    # 区别是：
                    # cond:   text prompt = 你的 prompt
                    # uncond: text prompt = negative prompt / empty prompt
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        # 标准 classifier-free guidance 公式。
                        # 可以写成：guided = uncond + scale * (cond - uncond)
                        # cond = 通用生成趋势 + prompt 影响
                        # uncond = 通用生成趋势
                        # 含义： cond - uncond = prompt 指向的方向； scale 越大，越强化 prompt 约束
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                if additional_inputs is not None:
                    debug_x0_interval = int(additional_inputs.get("debug_x0_interval", 0) or 0)
                else:
                    debug_x0_interval = 0
                if debug_x0_interval > 0 and (i % debug_x0_interval == 0 or i == len(timesteps) - 1):
                    debug_x0_dir = additional_inputs.get("debug_x0_dir", None)
                    if debug_x0_dir is not None:
                        with torch.no_grad():
                            if self.scheduler.step_index is None:
                                self.scheduler._init_step_index(t)
                            x0_debug = self.scheduler.convert_model_output(noise_pred.float().detach(), sample=latents.float())
                            if self.config.expand_timesteps:
                                x0_debug = (1 - first_frame_mask.float()) * condition.float() + first_frame_mask.float() * x0_debug.float()

                            debug_frames = additional_inputs.get("debug_x0_frames", None)
                            if debug_frames is None:
                                debug_frames = fixed_frames if fixed_frames is not None else [0, num_frames - 1]
                            if isinstance(debug_frames, int):
                                debug_frames = [debug_frames]
                            debug_frames = [min(max(int(x), 0), num_frames - 1) for x in debug_frames]

                            debug_decode_spatial_scale = float(additional_inputs.get("debug_x0_decode_spatial_scale", 1.0) or 1.0)
                            T_lat = x0_debug.shape[2]
                            temporal_scale = int(getattr(self, "vae_scale_factor_temporal", 4))
                            vae_dtype = self.vae.dtype
                            vae_device = self._get_geco_vae_device()
                            latents_mean = (
                                torch.tensor(self.vae.config.latents_mean)
                                .view(1, self.vae.config.z_dim, 1, 1, 1)
                                .to(vae_device, vae_dtype)
                            )
                            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                                1, self.vae.config.z_dim, 1, 1, 1
                            ).to(vae_device, vae_dtype)

                            step_dir = os.path.join(str(debug_x0_dir), f"step_{i:03d}")
                            os.makedirs(step_dir, exist_ok=True)
                            for fidx in debug_frames:
                                if uses_frame_guidance:
                                    chunk_start, chunk_end, rel = _wan_causal_frame_decode_plan(
                                        fidx,
                                        num_frames=num_frames,
                                        latent_frames=T_lat,
                                        temporal_scale=temporal_scale,
                                    )
                                else:
                                    if fidx == 0:
                                        center_lat = 0
                                    else:
                                        center_lat = min(T_lat - 1, (fidx - 1) // temporal_scale + 1)
                                    chunk_start = max(0, center_lat - 1)
                                    chunk_end = min(T_lat, center_lat + 2)
                                z_chunk = _geco_move_tensor(
                                    x0_debug[:, :, chunk_start:chunk_end].contiguous(),
                                    vae_device,
                                    vae_dtype,
                                    via_cpu=True,
                                )
                                if debug_decode_spatial_scale != 1.0:
                                    H_lat, W_lat = z_chunk.shape[-2:]
                                    H_new = max(1, int(round(H_lat * debug_decode_spatial_scale)))
                                    W_new = max(1, int(round(W_lat * debug_decode_spatial_scale)))
                                    z_chunk = torch.nn.functional.interpolate(
                                        z_chunk,
                                        size=(z_chunk.shape[2], H_new, W_new),
                                        mode="trilinear",
                                        align_corners=False,
                                    )
                                z_chunk = z_chunk / latents_std + latents_mean
                                decoded_chunk = self.vae.decode(z_chunk, return_dict=False)[0]
                                frames_chunk = ((decoded_chunk.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)
                                if uses_frame_guidance:
                                    if rel >= frames_chunk.shape[1]:
                                        raise RuntimeError(
                                            f"Wan Frame Guidance debug slice for frame {fidx} produced "
                                            f"{frames_chunk.shape[1]} frames, but needs local frame {rel}."
                                        )
                                else:
                                    if chunk_start == 0:
                                        chunk_first_frame = 0
                                    else:
                                        chunk_first_frame = 1 + (chunk_start - 1) * temporal_scale
                                    rel = fidx - chunk_first_frame
                                    rel = int(max(0, min(rel, frames_chunk.shape[1] - 1)))
                                frame = frames_chunk[0, rel]
                                frame_u8 = (frame * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                                PIL.Image.fromarray(frame_u8).save(os.path.join(step_dir, f"frame_{fidx:03d}.png"))
                                del z_chunk, decoded_chunk, frames_chunk, frame, frame_u8
                            del x0_debug

                # 826-970   加的 guidance block
                # Optional train-free latent guidance smoke test.
                # This verifies: latent -> Wan x0 prediction -> loss -> grad -> latent update.
                if guidance_step[i] > 0 and loss_fn in {
                    "latent_l2",
                    "residual_motion",
                    "frame",
                    "frame_residual_motion",
                }:
                    # guidance_step 是一个 list，长度等于 num_inference_steps。它告诉 pipeline：
                    # 每个 denoising timestep 做几次 guidance update。
                    for rep in range(guidance_step[i]):
                        # 把当前 latent 从旧计算图里切出来，然后打开梯度，只让当前 latents 可求梯度（把 latents 当作 optimization variable）
                        # detach()：避免把梯度传回整个 denoising history，省显存，也符合 training-free guidance。
                        # requires_grad_(True)：我们只想对当前 latent 求梯度。
                        # 整行完整意思是：把当前 noisy latent 从旧 denoising graph 里切出来，然后把它设为当前 guidance step 的可优化变量。
                        latents = latents.detach().requires_grad_(True)

                        # 用当前 latent 准备 transformer 输入
                        # 意思：构造 Wan transformer 当前 step 的输入。原 pipeline 怎么构造 input，guidance 就怎么构造 input。
                        # 我们的 guidance forward 必须和原始 denoising forward 用同样的输入格式，否则 noise_pred_g 就不是同一个模型分布下的 prediction。
                        # 把当前 latent + condition + timestep 整理成 Wan transformer 需要的输入格式。
                        # 这几行不是 GeCo 核心，是为了保持和原 Wan pipeline 的输入格式一致。
                        # 为什么有两种：
                        # expand_timesteps=True：Wan I2V 的 condition frame 和 generated latent 共享一个 expanded timestep/mask 机制。
                        if self.config.expand_timesteps: #这个分支通常用于 I2V / first-frame conditioning 的特殊输入方式。
                            # 用 mask 混合 condition 和当前 latent
                            # 这保证 transformer 看到的是：已知部分固定，未知部分正在 denoise。
                            latent_model_input_g = (1 - first_frame_mask) * condition + first_frame_mask * latents
                            # 把输入转成 transformer 的 dtype，比如 bfloat16
                            # 原因： transformer 权重通常是 bf16; 输入 dtype 匹配更省显存，也避免 dtype mismatch
                            latent_model_input_g = latent_model_input_g.to(transformer_dtype)
                            # 构造 token-level timestep
                            # first_frame_mask[0][0]取出 batch 0、channel 0 的 mask，形状大概：[T, H, W]
                            # [:, ::2, ::2]空间下采样 2 倍，匹配 transformer token grid：[T, H, W] -> [T, H/2, W/2]
                            #  * t把 mask 变成 timestep：mask = 0 -> timestep = 0; mask = 1 -> timestep = t
                            # 也就是：condition token 的 timestep = 0; generated token 的 timestep = 当前 t
                            # .flatten()展平成 token sequence：[T, H/2, W/2] -> [seq_len]
                            # 得到的是：
                            # temp_ts_g.shape = [seq_len]
                            # 也就是每个 transformer token 一个 timestep。
                            # 比如：
                            # temp_ts_g = [0, 0, t, t, t, ...]
                            # T = latent video 的时间维度，表示这个 latent tensor 里有多少个时间 token。
                            # Wan 里的 latent 通常形状是：latents.shape = [B, C, T, H, W]
                            # B = batch size; C = latent channels; T = latent temporal length; H = latent height; W = latent width
                            # 一个 channel 可以理解为一个 learned latent feature map；每个时空位置由 C 个 channel 的特征向量表示。
                            temp_ts_g = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                            # 整行意思是：把一条 token-level timestep 序列复制/扩展到 batch 维度，
                            # 让 batch 里的每个 sample 都有同样的 per-token timestep map。最后：timestep_g.shape = [B, seq_len]
                            # temp_ts_g.unsqueeze(0)
                            # 在第 0 维加一个新维度：[seq_len] -> [1, seq_len]
                            # 为什么？ 因为 transformer 通常需要 batch 维：[batch_size, seq_len]
                            # .expand(latents.shape[0], -1)
                            # latents.shape[0] 是 batch size，也就是 B。
                            # -1 的意思是这一维保持原大小，不改变。
                            # 所以：[1, seq_len] -> [B, seq_len]
                            # 所以在这个分支里：
                            # timestep_g 是 per-token timestep
                            # 每个 token 可以有不同 timestep
                            # condition token 是 0
                            # generated token 是 t
                            timestep_g = temp_ts_g.unsqueeze(0).expand(latents.shape[0], -1)
                        else:  #分支 2：expand_timesteps=False ， 否则：把 latents 和 condition 在 channel 维拼起来。
                            # dim=1 表示沿着 channel 维度 拼接。
                            # 原来：
                            # latents.shape   = [B, C1, T, H, W]
                            # condition.shape = [B, C2, T, H, W]
                            # 拼接后：
                            # latent_model_input.shape = [B, C1 + C2, T, H, W]
                            # 其他维度不变， 只有 channel 数加起来。
                            # 可以想成在每个时空位置 (t,h,w)：
                            # latents   给一个 C1 维 feature vector
                            # condition 给一个 C2 维 feature vector
                            # 拼接后：
                            # [C1 + C2] 维 feature vector
                            # 所以这不是在时间上拼，不是在空间上拼，而是在 feature/channel 维度上拼。
                            # 也就是说 transformer 一次性看到：
                            # 当前 noisy latent
                            # condition latent
                            latent_model_input_g = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                            # 这里 timestep 是 sample-level，不是 token-level。形状：[B]
                            # 意思是：整个 sample 都使用同一个 timestep t
                            # 没有区分：
                            # condition token = 0
                            # generated token = t
                            timestep_g = t.expand(latents.shape[0])

                        # 重新 forward transformer 得到当前 latent 对应的 model prediction： noise_pred_g
                        # 基于当前 latents 重新预测当前 timestep 的 model output，并做 CFG。
                        # Full guidance keeps the transformer forward in the autograd graph.
                        # Transformer weights stay frozen, but loss still needs d(noise_pred)/d(latent).
                        # 为什么要每次 repeat 都算：
                        # 因为 latent 每次 update 后变了，对应的 noise_pred 也应该变。否则就是旧 prediction 配新 latent，梯度方向会错。
                        with torch.enable_grad():
                            with current_model.cache_context("cond"): #这是 Wan transformer 的 cache 管理上下文
                            # cond 表示：conditional branch, 也就是使用 prompt embedding 的那次 forward。
                            # 因为 CFG 有两次 forward：
                            # cond branch:   用 prompt
                            # uncond branch: 用 negative prompt / empty prompt
                            # 所以 Wan 用：
                            # cache_context("cond")
                            # cache_context("uncond")
                            # 来区分缓存。

                                # 输出：noise_pred_g
                                # 名字叫 noise_pred，但对 Wan flow-matching 来说更接近：predicted velocity / flow / model output
                                # 它后面会被 scheduler 转成：x0_pred
                                noise_pred_g = current_model(
                                    hidden_states=latent_model_input_g,  # 当前 latent + condition
                                    timestep=timestep_g,  # 当前 noise level
                                    encoder_hidden_states=prompt_embeds,  # text prompt embedding
                                    encoder_hidden_states_image=image_embeds, # image condition embedding
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]
                            # 标准 classifier-free guidance。
                            if self.do_classifier_free_guidance:  #这段是 Classifier-Free Guidance, CFG。它的目的：让生成结果更听 prompt 的话
                            # 这段用 conditional prediction 和 unconditional prediction 的差值提取 prompt 方向，并用 guidance_scale 放大它，得到更符合 prompt 的 model output
                            # 在我们代码里为什么要做 CFG？因为这是原 Wan pipeline 原本就有的 text guidance。
                            # 在 guidance block 里也要重新做 CFG，是因为我们需要当前 latent 对应的正确 model output：
                            # 当前 latent + prompt guidance -> noise_pred_g。
                            # 如果这里不做 CFG，x0_pred 就不是原 pipeline 真正会用的 guided prediction，和正常 denoising 不一致。
                                with current_model.cache_context("uncond"):
                                    noise_uncond_g = current_model(
                                        hidden_states=latent_model_input_g,
                                        timestep=timestep_g,
                                        encoder_hidden_states=negative_prompt_embeds,
                                        encoder_hidden_states_image=image_embeds,
                                        attention_kwargs=attention_kwargs,
                                        return_dict=False,
                                    )[0]
                                # 数学上： guided_pred = uncond + scale * (cond - uncond)
                                # 为什么：保持和原始 Wan sampling 完全一致。我们不是改变 text guidance，
                                # 只是在它的 denoising prediction 上额外加 GeCo latent guidance。
                                noise_pred_g = noise_uncond_g + current_guidance_scale * (noise_pred_g - noise_uncond_g)

                        # 用 scheduler 得到 x0_pred
                        # 用 scheduler 把当前 noisy latent 和 model output 转成 predicted clean latent x0_pred。
                        with torch.enable_grad():
                            # 因为我们在 scheduler.step 之前提前调用 scheduler.convert_model_output，
                            # 所以先手动初始化 scheduler.step_index，让它用当前 timestep 对应的 sigma 来计算 x0_pred。
                            if self.scheduler.step_index is None:
                                # 意思：确保 scheduler 知道当前是第几个 step。
                                # 为什么：Wan 的 scheduler 内部用 step_index 找当前 sigma。没有初始化会取不到正确 sigma。
                                # Wan 用的是 Diffusers scheduler。scheduler 里面有一串：
                                # self.sigmas
                                # 表示每个 inference step 对应的 noise level。

                                # 平时 step_index 什么时候初始化？
                                # 正常情况下，scheduler 在：
                                # self.scheduler.step(noise_pred, t, latents)
                                # 里面会初始化/更新 step_index。
                                # 但我们的 guidance 插在：
                                # scheduler.step 之前
                                # 我们提前调用了：
                                # self.scheduler.convert_model_output(...)
                                # 所以这时候 step_index 可能还没被 scheduler 初始化。
                                # 因此要手动做：
                                # self.scheduler._init_step_index(t)
                                # 如果没有这两行会怎样？
                                # 可能会出现：
                                # self.scheduler.step_index is None
                                # 然后 convert_model_output 内部找 sigma 时失败。
                                # 或者更隐蔽地：
                                # 用错 sigma
                                # 那 x0_pred 就算错了，后面的 VAE decode 和 GeCo loss 都不可信。

                                self.scheduler._init_step_index(t)
                            # 这行是 guidance 里非常关键的一步：把当前 noisy latent x_t 和 model output 转成 predicted clean sample x0_pred。
                            # 为什么不用直接 decode latents： latents 是 noisy intermediate。
                            # GeCo/VGGT/UFM 需要看的是“当前模型认为最终视频大概长什么样”。
                            # 所以应该 decode x0_pred，不是 decode noisy x_t。
                            # Do not detach noise_pred_g: transformer weights are frozen, but the latent update
                            # should include the transformer Jacobian d(noise_pred)/d(latent).
                            x0_pred = self.scheduler.convert_model_output(
                                # noise_pred_g 是 Wan transformer 对当前 step 的 model output。对 Wan flow-matching 来说，它更像：
                                # velocity / flow prediction, 虽然变量名叫 noise_pred。
                                # =====FIXME+TODO：应该需要不detach noise_pred_g， 考虑transformer的prediction的影响，这样计算的梯度更准确=====
                                # IMPORTANT: 原 GeCo 是考虑 transformer Jacobian 的
                                noise_pred_g.float(),
                                # latents是当前 timestep 的 noisy latent：x_t，也就是还在 denoising 中间过程里的 latent，不是最终干净视频。
                                sample=latents.float()
                                )
                            debug_guidance_consistency = bool(
                                additional_inputs and additional_inputs.get("debug_guidance_consistency", False)
                            )
                            if debug_guidance_consistency and rep == 0:
                                # Debug only: the recomputed differentiable prediction must agree with
                                # the ordinary denoising prediction before its latent update. This does
                                # not enter the loss or alter the guidance update.
                                with torch.no_grad():
                                    x0_from_sampling_pred = self.scheduler.convert_model_output(
                                        noise_pred.float(), sample=latents.float()
                                    )
                                    pred_abs_diff = (noise_pred_g.detach().float() - noise_pred.detach().float()).abs()
                                    x0_abs_diff = (x0_pred.detach().float() - x0_from_sampling_pred.float()).abs()
                                    sampling_finite = torch.isfinite(noise_pred.detach()).float().mean().item()
                                    guidance_finite = torch.isfinite(noise_pred_g.detach()).float().mean().item()
                                    sampling_abs_mean = torch.nan_to_num(noise_pred.detach().float()).abs().mean().item()
                                    guidance_abs_mean = torch.nan_to_num(noise_pred_g.detach().float()).abs().mean().item()
                                    print(
                                        f"wan_guidance_prediction_compare({i}/{rep}): "
                                        f"sampling_finite={sampling_finite:.6f} "
                                        f"guidance_finite={guidance_finite:.6f} "
                                        f"sampling_abs_mean={sampling_abs_mean:.8e} "
                                        f"guidance_abs_mean={guidance_abs_mean:.8e} "
                                        f"pred_mean_abs_diff={pred_abs_diff.mean().item():.8e} "
                                        f"pred_max_abs_diff={pred_abs_diff.max().item():.8e} "
                                        f"x0_mean_abs_diff={x0_abs_diff.mean().item():.8e} "
                                        f"x0_max_abs_diff={x0_abs_diff.max().item():.8e}"
                                    )
                                del x0_from_sampling_pred, pred_abs_diff, x0_abs_diff
                                # 对于 Wan 的 flow_prediction，Diffusers scheduler 内部基本就是：x0_pred = sample - sigma * model_output
                                # 一句话：这个公式来自 flow matching：模型输出的是从当前 noisy latent 到 clean latent 的 velocity/flow，
                                # 所以 clean estimate 是当前 latent 减去 sigma 加权的 predicted flow。
                                # 为什么是这个公式？Flow matching 常见设定是：x_t = (1 - sigma_t) * x0 + sigma_t * noise
                                # 也就是说当前 sample 是：clean data x0和 noise之间的线性插值（Flow matching 通常把 noisy/intermediate sample 写成线性插值）
                                # 当：sigma = 0 -> x_t = x0; sigma = 1 -> x_t = noise
                                # 模型学习一个 velocity / flow，表示从当前点往 clean data 方向应该怎么走。
                                # 如果模型输出：v_theta(x_t, t) ≈ (x_t - x0) / sigma_t
                                # 那 rearrange 一下：x_t - x0 ≈ sigma_t * v_theta
                                # 所以：x0 ≈ x_t - sigma_t * v_theta
                                # 这就是：x0_pred = x_t - sigma_t * model_output
                                # 和 DDPM 不同
                                # DDPM epsilon-pred 里通常是：x_t = sqrt(alpha_t) x0 + sqrt(1-alpha_t) eps
                                # 所以：x0_pred = (x_t - sqrt(1-alpha_t) eps_pred) / sqrt(alpha_t)
                                # Flow matching 是线性路径，所以是：x0_pred = x_t - sigma_t * v_pred更直接。

                                # 多维输出对多维输入的导数”就叫 Jacobian。
                                # 在我们的 x0 里它为什么出现？
                                # flow matching 里：
                                # x0_pred = latents - sigma * noise_pred_g
                                # 但：
                                # noise_pred_g = transformer(latents)
                                # 所以严格写：
                                # x0_pred = latents - sigma * transformer(latents)
                                # 如果对 latents 求导：
                                # d x0_pred / d latents
                                # = I - sigma * d transformer(latents) / d latents
                                # 这里：
                                # d transformer(latents) / d latents
                                # 就是 transformer Jacobian。

                                # transformer Jacobian 指的是：
                                # transformer 输出对 transformer 输入 latent 的导数矩阵
                                # 在这里，transformer 是一个函数：
                                # noise_pred = v_theta(latents, t, prompt)
                                # 也就是：
                                # noise_pred_g = transformer(latents)
                                # 那么 transformer Jacobian 就是：
                                # ∂ noise_pred_g / ∂ latents
                                # transformer Jacobian 就是模型预测 noise_pred 对输入 latents 的敏感度
                                # detach noise_pred_g 等于忽略这个敏感度，只用近似梯度来更新 latents。
                                # d x0_pred / d latents = I - sigma * J_transformer 其中：J_transformer = d transformer(latents) / d latents
                                # 如果 detach noise_pred_g，就等于把：transformer(latents)当成常量，所以：
                                # J_transformer = 0， 于是：d x0_pred / d latents = I， 也就是梯度只通过 latents 这一项传。

                                # 为什么忽略它？
                                # 因为 transformer Jacobian 巨大且昂贵。
                                # 如果不忽略，反传会经过整个 transformer：
                                # loss -> VAE -> x0_pred -> transformer -> latents
                                # 需要保存 transformer activations，显存非常大。
                                # 所以我们采用近似：
                                # 把 transformer 输出当成当前点的固定方向，
                                # 只通过 latents 本身反传。

                                # 但 detach 有什么副作用？
                                # 有。它让 guidance 变成近似梯度。
                                # 完整依赖是：
                                # x0_pred = f(latents, transformer(latents))
                                # detach 后我们当成：
                                # x0_pred ≈ latents - constant
                                # 所以梯度方向可能不完美。
                                # 这也是为什么：
                                # guidance 有时弱
                                # 多次 repeat 如果不重新算 noise_pred 会错
                                # 我们修 repeat 重算 noise_pred_g，就是为了缓解这个近似：
                                # 每次 latent 更新后，重新 forward transformer 得到新的 stopgrad prediction
                                # 这相当于：
                                # 不用 transformer backward，
                                # 但每次更新后重新线性化当前点。

                                # sigma 大 = noisy 是 scheduler/noise schedule 的设计。
                                # 在 flow matching 里尤其直观，因为 x_t = (1-sigma)x0 + sigma noise。

                                # TODO+IMPORTANT： 解决和思考中间 step 的 x0_pred误差问题
                                # visualize 早中晚期的decoded x0_pred，观察噪声和模糊程度/语义和几何是否valid，决定在哪些time step开始guidance更有效。
                                # guidance timing 是 trade-off： 中间 step 的 x0_pred 是一个 noisy estimate of final clean sample；它越早越不准，越晚越准，但越晚越难改变生成轨迹。

                            # dummy latent_l2 loss 分支
                            # 一个最简单的 smoke-test loss，用来验证 gradient path，不代表真实 GeCo
                            if loss_fn == "latent_l2":
                                loss = x0_pred.float().square().mean()
                            # RGB GeCo, Frame Guidance, or their controlled joint loss.
                            # The x0 prediction and decoder below are the existing RGB-GeCo path;
                            # Frame Guidance only adds an anchor MSE term on the same decoded frames.
                            elif loss_fn in {"residual_motion", "frame", "frame_residual_motion"}:
                                if uses_residual_motion and (
                                    additional_inputs is None
                                    or not callable(additional_inputs.get("residual_motion_metric", None))
                                ):
                                    raise ValueError(
                                        "Pass additional_inputs={'residual_motion_metric': callable(frames_01)->scalar_score}"
                                    )
                                if uses_frame_guidance and frame_guidance_targets is None:
                                    raise RuntimeError("Frame Guidance targets were not prepared.")
                                # 选要 decode/评估的 frames
                                # 意思：决定哪些视频帧参与 GeCo loss。
                                # 为什么：不能 decode 全部视频，太占显存；而且 guidance 只需要抽样几个关键帧估计几何 residual。
                                if uses_frame_guidance:
                                    # Official Frame Guidance treats frame zero as an immutable I2V
                                    # condition.  Decode it only when the joint GeCo metric needs it.
                                    fixed_frames_ = (
                                        frame_guidance_fixed_frames
                                        if uses_residual_motion
                                        else [index for index in frame_guidance_fixed_frames if index != 0]
                                    )
                                else:
                                    fixed_frames_ = fixed_frames if fixed_frames is not None else [0, num_frames - 1]
                                if isinstance(fixed_frames_, int):
                                    fixed_frames_ = [fixed_frames_]
                                fixed_frames_ = [int(x) for x in fixed_frames_]
                                # 意思：VAE decode 时是否降低空间分辨率。
                                # 为什么：full-res differentiable VAE decode 会 OOM。比如 0.5 就是在 latent spatial size 上减半，显存大概降很多。
                                # 代价：scale 太低，UFM/VGGT 信号会变差。
                                # 如果 additional_inputs 里有 decode_spatial_scale，就取它；如果没有，就默认用 1.0。
                                # TODO+IMPORTANT：原 GeCo CogVideoX: decode selected temporal chunks，但 spatial resolution 不降。也就是 decode 到原 pipeline 视频分辨率。
                                decode_spatial_scale = float(additional_inputs.get("decode_spatial_scale", 1.0))
                                if decode_spatial_scale != 1.0:
                                    raise ValueError('Full Wan GeCo guidance requires decode_spatial_scale=1.0')
                                cross_device_grad_via_cpu = bool(
                                    additional_inputs.get("cross_device_grad_via_cpu", False)
                                )

                                # Decode only the temporal chunks needed by fixed_frames.
                                # Full-video differentiable VAE decode is too expensive at 480x832.
                                # 意思：对于 I2V，把 condition 部分保持为原始 condition，不让 x0_pred 覆盖它。
                                # 为什么：第一帧/condition frame 是给定的，不应该被 guidance 改坏。我们只希望指导生成部分。
                                x0_for_decode = x0_pred
                                if self.config.expand_timesteps:
                                    # 它的效果是：
                                    # mask=0 的区域保留 condition
                                    # mask=1 的区域使用 predicted/generated x0
                                    x0_for_decode = (1 - first_frame_mask.float()) * condition.float() + first_frame_mask.float() * x0_pred.float()

                                # Keep the full x0 tensor in its existing dtype, but only cast/normalize
                                # the small temporal chunk that will actually be decoded. Holding a full
                                # normalized z_all copy is enough to OOM at 320x576 guidance.
                                # 意思：拿到 latent 时间长度和 VAE 时间压缩比例。
                                # 为什么：Wan VAE 不是一帧 latent 对一帧视频。通常约 4 帧视频对应 1 个 temporal latent token。所以要把目标 video frame index 映射到 latent chunk。
                                # x0_for_decode 的形状是：[B, C, T_lat, H_lat, W_lat]， shape[2] 就是 latent 的时间长度。
                                # T_lat 是当前 x0 latent 里有多少个 temporal latent tokens；
                                # temporal_scale 是 VAE 时间压缩比例，用来把 RGB frame index 映射到 latent time index。
                                T_lat = x0_for_decode.shape[2]
                                temporal_scale = int(getattr(self, "vae_scale_factor_temporal", 4))

                                vae_dtype = self.vae.dtype
                                # 意思：把 diffusion latent 转成 VAE decode 需要的 latent normalization。
                                # 为什么：Diffusers Wan 最后正式 decode 前也做这一步。如果 guidance decode 不做，VAE 输入分布错，decoded frames 就不可信。
                                vae_device = self._get_geco_vae_device()
                                latents_mean = (
                                    torch.tensor(self.vae.config.latents_mean)
                                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                                    .to(vae_device, vae_dtype)
                                )
                                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                                    1, self.vae.config.z_dim, 1, 1, 1
                                ).to(vae_device, vae_dtype)

                                rm_frames = []
                                decoded_frame_indices = []
                                # 只 decode fixed frame 附近的小 temporal chunk，并取出对应 RGB frame。
                                # 意思：遍历每个要算 loss 的帧，并确保 index 合法。
                                # 这两行遍历 guidance 使用的 frame indices，并把每个 index 限制在合法视频帧范围内。
                                for fidx in fixed_frames_:
                                    fidx = int(max(0, min(fidx, num_frames - 1)))
                                    decoded_frame_indices.append(fidx)

                                    if uses_frame_guidance:
                                        # Use the official predecessor/target-pair convention.  In
                                        # particular, f=60 selects local slot 4 of latent pair 14:16,
                                        # not slot 7 of a three-token slice (which is a future token).
                                        chunk_start, chunk_end, rel = _wan_causal_frame_decode_plan(
                                            fidx,
                                            num_frames=num_frames,
                                            latent_frames=T_lat,
                                            temporal_scale=temporal_scale,
                                        )
                                    else:
                                        # Preserve the existing RGB-GeCo temporal slice behavior for
                                        # pure residual_motion runs.
                                        if fidx == 0:
                                            center_lat = 0
                                        else:
                                            center_lat = min(T_lat - 1, (fidx - 1) // temporal_scale + 1)
                                        chunk_start = max(0, center_lat - 1)
                                        chunk_end = min(T_lat, center_lat + 2)

                                    # 从完整的 x0_for_decode latent 里切出刚才选好的 temporal chunk，并准备送进 VAE。
                                    # 先看 x0_for_decode 的形状
                                    # 它是 video latent，形状大概是：
                                    # [B, C, T_lat, H_lat, W_lat]
                                    # 分别是：
                                    # B = batch
                                    # C = latent channels
                                    # T_lat = latent 时间长度
                                    # H_lat = latent 高度
                                    # W_lat = latent 宽度

                                    # [:, :, chunk_start:chunk_end]
                                    # 这是 tensor slicing。
                                    # x0_for_decode[:, :, chunk_start:chunk_end]
                                    # 意思：
                                    # 第 1 维 B：全部保留
                                    # 第 2 维 C：全部保留
                                    # 第 3 维 T：只取 chunk_start 到 chunk_end-1
                                    # 后面的 H/W：没写，默认全部保留
                                    # 等价完整写法是：
                                    # x0_for_decode[:, :, chunk_start:chunk_end, :, :]
                                    # 所以结果形状是：
                                    # [B, C, chunk_T, H_lat, W_lat]
                                    # 其中：
                                    # chunk_T = chunk_end - chunk_start
                                    # 通常是 2 或 3。

                                    # .contiguous()
                                    # 切片以后 tensor 可能不是 contiguous memory layout。
                                    # VAE decode / interpolate 等操作通常更喜欢连续内存。
                                    # .contiguous()
                                    # 会创建一个内存连续的 tensor，避免后续操作报错或变慢。
                                    z_chunk = x0_for_decode[:, :, chunk_start:chunk_end].contiguous()
                                    z_chunk = _geco_move_tensor(
                                        z_chunk,
                                        vae_device,
                                        vae_dtype,
                                        via_cpu_for_grad=cross_device_grad_via_cpu,
                                    )

                                    # 意思：可选地缩小 latent 的 H/W。
                                    # 为什么：进一步省显存。这个操作仍然可导，所以 gradient 可以从 decoded frames 回到 z_chunk，再回到 latents
                                    # 如果：
                                    # decode_spatial_scale = 1.0
                                    # 就 full latent resolution decode。
                                    # 如果：
                                    # decode_spatial_scale = 0.5
                                    # 就把 latent spatial H/W 缩小一半再 decode。
                                    # 如果：
                                    # decode_spatial_scale = 0.25
                                    # 就缩小到四分之一。
                                    # 代价是什么？
                                    # 低分辨率 decode 会让 GeCo loss 变粗糙：
                                    # UFM flow 不准
                                    # VGGT depth/pose 不准
                                    # small object/detail 丢失
                                    # guidance signal 变弱或变噪声
                                    # 所以它是 trade-off：
                                    # scale 越大：信号更准，但更容易 OOM
                                    # scale 越小：更省显存，但 metric 可能不可靠
                                    # IMPORTANT: GeCo 原来的 pipeline 是不是 full-res differentiable VAE decode？
                                    # 是的，spatial 上是 full-res differentiable VAE decode。
                                    # decode_spatial_scale 是工程折中，不是 GeCo 原始方法。
                                    # 它帮助跑得动，但也可能削弱 guidance 正确性。
                                    if decode_spatial_scale != 1.0:
                                        H_lat, W_lat = z_chunk.shape[-2:]
                                        H_new = max(1, int(round(H_lat * decode_spatial_scale)))
                                        W_new = max(1, int(round(W_lat * decode_spatial_scale)))
                                        z_chunk = torch.nn.functional.interpolate(
                                            z_chunk,
                                            size=(z_chunk.shape[2], H_new, W_new),
                                            mode="trilinear",
                                            align_corners=False,
                                        )

                                    # 意思：VAE decode 前做 latent un-normalization。
                                    # 为什么：和正式 pipeline 输出 decode 保持一致。
                                    z_chunk = z_chunk / latents_std + latents_mean

                                    # 意思：把 x0 latent chunk decode 成 RGB video chunk。
                                    # 为什么必须在 torch.enable_grad() 里：
                                    # 我们需要 loss -> decoded pixels -> z_chunk -> x0_pred -> latents 的梯度路径。
                                    # IMPORTANT: 原 GeCo 代码也是只 decode selected frames 附近的 temporal chunk
                                    tile_latent_height = (
                                        self.vae.tile_sample_min_height // self.vae.spatial_compression_ratio
                                    )
                                    tile_latent_width = (
                                        self.vae.tile_sample_min_width // self.vae.spatial_compression_ratio
                                    )
                                    use_per_tile_checkpoint = self.vae.use_tiling and (
                                        z_chunk.shape[-2] > tile_latent_height
                                        or z_chunk.shape[-1] > tile_latent_width
                                    )
                                    if use_per_tile_checkpoint:
                                        decoded_chunk = _geco_tiled_decode_with_per_tile_checkpoint(self.vae, z_chunk)
                                    else:
                                        # Small non-tiled chunks keep the existing whole-VAE checkpoint.
                                        def _decode_chunk_for_checkpoint(z):
                                            return self.vae.decode(z, return_dict=False)[0]

                                        decoded_chunk = checkpoint(
                                            _decode_chunk_for_checkpoint, z_chunk, use_reentrant=False
                                        )

                                    if debug_guidance_consistency:
                                        # Debug only: verify the memory-efficient per-tile checkpointed
                                        # decode has the same pixels as Diffusers' native tiled decode.
                                        # The native result is detached and never affects the loss/gradient.
                                        with torch.no_grad():
                                            native_decoded_chunk = self.vae.decode(
                                                z_chunk.detach(), return_dict=False
                                            )[0]
                                            decode_abs_diff = (
                                                decoded_chunk.detach().float() - native_decoded_chunk.float()
                                            ).abs()
                                            print(
                                                f"wan_guidance_vae_compare({i}/{rep},frame={fidx}): "
                                                f"per_tile={use_per_tile_checkpoint} "
                                                f"mean_abs_diff={decode_abs_diff.mean().item():.8e} "
                                                f"max_abs_diff={decode_abs_diff.max().item():.8e}"
                                            )
                                        del native_decoded_chunk, decode_abs_diff

                                    # 意思：把 VAE 输出从 [B,C,T,H,W] 转成 [B,T,H,W,C]，并从 [-1,1] 映射到 [0,1]。
                                    # 为什么：make_motion_metric 期望输入是 [B,F,H,W,3]，值域 [0,1]。
                                    #  .clamp(0, 1)
                                    # 把值限制在合法图像范围：
                                    # 小于 0 的变成 0
                                    # 大于 1 的变成 1
                                    # 因为 VAE 输出可能略微超出 [-1,1]，比如：
                                    # 1.03 或 -1.05
                                    # 映射后可能超出 [0,1]，所以 clamp。
                                    frames_chunk = ((decoded_chunk.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)

                                    if uses_frame_guidance:
                                        if rel >= frames_chunk.shape[1]:
                                            raise RuntimeError(
                                                f"Wan Frame Guidance slice for frame {fidx} produced "
                                                f"{frames_chunk.shape[1]} frames, but needs local frame {rel}."
                                            )
                                    else:
                                        # Keep the existing pure RGB-GeCo local-frame calculation.
                                        if chunk_start == 0:
                                            chunk_first_frame = 0
                                        else:
                                            chunk_first_frame = 1 + (chunk_start - 1) * temporal_scale
                                        rel = fidx - chunk_first_frame
                                        rel = int(max(0, min(rel, frames_chunk.shape[1] - 1)))
                                    # 这里用 slice：
                                    # rel:rel+1
                                    # 而不是：
                                    # rel
                                    # 是为了保留时间维度。
                                    # 如果用：
                                    # frames_chunk[:, rel]
                                    # 形状会变成：
                                    # [B, H, W, C]
                                    # 时间维消失。
                                    # 用：
                                    # frames_chunk[:, rel:rel + 1]
                                    # 形状是：
                                    # [B, 1, H, W, C]
                                    # 这样后面多个 frame 可以：
                                    # torch.cat(rm_frames, dim=1)
                                    # 拼成：
                                    # [B, F_selected, H, W, C]
                                    rm_frames.append(frames_chunk[:, rel:rel + 1])

                                    # 手动删除临时变量，帮助释放显存引用
                                    del z_chunk, decoded_chunk, frames_chunk

                                # 它是 selected decoded frames，形状：
                                # [B, F_selected, H, W, C]
                                frames_01 = torch.cat(rm_frames, dim=1)

                                if debug_guidance_consistency:
                                    with torch.no_grad():
                                        print(
                                            f"wan_guidance_frames({i}/{rep}): "
                                            f"shape={tuple(frames_01.shape)} "
                                            f"min={frames_01.detach().amin().item():.8f} "
                                            f"max={frames_01.detach().amax().item():.8f} "
                                            f"mean={frames_01.detach().mean().item():.8f}"
                                        )

                                frame_loss = None
                                if uses_frame_guidance:
                                    frame_loss_terms = []
                                    for local_index, frame_index in enumerate(decoded_frame_indices):
                                        # The condition is held fixed by Wan.  It is retained in the
                                        # manifest for provenance, but cannot provide a latent update.
                                        if frame_index == 0:
                                            continue
                                        target_11 = frame_guidance_targets[frame_index]
                                        prediction_11 = frames_01[:, local_index] * 2.0 - 1.0
                                        if prediction_11.shape != target_11.shape:
                                            raise RuntimeError(
                                                "Frame Guidance target/prediction shape mismatch for "
                                                f"frame {frame_index}: predicted={tuple(prediction_11.shape)}, "
                                                f"target={tuple(target_11.shape)}."
                                            )
                                        frame_loss_terms.append(
                                            torch.nn.functional.mse_loss(prediction_11.float(), target_11.float())
                                        )
                                    if not frame_loss_terms:
                                        raise RuntimeError(
                                            "Frame Guidance has no nonzero anchor frame available for its MSE loss."
                                        )
                                    frame_loss = torch.stack(frame_loss_terms).mean()

                                geco_loss = None
                                geco_score = None
                                if uses_residual_motion:
                                    # Keep the audited RGB GeCo loss/sign unchanged: the metric returns
                                    # a score whose negative is the residual minimization objective.
                                    score = additional_inputs["residual_motion_metric"](frames_01)
                                    if not score.requires_grad:
                                        raise RuntimeError("Residual motion metric returned a detached score.")
                                    geco_score = score
                                    geco_loss = -score

                                if frame_loss is None and geco_loss is None:
                                    raise RuntimeError(f"No guidance loss was constructed for loss_fn={loss_fn!r}.")
                                if frame_loss is None:
                                    loss = geco_loss
                                elif geco_loss is None:
                                    loss = frame_loss_weight * frame_loss
                                else:
                                    loss = (
                                        frame_loss_weight * frame_loss
                                        + geco_loss_weight * geco_loss
                                    )

                                if uses_frame_guidance:
                                    diagnostics = additional_inputs.get("guidance_diagnostics")
                                    if diagnostics is not None:
                                        if not isinstance(diagnostics, list):
                                            raise ValueError(
                                                "additional_inputs['guidance_diagnostics'] must be a list when provided."
                                            )
                                        diagnostics.append(
                                            {
                                                "step_index": int(i),
                                                "repeat_index": int(rep),
                                                "loss_fn": loss_fn,
                                                "decoded_frame_indices": list(decoded_frame_indices),
                                                "frame_loss_raw": float(frame_loss.detach().item()),
                                                "geco_score_raw": (
                                                    None
                                                    if geco_score is None
                                                    else float(geco_score.detach().item())
                                                ),
                                                "geco_loss_raw": (
                                                    None
                                                    if geco_loss is None
                                                    else float(geco_loss.detach().item())
                                                ),
                                                "combined_loss": float(loss.detach().item()),
                                                "frame_loss_weight": float(frame_loss_weight),
                                                "geco_loss_weight": (
                                                    None
                                                    if geco_loss is None
                                                    else float(geco_loss_weight)
                                                ),
                                                "weights_normalized": False,
                                            }
                                        )
                                    geco_value = "n/a" if geco_loss is None else f"{geco_loss.detach().item():.6f}"
                                    print(
                                        f"wan_frame_guidance_loss({i}/{rep}): "
                                        f"frame={frame_loss.detach().item():.6f} "
                                        f"geco={geco_value} total={loss.detach().item():.6f}"
                                    )

                                del frames_01, rm_frames, decoded_frame_indices
                            else:
                                raise RuntimeError(f"Unexpected loss_fn: {loss_fn}")

                            # 计算 guidance loss 对当前 latents 的梯度

                            # retain_graph=False
                            # 意思是：
                            # 算完这次梯度后，不保留计算图
                            # 为什么？
                            # 因为这次 guidance update 只用一次这个 graph。
                            # 不保留可以省显存。
                            # 如果后面还想对同一个 graph 再 backward 一次，才需要 retain_graph=True。这里不需要。

                            # create_graph=False
                            # 意思是：
                            # 不要为梯度本身再建立计算图
                            # 也就是不需要二阶梯度。
                            # 我们只需要一阶梯度：
                            # ∂loss / ∂latents
                            # 不需要：
                            # ∂²loss / ∂latents²
                            # 所以设 False 省显存。

                            # [0]
                            # torch.autograd.grad 返回的是 tuple。
                            # 因为你可以一次对多个 tensor 求梯度：
                            # torch.autograd.grad(loss, [latents, other_tensor])
                            # 它会返回：
                            # (grad_latents, grad_other)
                            # 这里我们只传了一个 latents，所以返回：
                            # (grad_latents,)
                            # 取 [0] 就是拿出这个 tensor。
                            grad = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]

                        # 这两行把梯度转成 float32 并计算 L2 norm，用于后面做归一化梯度更新，避免 update 大小被 GeCo loss 的绝对尺度直接支配。
                        # 第一行：grad_f = grad.float()
                        # 把梯度转成 float32。
                        # 为什么？
                        # grad 可能是：
                        # bf16 / fp16 / fp32
                        # 如果用低精度算 norm，可能不够稳定。
                        # 所以先转成：
                        # float32
                        # 再算统计量。
                        # 这不会改变原始 grad 本身，只是创建一个用于计算 norm 的 float32 版本。
                        # 第二行：grad_f.norm()
                        # grad_f.norm()
                        # 默认是 L2 norm。
                        # 数学上：
                        # ||grad||_2 = sqrt(sum_i grad_i^2)
                        # 也就是把整个 gradient tensor 展平成一大串数，然后算欧几里得长度。整个梯度 tensor 的 L2 norm。
                        # + 1e-8
                        # grad_norm = grad_f.norm() + 1e-8
                        # 加一个很小的数，防止后面除以 0。
                        # 后面有：
                        # update = guidance_lr[i] * grad / grad_norm
                        # 如果 grad_norm = 0，就会除零。
                        # 所以加：
                        # 1e-8
                        # 保证数值安全。
                        # 为什么需要 grad_norm？
                        # 因为后面不是直接：
                        # latents = latents - lr * grad
                        # 而是：
                        # latents = latents - lr * grad / grad_norm
                        # 这叫 normalized gradient update。
                        # 它让更新方向保持是梯度方向，但更新大小主要由 guidance_lr 控制，而不是由 loss 的数值尺度控制。
                        # 举例：
                        # grad_norm 很大：
                        # grad / grad_norm 后仍然是单位方向，不会更新爆炸
                        # grad_norm 很小：
                        # grad / grad_norm 后方向仍然可用，不会因为梯度太小完全没动
                        # FIXME当然这也有风险，所以后面又加了：
                        # max_relative_delta
                        # 限制实际 update 幅度。
                        # 为什么加 max_relative_delta？一句话：gradient normalization 可以让不同 loss scale 下 update 大小稳定，
                        # 但也可能把很弱或很噪声的梯度强行放大；max_relative_delta 是安全阀，限制每次 latent update 相对 latent 本身的最大幅度，避免视频崩坏。
                        grad_f = grad.float()
                        grad_norm = grad_f.norm() + 1e-8

                        grad_mean_abs = grad_f.abs().mean().item()
                        grad_max_abs = grad_f.abs().max().item()
                        latent_mean_abs = latents.detach().float().abs().mean().item()
                        latents_before_guidance = latents
                        # 计算 latent guidance 的更新量。
                        # grad / grad_norm
                        # 是把梯度归一化成一个“单位方向”。
                        # 意思：
                        # 保留梯度方向
                        # 去掉梯度绝对大小
                        # 所以它表示：
                        # 朝 loss 增大的方向的单位向量
                        # 为什么不是直接用 grad？
                        # 如果直接：
                        # update = guidance_lr[i] * grad
                        # 那 update 大小会强烈依赖 loss scale。
                        # 比如：
                        # GeCo loss 数值大 -> grad 大 -> update 爆炸
                        # GeCo loss 数值小 -> grad 小 -> 几乎没更新
                        # 而不同场景、不同 scale、不同 frame 数都会改变 loss/grad 大小。
                        # 归一化以后：
                        # update 大小主要由 guidance_lr 控制
                        # 更容易调。
                        update = guidance_lr[i] * grad / grad_norm

                        # 如果 guidance update 的平均幅度超过 latent 本身平均幅度的一定比例，就按比例缩小 update，避免 latent 被一次推太远导致视频崩坏。
                        # IMPORTANT: GeCo 原本没有 max_relative_delta 这种 cap。它只做了 global grad norm normalization：update = guidance_lr * grad / ||grad||
                        # 但没有再限制：update 的 mean abs 不能超过 latent_mean_abs 的某个比例
                        # max_relative_delta
                        # 是我们后面给 Wan 加的安全阀。原因是 Wan 里有些 step 的 gradient 特别大，虽然做了 norm normalization，但局部或者平均 update 仍可能把 latent 推得太远，导致视频崩坏。
                        # 所以区别是：
                        # GeCo 原本:
                        # 只 normalize gradient norm。
                        # 我们的 Wan:
                        # normalize gradient norm
                        # + optional max_relative_delta cap。
                        # 这个 cap 是工程稳定性改动，不是原论文方法。
                        # 这个还是要时刻关注对guidance是否会有影响，比如是否会对guidance 的效果显著限制
                        max_relative_delta = 0.0
                        if additional_inputs is not None:
                            max_relative_delta = float(additional_inputs.get("max_relative_delta", 0.0) or 0.0)
                        if max_relative_delta > 0.0:
                            update_delta = update.float().abs().mean()
                            max_delta = max_relative_delta * (latent_mean_abs + 1e-8)
                            if update_delta.item() > max_delta:
                                update = update * (max_delta / (update_delta + 1e-8))

                        #  guidance loss 的负梯度更新当前 noisy latent，
                        # 然后切断这次 guidance update 的计算图，
                        # 让新 latent 作为后续 denoising 的状态继续运行。

                        # 数学形式
                        # update = η * ∇L / ||∇L||
                        # 其中：
                        # η = guidance_lr[i]
                        # L = guidance loss
                        # 然后：
                        # x_t ← x_t - update
                        latents = (latents - update).detach()
                        # latent_delta = 本次 update 平均每个 latent 元素改了多少
                        latent_delta = (latents - latents_before_guidance).float().abs().mean().item()
                        # 把 update 幅度换算成相对于 latent 本身大小的百分比。
                        relative_delta = latent_delta / (latent_mean_abs + 1e-8) * 100.0
                        # 把 gradient 平均绝对值也换算成相对于 latent 本身大小的百分比。
                        relative_grad = grad_mean_abs / (latent_mean_abs + 1e-8) * 100.0
                        print(
                            f"wan_guidance_loss({i}/{rep}): {loss.item():.6f} "
                            f"latent_mean_abs={latent_mean_abs:.6f} grad_norm={float(grad_norm):.6f} "
                            f"grad_mean_abs={grad_mean_abs:.10f} grad_max_abs={grad_max_abs:.8f} "
                            f"latent_delta={latent_delta:.8f} relative_delta={relative_delta:.6f}% relative_grad={relative_grad:.8f}%",
                            flush=True,
                        )

                        del grad, noise_pred_g, x0_pred, loss

                    # Recompute noise_pred after latent update, so scheduler.step uses the updated latent.
                    if self.config.expand_timesteps:
                        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                        latent_model_input = latent_model_input.to(transformer_dtype)
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                        timestep = t.expand(latents.shape[0])

                    with current_model.cache_context("cond"):
                        # IMPORTANT: 用更新后的 latents 重新预测 model output。
                        # 为什么：这是修复 stale noise_pred 的关键。如果不做这步，就会变成：
                        # updated latents + old noise_pred
                        # scheduler step 就不一致。
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # 正常 denoising 进入下一步。
                # 这里使用的是：
                # guidance 更新后的 latents
                # guidance 后重算的 noise_pred
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

        if self.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if not output_type == "latent":
            vae_device = self._get_geco_vae_device()
            # Final non-gradient decode also crosses transformer -> VAE devices.
            latents = _geco_move_tensor(latents, vae_device, self.vae.dtype, via_cpu=True)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
