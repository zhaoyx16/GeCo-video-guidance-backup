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

import html
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, List

import PIL
import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import _get_qkv_projections
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from draft_geometry_map import frame_to_latent_index, load_draft_geometry_map


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

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
        # 给 __call__ 新增 attention averaging 参数
        # 作用：
        # attn_avg_alpha=0.0 表示完全关闭；
        # attn_avg_layers=[10,15,20] 表示在哪些 transformer block 的 self-attention 输出后做 feature averaging。对应文档在 606-609 行。
        attn_avg_alpha: float = 0.0,
        attn_avg_layers: List[int] = None,
        attn_avg_mode: str = "global",
        attn_avg_start: int = 0,
        attn_avg_end: int | None = None,
        attn_avg_temporal_radius: int = 1,
        attn_avg_match_radius: int = 1,
        attn_avg_match_confidence: float = 0.0,
        attn_avg_match_mutual: bool = False,
        attn_avg_descriptor_dim: int = 64,
        attn_avg_coarse_factor: int = 2,
        attn_avg_memory_lookback: int = 3,
        attn_avg_cond_only: bool = True,
        attn_avg_preserve_first_frame: bool = True,
        attn_avg_debug: bool = False,
        # Two-pass geometry transport. The map is built offline from a decoded
        # baseline draft; this sampling pass only loads token correspondences.
        draft_geometry_map_path: str | None = None,
        geometry_transport_alpha: float = 0.0,
        geometry_transport_layers: List[int] = None,
        geometry_transport_mode: str = "shadow",
        geometry_transport_start: int | None = None,
        geometry_transport_end: int | None = None,
        geometry_transport_schedule: str = "constant",
        geometry_transport_min_confidence: float = 0.0,
        geometry_transport_gate_mode: str = "confidence",
        geometry_transport_null_logit: float = 0.0,
        geometry_transport_source_logit_bias: float = 0.0,
        geometry_transport_sparse_logit_boost: float = 0.0,
        geometry_transport_surface_depth_threshold: float = 0.0,
        geometry_transport_surface_graph_scales: List[int] = None,
        geometry_transport_surface_boundary_logit_suppress: float = 0.0,
        geometry_transport_max_relative_rms: float = 0.0,
        geometry_transport_preserve_qk_rms: bool = True,
        geometry_transport_consensus_threshold: float = -1.0,
        geometry_transport_value_lowpass_radius: int = 0,
        geometry_transport_cond_only: bool = True,
        geometry_transport_debug: bool = False,
        draft_feature_cache_out_path: str | None = None,
        draft_feature_cache_in_path: str | None = None,
        draft_feature_cache_kind: str = "block_output",
        draft_feature_cache_fingerprint: str | None = None,
        draft_vector_cache_out_path: str | None = None,
        draft_vector_cache_in_path: str | None = None,
        draft_vector_anchor_outside_geometry: bool = False,
        draft_vector_mask_spatial_dilation: int = 1,
        draft_vector_mask_temporal_dilation: int = 1,
        draft_vector_geometry_alpha: float = 0.0,
        draft_vector_geometry_gate_mode: str = "binary_support",
        draft_vector_geometry_debug: bool = False,
        draft_vector_anchor_debug: bool = False,
        draft_clean_latent_cache_out_path: str | None = None,
        draft_clean_latent_cache_in_path: str | None = None,
        draft_clean_latent_alpha: float = 0.0,
        draft_clean_latent_gate_mode: str = "confidence",
        draft_clean_latent_transport_mode: str = "patch_mean",
        draft_clean_latent_reference_mode: str = "source_to_current",
        draft_clean_latent_delta_quantile: float = 0.0,
        draft_clean_latent_lowpass_radius: int = 2,
        draft_clean_latent_debug: bool = False,
        draft_geometry_condition_alpha: float = 0.0,
        draft_geometry_condition_gate_mode: str = "confidence",
        draft_geometry_condition_spatial_dilation: int = 0,
        draft_geometry_condition_reference_mode: str = "full_target",
        draft_geometry_condition_lowpass_radius: int = 2,
        draft_geometry_condition_start_step: int = 0,
        draft_geometry_condition_ramp_end_step: int = 0,
        draft_geometry_condition_schedule: str = "constant",
        draft_geometry_condition_final_blend: bool = True,
        draft_geometry_condition_debug: bool = False,
        draft_clean_latent_sigma_scaled: bool = False,
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
            attn_avg_alpha (`float`, defaults to `0.0`):
                Mixing weight for attention feature averaging. `0.0` disables the manipulation.
            attn_avg_layers (`List[int]`, *optional*):
                Transformer block indices whose self-attention outputs are averaged across frames.
            attn_avg_mode (`str`, defaults to `"global"`):
                `global`, `local`, `anchor`, `match_prev`, `c2f_match_prev`, `query_match_prev`,
                `key_match_prev`, `kv_match_prev`, `value_residual_prev`, `c2f_value_residual_prev`,
                or `c2f_value_residual_anchor`.
                The correspondence modes use the most similar local token in the previous frame rather than
                assuming identical pixel coordinates. `query_match_prev` transports only queries,
                `key_match_prev` transports only keys, and `kv_match_prev` transports keys and values.
                `value_residual_prev` leaves Q/K and the full attention weights unchanged, then mixes a
                confidence-gated previous-frame value into the resulting attention context.
            attn_avg_start / attn_avg_end (`int`, *optional*):
                Inclusive sampling-step interval where the intervention is active.
            attn_avg_temporal_radius (`int`, defaults to `1`):
                Temporal half-window used by `local` averaging.
            attn_avg_match_radius (`int`, defaults to `1`):
                Spatial search radius used by `match_prev`.
            attn_avg_match_confidence (`float`, defaults to `0.0`):
                Cosine-similarity threshold that gates `match_prev` propagation.
            attn_avg_match_mutual (`bool`, defaults to `False`):
                Require the local correspondence to be reciprocal before transporting features.
            attn_avg_descriptor_dim (`int`, defaults to `64`):
                Number of deterministic pooled descriptor channels for local correspondence.
            attn_avg_coarse_factor (`int`, defaults to `2`):
                Spatial downsampling factor used by `c2f_match_prev` to obtain a global coarse correspondence
                before its local full-resolution refinement.
            attn_avg_cond_only (`bool`, defaults to `True`):
                Apply only to the conditional CFG branch, leaving the unconditional model prediction unchanged.
            attn_avg_preserve_first_frame (`bool`, defaults to `True`):
                Do not directly overwrite the first-frame conditioning tokens.
            attn_avg_debug (`bool`, defaults to `False`):
                Print one compact correspondence-confidence summary per active sampling step.

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

        # 在 latents 准备完成后，开始 sampling loop 前，决定要不要注册 attention feature averaging hook       
        # expand_timesteps=True 时，Wan2.2 会给不同 token 分配不同 timestep，所以需要 first_frame_mask。如果不是这个模式，就只返回 latents, condition
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

        attn_avg_hook_handles = []
        draft_feature_hook_handles = []
        attn_avg_state = {"step": -1, "branch": "cond", "printed": set()}
        geometry_transport_layers = [] if geometry_transport_layers is None else sorted(set(geometry_transport_layers))
        geometry_transport_state = {
            "ready": False,
            "source_time": None,
            "source_index": None,
            "confidence": None,
            "pair_stats": [],
            "metadata": {},
            "map_sha256": None,
            "surface_graph_neighbors": None,
            "surface_graph_confidence": None,
            "surface_graph_scale_weight": None,
            "surface_boundary_neighbors": None,
            "surface_boundary_confidence": None,
            "surface_relative_edges": None,
            "printed": set(),
        }
        draft_feature_cache_state = {
            "capture": draft_feature_cache_out_path is not None,
            "loaded": False,
            "kind": draft_feature_cache_kind,
            "anchor_token_times": [],
            "features": {},
            "metadata": {},
        }
        draft_vector_cache_state = {
            "capture": draft_vector_cache_out_path is not None,
            "loaded": False,
            "predictions": {},
            "metadata": {},
            "geometry_mask": None,
            "printed": set(),
            "geometry_printed": set(),
        }
        draft_clean_latent_state = {
            "loaded": False,
            "latents": None,
            "metadata": {},
            "printed": set(),
        }
        if not 0.0 <= attn_avg_alpha <= 1.0:
            raise ValueError("attn_avg_alpha must lie in [0, 1]")
        if attn_avg_mode not in {
            "global", "local", "anchor", "match_prev", "c2f_match_prev", "query_match_prev",
            "key_match_prev", "kv_match_prev", "value_residual_prev", "c2f_value_residual_prev",
            "c2f_value_residual_anchor", "c2f_value_residual_memory",
        }:
            raise ValueError(
                "attn_avg_mode must be one of: global, local, anchor, match_prev, c2f_match_prev, "
                "query_match_prev, key_match_prev, kv_match_prev, value_residual_prev, "
                "c2f_value_residual_prev, c2f_value_residual_anchor, c2f_value_residual_memory"
            )
        if attn_avg_temporal_radius < 1:
            raise ValueError("attn_avg_temporal_radius must be at least 1")
        if attn_avg_match_radius < 0:
            raise ValueError("attn_avg_match_radius must be non-negative")
        if not 0.0 <= attn_avg_match_confidence < 1.0:
            raise ValueError("attn_avg_match_confidence must lie in [0, 1)")
        if attn_avg_descriptor_dim < 1:
            raise ValueError("attn_avg_descriptor_dim must be positive")
        if attn_avg_coarse_factor < 1:
            raise ValueError("attn_avg_coarse_factor must be positive")
        if attn_avg_memory_lookback < 1:
            raise ValueError("attn_avg_memory_lookback must be positive")

        if not 0.0 <= geometry_transport_alpha <= 1.0:
            raise ValueError("geometry_transport_alpha must lie in [0, 1]")
        if geometry_transport_mode not in {
            "shadow",
            "value_residual",
            "kv_memory",
            "attn_output_memory",
            "geometry_attention_bias",
            "geometry_attention_output_bias",
            "geometry_sparse_logit_bias",
            "geometry_virtual_token_bias",
            "geometry_surface_graph_logit_bias",
            "geometry_signed_surface_graph_logit_bias",
            "geometry_relative_edge_output",
            "cached_virtual_token_bias",
            "query_transport",
            "qk_transport",
            "cached_block_transport",
            "cached_value_transport",
            "cached_value_residual",
            "cached_value_memory",
            "cached_value_attention_blend",
            "cached_draft_value_delta",
            "cached_draft_value_delta_consensus",
            "cached_draft_output_delta",
            "cached_draft_output_delta_centered",
        }:
            raise ValueError(
                "geometry_transport_mode must be shadow, value_residual, kv_memory, attn_output_memory, "
                "geometry_attention_bias, geometry_attention_output_bias, "
                "geometry_sparse_logit_bias, geometry_virtual_token_bias, "
                "geometry_surface_graph_logit_bias, geometry_signed_surface_graph_logit_bias, "
                "geometry_relative_edge_output, "
                "cached_virtual_token_bias, "
                "query_transport, qk_transport, cached_block_transport, cached_value_transport, "
                "cached_value_residual, cached_value_memory, cached_value_attention_blend, "
                "cached_draft_value_delta, cached_draft_value_delta_consensus, "
                "cached_draft_output_delta, "
                "cached_draft_output_delta_centered, "
                "or cached_virtual_token_bias"
            )
        if draft_feature_cache_kind not in {
            "block_output",
            "attention_value",
            "attention_kv",
            "attention_value_delta",
        }:
            raise ValueError(
                "draft_feature_cache_kind must be block_output, attention_value, "
                "attention_kv, or attention_value_delta"
            )
        if geometry_transport_schedule not in {"constant", "linear_decay", "cosine_decay"}:
            raise ValueError("geometry_transport_schedule must be constant, linear_decay, or cosine_decay")
        if not 0.0 <= geometry_transport_min_confidence <= 1.0:
            raise ValueError("geometry_transport_min_confidence must lie in [0, 1]")
        if geometry_transport_gate_mode not in {"confidence", "binary_support"}:
            raise ValueError("geometry_transport_gate_mode must be confidence or binary_support")
        if not math.isfinite(geometry_transport_source_logit_bias):
            raise ValueError("geometry_transport_source_logit_bias must be finite")
        if not 0.0 <= geometry_transport_sparse_logit_boost <= 20.0:
            raise ValueError("geometry_transport_sparse_logit_boost must lie in [0, 20]")
        if not 0.0 <= geometry_transport_surface_boundary_logit_suppress <= 4.0:
            raise ValueError(
                "geometry_transport_surface_boundary_logit_suppress must lie in [0, 4]"
            )
        if not 0.0 <= geometry_transport_max_relative_rms <= 1.0:
            raise ValueError("geometry_transport_max_relative_rms must lie in [0, 1]")
        if not -1.0 <= geometry_transport_consensus_threshold <= 1.0:
            raise ValueError("geometry_transport_consensus_threshold must lie in [-1, 1]")
        if (
            geometry_transport_mode == "cached_draft_value_delta_consensus"
            and not 0.0 <= geometry_transport_consensus_threshold < 1.0
        ):
            raise ValueError(
                "cached_draft_value_delta_consensus requires "
                "geometry_transport_consensus_threshold in [0, 1)"
            )
        if geometry_transport_value_lowpass_radius < 0:
            raise ValueError("geometry_transport_value_lowpass_radius must be non-negative")
        if geometry_transport_surface_depth_threshold < 0:
            raise ValueError(
                "geometry_transport_surface_depth_threshold must be non-negative"
            )
        if geometry_transport_surface_graph_scales is None:
            geometry_transport_surface_graph_scales = [1]
        geometry_transport_surface_graph_scales = [
            int(scale) for scale in geometry_transport_surface_graph_scales
        ]
        if (
            not geometry_transport_surface_graph_scales
            or any(scale <= 0 for scale in geometry_transport_surface_graph_scales)
            or len(set(geometry_transport_surface_graph_scales))
            != len(geometry_transport_surface_graph_scales)
        ):
            raise ValueError(
                "geometry_transport_surface_graph_scales must contain unique "
                "positive integers"
            )
        geometry_transport_surface_graph_scales = sorted(
            geometry_transport_surface_graph_scales
        )
        if (
            geometry_transport_surface_graph_scales != [1]
            and geometry_transport_mode != "geometry_relative_edge_output"
        ):
            raise ValueError(
                "Multiscale surface graphs are supported only by "
                "geometry_relative_edge_output"
            )
        if geometry_transport_layers and not draft_geometry_map_path:
            raise ValueError("draft_geometry_map_path is required when geometry transport layers are set")
        if geometry_transport_mode != "shadow" and not geometry_transport_layers:
            raise ValueError(
                f"{geometry_transport_mode} requires at least one geometry transport layer"
            )
        if geometry_transport_layers and attn_avg_alpha > 0.0:
            raise ValueError("Do not combine geometry transport with feature-match transport in one run")
        if draft_feature_cache_out_path and not geometry_transport_layers:
            raise ValueError("Draft feature capture requires at least one geometry transport layer")
        if draft_feature_cache_out_path and draft_feature_cache_in_path:
            raise ValueError("Use either draft_feature_cache_out_path or draft_feature_cache_in_path, not both")
        if (
            draft_feature_cache_out_path or draft_feature_cache_in_path
        ) and not draft_feature_cache_fingerprint:
            raise ValueError(
                "Draft feature caching requires a non-empty run fingerprint"
            )
        if draft_feature_cache_out_path and geometry_transport_mode != "shadow":
            raise ValueError("Draft feature capture must use geometry_transport_mode='shadow'")
        if draft_feature_cache_out_path and geometry_transport_alpha != 0.0:
            raise ValueError("Draft feature capture must use geometry_transport_alpha=0")
        if draft_feature_cache_out_path and (
            attn_avg_alpha > 0.0
            or draft_vector_geometry_alpha > 0.0
            or draft_clean_latent_alpha > 0.0
            or draft_geometry_condition_alpha > 0.0
        ):
            raise ValueError(
                "Draft feature capture must be a pure baseline pass with all "
                "other interventions disabled"
            )
        cached_transport_modes = {
            "cached_block_transport",
            "cached_value_transport",
            "cached_value_residual",
            "cached_value_memory",
            "cached_value_attention_blend",
            "cached_virtual_token_bias",
            "cached_draft_value_delta",
            "cached_draft_value_delta_consensus",
            "cached_draft_output_delta",
            "cached_draft_output_delta_centered",
        }
        if geometry_transport_mode in cached_transport_modes and not draft_feature_cache_in_path:
            raise ValueError(f"{geometry_transport_mode} requires draft_feature_cache_in_path")
        if (
            geometry_transport_mode == "cached_virtual_token_bias"
            and geometry_transport_alpha > 0.0
            and geometry_transport_sparse_logit_boost <= 0.0
        ):
            raise ValueError(
                "cached_virtual_token_bias requires a positive sparse logit boost "
                "when geometry_transport_alpha is positive"
            )
        if draft_feature_cache_in_path and geometry_transport_mode not in cached_transport_modes:
            raise ValueError(
                "draft_feature_cache_in_path is only valid with cached_block_transport, "
                "cached_value_transport, cached_value_residual, cached_value_memory, "
                "cached_value_attention_blend, cached_draft_value_delta, "
                "cached_draft_value_delta_consensus, "
                "cached_draft_output_delta, cached_draft_output_delta_centered, "
                "or cached_virtual_token_bias"
            )
        if (draft_feature_cache_out_path or draft_feature_cache_in_path) and not geometry_transport_cond_only:
            raise ValueError("Draft feature caching currently requires geometry_transport_cond_only=True")
        if draft_vector_cache_out_path and draft_vector_cache_in_path:
            raise ValueError("Use either draft_vector_cache_out_path or draft_vector_cache_in_path, not both")
        if draft_vector_cache_out_path and (
            attn_avg_alpha > 0.0
            or geometry_transport_alpha > 0.0
            or geometry_transport_mode != "shadow"
        ):
            raise ValueError(
                "Draft vector capture must be a baseline pass: disable feature averaging and "
                "use geometry_transport_mode='shadow' with geometry_transport_alpha=0"
            )
        if draft_vector_anchor_outside_geometry and not draft_vector_cache_in_path:
            raise ValueError(
                "draft_vector_anchor_outside_geometry requires draft_vector_cache_in_path"
            )
        if draft_vector_anchor_outside_geometry and not geometry_transport_layers:
            raise ValueError(
                "draft_vector_anchor_outside_geometry requires geometry_transport_layers "
                "so the offline geometry support can be loaded"
            )
        if not 0.0 <= draft_vector_geometry_alpha <= 1.0:
            raise ValueError("draft_vector_geometry_alpha must lie in [0, 1]")
        if draft_vector_geometry_gate_mode not in {"confidence", "binary_support"}:
            raise ValueError(
                "draft_vector_geometry_gate_mode must be confidence or binary_support"
            )
        if draft_vector_geometry_alpha > 0.0 and not draft_vector_cache_in_path:
            raise ValueError(
                "draft_vector_geometry_alpha > 0 requires draft_vector_cache_in_path"
            )
        if draft_vector_geometry_alpha > 0.0 and not geometry_transport_layers:
            raise ValueError(
                "draft_vector_geometry_alpha > 0 requires geometry_transport_layers "
                "so the offline geometry correspondences can be loaded"
            )
        if draft_vector_mask_spatial_dilation < 0:
            raise ValueError("draft_vector_mask_spatial_dilation must be non-negative")
        if draft_vector_mask_temporal_dilation < 0:
            raise ValueError("draft_vector_mask_temporal_dilation must be non-negative")
        if draft_clean_latent_cache_out_path and draft_clean_latent_cache_in_path:
            raise ValueError(
                "Use either draft_clean_latent_cache_out_path or "
                "draft_clean_latent_cache_in_path, not both"
            )
        if not 0.0 <= draft_clean_latent_alpha <= 1.0:
            raise ValueError("draft_clean_latent_alpha must lie in [0, 1]")
        if draft_clean_latent_gate_mode not in {"confidence", "binary_support"}:
            raise ValueError(
                "draft_clean_latent_gate_mode must be confidence or binary_support"
            )
        if draft_clean_latent_transport_mode not in {"patch_mean", "full_patch"}:
            raise ValueError(
                "draft_clean_latent_transport_mode must be patch_mean or full_patch"
            )
        if draft_clean_latent_reference_mode not in {
            "source_to_current",
            "draft_delta",
            "draft_delta_centered",
            "draft_delta_highpass",
            "draft_delta_lowpass",
        }:
            raise ValueError(
                "draft_clean_latent_reference_mode must be source_to_current, "
                "draft_delta, draft_delta_centered, draft_delta_highpass, "
                "or draft_delta_lowpass"
            )
        if not 0.0 <= draft_clean_latent_delta_quantile < 1.0:
            raise ValueError("draft_clean_latent_delta_quantile must lie in [0, 1)")
        if draft_clean_latent_lowpass_radius < 0:
            raise ValueError("draft_clean_latent_lowpass_radius must be non-negative")
        if (
            draft_clean_latent_reference_mode
            in {"draft_delta_highpass", "draft_delta_lowpass"}
            and draft_clean_latent_lowpass_radius == 0
        ):
            raise ValueError(
                "draft_delta_highpass and draft_delta_lowpass require "
                "draft_clean_latent_lowpass_radius > 0"
            )
        if not 0.0 <= draft_geometry_condition_alpha <= 1.0:
            raise ValueError("draft_geometry_condition_alpha must lie in [0, 1]")
        if draft_geometry_condition_gate_mode not in {
            "confidence",
            "target_p90",
            "binary_support",
        }:
            raise ValueError(
                "draft_geometry_condition_gate_mode must be confidence, "
                "target_p90, or binary_support"
            )
        if draft_geometry_condition_spatial_dilation < 0:
            raise ValueError(
                "draft_geometry_condition_spatial_dilation must be non-negative"
            )
        if draft_geometry_condition_reference_mode not in {
            "full_target",
            "lowpass_delta",
        }:
            raise ValueError(
                "draft_geometry_condition_reference_mode must be "
                "full_target or lowpass_delta"
            )
        if draft_geometry_condition_lowpass_radius < 0:
            raise ValueError(
                "draft_geometry_condition_lowpass_radius must be non-negative"
            )
        if (
            draft_geometry_condition_alpha > 0.0
            and draft_geometry_condition_reference_mode == "lowpass_delta"
            and draft_geometry_condition_lowpass_radius == 0
        ):
            raise ValueError(
                "lowpass_delta requires "
                "draft_geometry_condition_lowpass_radius > 0"
            )
        if draft_geometry_condition_schedule not in {
            "constant",
            "linear_ramp",
            "cosine_ramp",
        }:
            raise ValueError(
                "draft_geometry_condition_schedule must be constant, "
                "linear_ramp, or cosine_ramp"
            )
        if draft_geometry_condition_alpha > 0.0:
            if not (
                0
                <= draft_geometry_condition_start_step
                < num_inference_steps
            ):
                raise ValueError(
                    "draft_geometry_condition_start_step must lie in "
                    "[0, num_inference_steps)"
                )
            if draft_geometry_condition_schedule != "constant" and not (
                draft_geometry_condition_start_step
                < draft_geometry_condition_ramp_end_step
                < num_inference_steps
            ):
                raise ValueError(
                    "A geometry condition ramp requires "
                    "start_step < ramp_end_step < num_inference_steps"
                )
        if draft_geometry_condition_alpha > 0.0 and not self.config.expand_timesteps:
            raise ValueError(
                "Draft geometry conditioning requires Wan expand_timesteps=True"
            )
        if draft_geometry_condition_alpha > 0.0 and (
            draft_clean_latent_transport_mode != "full_patch"
        ):
            raise ValueError(
                "Draft geometry conditioning requires full_patch latent transport"
            )
        if draft_geometry_condition_alpha > 0.0 and draft_clean_latent_alpha > 0.0:
            raise ValueError(
                "Use either draft geometry conditioning or clean x0 transport, not both"
            )
        if draft_clean_latent_cache_out_path and (
            attn_avg_alpha > 0.0
            or geometry_transport_alpha > 0.0
            or geometry_transport_mode != "shadow"
            or draft_vector_geometry_alpha > 0.0
            or draft_clean_latent_alpha > 0.0
            or draft_geometry_condition_alpha > 0.0
        ):
            raise ValueError(
                "Clean latent capture must be a baseline pass with all guidance disabled"
            )
        if (
            draft_clean_latent_alpha > 0.0
            or draft_geometry_condition_alpha > 0.0
        ) and not draft_clean_latent_cache_in_path:
            raise ValueError(
                "Clean latent guidance requires draft_clean_latent_cache_in_path"
            )
        if (
            draft_clean_latent_alpha > 0.0
            or draft_geometry_condition_alpha > 0.0
        ) and not geometry_transport_layers:
            raise ValueError(
                "Clean latent guidance requires geometry_transport_layers "
                "so the offline geometry correspondences can be loaded"
            )

        if draft_clean_latent_cache_in_path:
            clean_latent_payload = torch.load(
                draft_clean_latent_cache_in_path,
                map_location="cpu",
                weights_only=False,
            )
            clean_latent_metadata = clean_latent_payload.get("metadata", {})
            if clean_latent_metadata.get("format_version") != 1:
                raise ValueError("Unsupported clean latent cache format")
            if clean_latent_metadata.get("num_inference_steps") != num_inference_steps:
                raise ValueError(
                    "Clean latent cache inference-step count does not match this Wan run"
                )
            if tuple(clean_latent_metadata.get("latent_shape", ())) != tuple(latents.shape):
                raise ValueError(
                    "Clean latent cache shape does not match this Wan run: "
                    f"cache={clean_latent_metadata.get('latent_shape')} "
                    f"run={list(latents.shape)}"
                )
            if float(clean_latent_metadata.get("guidance_scale")) != float(guidance_scale):
                raise ValueError(
                    "Clean latent cache guidance scale does not match this Wan run"
                )
            if clean_latent_metadata.get("prompt") != prompt:
                raise ValueError("Clean latent cache prompt does not match this Wan run")
            clean_latents = clean_latent_payload.get("latents")
            if not isinstance(clean_latents, torch.Tensor):
                raise ValueError("Clean latent cache has no latent tensor")
            draft_clean_latent_state["latents"] = clean_latents.to(
                device=latents.device,
                dtype=latents.dtype,
            )
            draft_clean_latent_state["metadata"] = clean_latent_metadata
            draft_clean_latent_state["loaded"] = True
            print(
                f"[draft-clean-latent] loaded={draft_clean_latent_cache_in_path} "
                f"shape={list(clean_latents.shape)}",
                flush=True,
            )

        if draft_vector_cache_in_path:
            vector_cache_payload = torch.load(
                draft_vector_cache_in_path,
                map_location="cpu",
                weights_only=False,
            )
            vector_cache_metadata = vector_cache_payload.get("metadata", {})
            if vector_cache_metadata.get("format_version") != 1:
                raise ValueError("Unsupported draft vector cache format")
            if vector_cache_metadata.get("num_inference_steps") != num_inference_steps:
                raise ValueError("Draft vector cache inference-step count does not match this Wan run")
            if tuple(vector_cache_metadata.get("latent_shape", ())) != tuple(latents.shape):
                raise ValueError(
                    "Draft vector cache latent shape does not match this Wan run: "
                    f"cache={vector_cache_metadata.get('latent_shape')} run={list(latents.shape)}"
                )
            if float(vector_cache_metadata.get("guidance_scale")) != float(guidance_scale):
                raise ValueError("Draft vector cache guidance scale does not match this Wan run")
            if vector_cache_metadata.get("prompt") != prompt:
                raise ValueError("Draft vector cache prompt does not match this Wan run")
            vector_predictions = vector_cache_payload.get("predictions", {})
            missing_steps = sorted(set(range(num_inference_steps)) - set(vector_predictions))
            if missing_steps:
                raise ValueError(f"Draft vector cache is missing sampling steps {missing_steps}")
            draft_vector_cache_state["predictions"] = vector_predictions
            draft_vector_cache_state["metadata"] = vector_cache_metadata
            draft_vector_cache_state["loaded"] = True
            print(
                f"[draft-vector-cache] loaded={draft_vector_cache_in_path} "
                f"entries={len(vector_predictions)}",
                flush=True,
            )

        attn_avg_end = num_inference_steps - 1 if attn_avg_end is None else attn_avg_end
        if attn_avg_start < 0 or attn_avg_end < attn_avg_start or attn_avg_end >= num_inference_steps:
            raise ValueError("attn_avg_start/end must be a valid inclusive interval of sampling-step indices")

        geometry_transport_start = 10 if geometry_transport_start is None else geometry_transport_start
        geometry_transport_end = min(39, num_inference_steps - 1) if geometry_transport_end is None else geometry_transport_end
        if geometry_transport_layers:
            if (
                geometry_transport_start < 0
                or geometry_transport_end < geometry_transport_start
                or geometry_transport_end >= num_inference_steps
            ):
                raise ValueError("geometry transport start/end must be a valid inclusive interval")

        attn_avg_processor_backups = []
        if (attn_avg_alpha > 0.0 and attn_avg_layers) or geometry_transport_layers:
            p_t, p_h, p_w = patch_size
            token_grid = (
                latents.shape[2] // p_t,
                latents.shape[3] // p_h,
                latents.shape[4] // p_w,
            )
            if geometry_transport_layers:
                geometry_map_sha256 = hashlib.sha256(
                    Path(draft_geometry_map_path).read_bytes()
                ).hexdigest()
                transport = load_draft_geometry_map(draft_geometry_map_path, latents.device)
                if tuple(transport.metadata.get("token_grid", ())) != token_grid:
                    raise ValueError(
                        "Draft geometry token grid does not match this Wan run: "
                        f"map={transport.metadata.get('token_grid')} run={token_grid}"
                    )
                if transport.source_time.shape[:2] != (token_grid[0] - 1, token_grid[1] * token_grid[2]):
                    raise ValueError("Draft geometry source map has an incompatible shape")
                if transport.confidence.shape[:3] != (token_grid[0] - 1, token_grid[1], token_grid[2]):
                    raise ValueError("Draft geometry confidence map has an incompatible shape")
                geometry_transport_state["source_time"] = transport.source_time
                geometry_transport_state["source_index"] = transport.source_index
                geometry_transport_state["confidence"] = transport.confidence
                geometry_transport_state["pair_stats"] = transport.pair_stats
                geometry_transport_state["metadata"] = transport.metadata
                geometry_transport_state["map_sha256"] = geometry_map_sha256
                geometry_transport_state["ready"] = True
                active_geometry = transport.confidence.reshape_as(
                    transport.source_time
                ) > 0
                if active_geometry.any():
                    active_confidence = transport.confidence.reshape_as(
                        transport.source_time
                    )[active_geometry]
                    if (
                        not torch.isfinite(active_confidence).all()
                        or (active_confidence < 0).any()
                        or (active_confidence > 1).any()
                    ):
                        raise ValueError(
                            "Draft geometry map active confidence must be finite "
                            "and lie in [0, 1]"
                        )
                    active_source_time = transport.source_time[active_geometry]
                    active_source_index = transport.source_index[active_geometry]
                    spatial_tokens = token_grid[1] * token_grid[2]
                    if (
                        (active_source_time < 0).any()
                        or (active_source_time >= token_grid[0]).any()
                    ):
                        raise ValueError(
                            "Draft geometry map contains an out-of-range active source_time"
                        )
                    if (
                        (active_source_index < 0).any()
                        or (active_source_index >= spatial_tokens).any()
                    ):
                        raise ValueError(
                            "Draft geometry map contains an out-of-range active source_index"
                        )
                if geometry_transport_mode in {
                    "geometry_surface_graph_logit_bias",
                    "geometry_signed_surface_graph_logit_bias",
                    "geometry_relative_edge_output",
                }:
                    if transport.source_time.shape[-1] != 1:
                        raise ValueError(
                            f"{geometry_transport_mode} requires a top-1 geometry map"
                        )
                    if (
                        geometry_transport_mode
                        in {
                            "geometry_signed_surface_graph_logit_bias",
                            "geometry_relative_edge_output",
                        }
                        and geometry_transport_surface_depth_threshold <= 0
                    ):
                        raise ValueError(
                            f"{geometry_transport_mode} requires a positive "
                            "geometry_transport_surface_depth_threshold"
                        )
                    if (
                        geometry_transport_mode
                        == "geometry_relative_edge_output"
                        and geometry_transport_max_relative_rms <= 0
                    ):
                        raise ValueError(
                            "geometry_relative_edge_output requires a positive "
                            "geometry_transport_max_relative_rms"
                        )
                    num_targets = token_grid[0] - 1
                    spatial_tokens = token_grid[1] * token_grid[2]
                    neighbor_specs = []
                    for graph_scale in geometry_transport_surface_graph_scales:
                        graph_weight = 1.0 / float(graph_scale)
                        neighbor_specs.extend(
                            (
                                (-graph_scale, 0, graph_weight, graph_scale),
                                (graph_scale, 0, graph_weight, graph_scale),
                                (0, -graph_scale, graph_weight, graph_scale),
                                (0, graph_scale, graph_weight, graph_scale),
                            )
                        )
                    num_neighbor_slots = len(neighbor_specs)
                    graph_neighbors = torch.full(
                        (num_targets, spatial_tokens, num_neighbor_slots),
                        -1,
                        dtype=torch.long,
                    )
                    graph_confidence = torch.zeros(
                        (num_targets, spatial_tokens, num_neighbor_slots),
                        dtype=torch.float32,
                    )
                    graph_scale_weight = torch.zeros(
                        (num_targets, spatial_tokens, num_neighbor_slots),
                        dtype=torch.float32,
                    )
                    boundary_neighbors = torch.full(
                        (num_targets, spatial_tokens, num_neighbor_slots),
                        -1,
                        dtype=torch.long,
                    )
                    boundary_confidence = torch.zeros(
                        (num_targets, spatial_tokens, num_neighbor_slots),
                        dtype=torch.float32,
                    )
                    source_depth_tokens = None
                    source_depth_frame_to_index = {}
                    temporal_scale = int(
                        transport.metadata.get("temporal_scale", 4)
                    )
                    if geometry_transport_surface_depth_threshold > 0:
                        geometry_cache_path = transport.metadata.get("geometry")
                        if not geometry_cache_path:
                            raise ValueError(
                                "Surface-depth filtering requires metadata['geometry']"
                            )
                        geometry_payload = torch.load(
                            geometry_cache_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                        geometry_depth = geometry_payload["depth"][
                            ...,
                            0,
                        ].detach().float().cpu()
                        image_height, image_width = (
                            transport.metadata["image_size"]
                        )
                        height_tokens, width_tokens = token_grid[1:]
                        if (
                            image_height % height_tokens
                            or image_width % width_tokens
                        ):
                            raise ValueError(
                                "Geometry image size is not divisible by the Wan token grid"
                            )
                        patch_height = image_height // height_tokens
                        patch_width = image_width // width_tokens
                        source_depth_tokens = geometry_depth.reshape(
                            geometry_depth.shape[0],
                            height_tokens,
                            patch_height,
                            width_tokens,
                            patch_width,
                        ).permute(
                            0,
                            1,
                            3,
                            2,
                            4,
                        ).reshape(
                            geometry_depth.shape[0],
                            height_tokens,
                            width_tokens,
                            patch_height * patch_width,
                        ).median(dim=-1).values
                        source_depth_frame_to_index = {
                            int(frame): index
                            for index, frame in enumerate(
                                geometry_payload["frame_indices"]
                            )
                        }
                    source_time_cpu = transport.source_time[..., 0].detach().cpu()
                    source_index_cpu = transport.source_index[..., 0].detach().cpu()
                    confidence_cpu = (
                        transport.confidence[..., 0]
                        .reshape(num_targets, spatial_tokens)
                        .detach()
                        .float()
                        .cpu()
                    )
                    height_tokens, width_tokens = token_grid[1:]
                    for target_offset in range(num_targets):
                        active_targets = (
                            confidence_cpu[target_offset] > 0
                        ).nonzero(as_tuple=False).squeeze(-1)
                        best_target_by_source = {}
                        for target_spatial in active_targets.tolist():
                            source_key = (
                                int(source_time_cpu[target_offset, target_spatial]),
                                int(source_index_cpu[target_offset, target_spatial]),
                            )
                            confidence_value = float(
                                confidence_cpu[target_offset, target_spatial]
                            )
                            previous = best_target_by_source.get(source_key)
                            if previous is None or confidence_value > previous[1]:
                                best_target_by_source[source_key] = (
                                    target_spatial,
                                    confidence_value,
                                )
                        for target_spatial in active_targets.tolist():
                            source_token = int(
                                source_index_cpu[target_offset, target_spatial]
                            )
                            source_token_time = int(
                                source_time_cpu[target_offset, target_spatial]
                            )
                            source_y, source_x = divmod(
                                source_token,
                                width_tokens,
                            )
                            for edge_index, (
                                delta_y,
                                delta_x,
                                graph_weight,
                                graph_scale,
                            ) in enumerate(
                                neighbor_specs
                            ):
                                neighbor_y = source_y + delta_y
                                neighbor_x = source_x + delta_x
                                if not (
                                    0 <= neighbor_y < height_tokens
                                    and 0 <= neighbor_x < width_tokens
                                ):
                                    continue
                                neighbor_source = (
                                    source_token_time,
                                    neighbor_y * width_tokens + neighbor_x,
                                )
                                neighbor = best_target_by_source.get(neighbor_source)
                                if neighbor is None:
                                    continue
                                if source_depth_tokens is not None:
                                    source_frame = (
                                        source_token_time * temporal_scale
                                    )
                                    if (
                                        source_frame
                                        not in source_depth_frame_to_index
                                    ):
                                        raise ValueError(
                                            "Geometry cache is missing source frame "
                                            f"{source_frame}"
                                        )
                                    geometry_index = (
                                        source_depth_frame_to_index[source_frame]
                                    )
                                    step_y = delta_y // graph_scale
                                    step_x = delta_x // graph_scale
                                    path_invalid = False
                                    path_crosses_boundary = False
                                    for path_step in range(graph_scale):
                                        depth_a = source_depth_tokens[
                                            geometry_index,
                                            source_y + path_step * step_y,
                                            source_x + path_step * step_x,
                                        ]
                                        depth_b = source_depth_tokens[
                                            geometry_index,
                                            source_y + (path_step + 1) * step_y,
                                            source_x + (path_step + 1) * step_x,
                                        ]
                                        if (
                                            not torch.isfinite(depth_a)
                                            or not torch.isfinite(depth_b)
                                            or depth_a <= 0
                                            or depth_b <= 0
                                        ):
                                            path_invalid = True
                                            break
                                        relative_depth_jump = (
                                            (depth_a - depth_b).abs()
                                            / torch.minimum(
                                                depth_a.abs(),
                                                depth_b.abs(),
                                            ).clamp_min(1e-6)
                                        )
                                        if (
                                            relative_depth_jump
                                            > geometry_transport_surface_depth_threshold
                                        ):
                                            path_crosses_boundary = True
                                            break
                                    if path_invalid:
                                        continue
                                    if path_crosses_boundary:
                                        if (
                                            geometry_transport_mode
                                            == "geometry_signed_surface_graph_logit_bias"
                                        ):
                                            neighbor_target, neighbor_confidence = neighbor
                                            if neighbor_target != target_spatial:
                                                boundary_neighbors[
                                                    target_offset,
                                                    target_spatial,
                                                    edge_index,
                                                ] = neighbor_target
                                                boundary_confidence[
                                                    target_offset,
                                                    target_spatial,
                                                    edge_index,
                                                ] = min(
                                                    float(
                                                        confidence_cpu[
                                                            target_offset,
                                                            target_spatial,
                                                        ]
                                                    ),
                                                    neighbor_confidence,
                                                )
                                        continue
                                neighbor_target, neighbor_confidence = neighbor
                                if neighbor_target == target_spatial:
                                    continue
                                graph_neighbors[
                                    target_offset,
                                    target_spatial,
                                    edge_index,
                                ] = neighbor_target
                                graph_confidence[
                                    target_offset,
                                    target_spatial,
                                    edge_index,
                                ] = min(
                                    float(
                                        confidence_cpu[
                                            target_offset,
                                            target_spatial,
                                        ]
                                    ),
                                    neighbor_confidence,
                                )
                                graph_scale_weight[
                                    target_offset,
                                    target_spatial,
                                    edge_index,
                                ] = graph_weight
                    geometry_transport_state["surface_graph_neighbors"] = (
                        graph_neighbors.to(latents.device)
                    )
                    geometry_transport_state["surface_graph_confidence"] = (
                        graph_confidence.to(latents.device)
                    )
                    geometry_transport_state["surface_graph_scale_weight"] = (
                        graph_scale_weight.to(latents.device)
                    )
                    geometry_transport_state["surface_boundary_neighbors"] = (
                        boundary_neighbors.to(latents.device)
                    )
                    geometry_transport_state["surface_boundary_confidence"] = (
                        boundary_confidence.to(latents.device)
                    )
                    relative_edge_records = []
                    relative_edge_count = 0
                    for target_offset in range(num_targets):
                        canonical_edges = {}
                        for target_spatial in range(spatial_tokens):
                            for edge_index in range(num_neighbor_slots):
                                neighbor_target = int(
                                    graph_neighbors[
                                        target_offset,
                                        target_spatial,
                                        edge_index,
                                    ]
                                )
                                if neighbor_target < 0 or neighbor_target == target_spatial:
                                    continue
                                edge_key = tuple(
                                    sorted((target_spatial, neighbor_target))
                                )
                                edge_confidence = float(
                                    graph_confidence[
                                        target_offset,
                                        target_spatial,
                                        edge_index,
                                    ]
                                )
                                edge_scale_weight = float(
                                    graph_scale_weight[
                                        target_offset,
                                        target_spatial,
                                        edge_index,
                                    ]
                                )
                                previous_edge = canonical_edges.get(edge_key)
                                if (
                                    previous_edge is None
                                    or edge_confidence * edge_scale_weight
                                    > previous_edge[0] * previous_edge[1]
                                ):
                                    canonical_edges[edge_key] = (
                                        edge_confidence,
                                        edge_scale_weight,
                                    )

                        target_i_values = []
                        target_j_values = []
                        source_i_values = []
                        source_j_values = []
                        edge_confidence_values = []
                        edge_scale_weight_values = []
                        for (
                            target_i,
                            target_j,
                        ), (
                            edge_confidence,
                            edge_scale_weight,
                        ) in sorted(
                            canonical_edges.items()
                        ):
                            source_i_time = int(
                                source_time_cpu[target_offset, target_i]
                            )
                            source_j_time = int(
                                source_time_cpu[target_offset, target_j]
                            )
                            if source_i_time != source_j_time:
                                raise ValueError(
                                    "A surface edge connected endpoints from different "
                                    "source anchor times"
                                )
                            target_i_values.append(target_i)
                            target_j_values.append(target_j)
                            source_i_values.append(
                                source_i_time * spatial_tokens
                                + int(source_index_cpu[target_offset, target_i])
                            )
                            source_j_values.append(
                                source_j_time * spatial_tokens
                                + int(source_index_cpu[target_offset, target_j])
                            )
                            edge_confidence_values.append(edge_confidence)
                            edge_scale_weight_values.append(edge_scale_weight)

                        relative_edge_count += len(target_i_values)
                        relative_edge_records.append(
                            {
                                "target_i": torch.tensor(
                                    target_i_values,
                                    dtype=torch.long,
                                    device=latents.device,
                                ),
                                "target_j": torch.tensor(
                                    target_j_values,
                                    dtype=torch.long,
                                    device=latents.device,
                                ),
                                "source_i": torch.tensor(
                                    source_i_values,
                                    dtype=torch.long,
                                    device=latents.device,
                                ),
                                "source_j": torch.tensor(
                                    source_j_values,
                                    dtype=torch.long,
                                    device=latents.device,
                                ),
                                "confidence": torch.tensor(
                                    edge_confidence_values,
                                    dtype=torch.float32,
                                    device=latents.device,
                                ),
                                "scale_weight": torch.tensor(
                                    edge_scale_weight_values,
                                    dtype=torch.float32,
                                    device=latents.device,
                                ),
                            }
                        )
                    geometry_transport_state["surface_relative_edges"] = (
                        relative_edge_records
                    )
                    graph_support = graph_neighbors.ge(0)
                    boundary_support = boundary_neighbors.ge(0)
                    active_graph_queries = graph_support
                    if (
                        geometry_transport_mode
                        == "geometry_signed_surface_graph_logit_bias"
                    ):
                        active_graph_queries = (
                            graph_support | boundary_support
                        )
                    print(
                        "[draft-geometry] surface_graph "
                        f"directed_edges={graph_support.sum().item()} "
                        f"boundary_edges={boundary_support.sum().item()} "
                        f"undirected_edges={relative_edge_count} "
                        f"active_queries={active_graph_queries.any(dim=-1).sum().item()} "
                        "depth_threshold="
                        f"{geometry_transport_surface_depth_threshold:.4f} "
                        f"scales={geometry_transport_surface_graph_scales}",
                        flush=True,
                    )
                if draft_feature_cache_out_path or draft_feature_cache_in_path:
                    map_fingerprint = transport.metadata.get("draft_run_fingerprint")
                    if map_fingerprint != draft_feature_cache_fingerprint:
                        raise ValueError(
                            "Draft geometry map run fingerprint does not match this Wan run: "
                            f"map={map_fingerprint} run={draft_feature_cache_fingerprint}"
                        )
                anchor_video_frames = transport.metadata.get("anchor_video_frames", [])
                temporal_scale = int(transport.metadata.get("temporal_scale", 4))
                anchor_token_times = sorted(
                    {
                        frame_to_latent_index(
                            int(frame),
                            temporal_scale,
                            token_grid[0],
                        )
                        for frame in anchor_video_frames
                    }
                )
                if not anchor_token_times:
                    raise ValueError("Draft geometry map does not define anchor_video_frames")
                if active_geometry.any() and geometry_transport_mode in cached_transport_modes:
                    active_source_times = set(
                        transport.source_time[active_geometry].detach().cpu().tolist()
                    )
                    missing_anchor_times = active_source_times - set(anchor_token_times)
                    if missing_anchor_times:
                        raise ValueError(
                            "Draft geometry map references active source times that are "
                            f"not cached anchors: {sorted(missing_anchor_times)}"
                        )
                draft_feature_cache_state["anchor_token_times"] = anchor_token_times
                coverage = transport.confidence.amax(dim=-1).gt(0).float().mean().item()
                print(
                    f"[draft-geometry] loaded={draft_geometry_map_path} "
                    f"coverage={coverage:.3f} pairs={len(transport.pair_stats)}",
                    flush=True,
                )
                if draft_feature_cache_in_path:
                    cache_payload = torch.load(
                        draft_feature_cache_in_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    cache_metadata = cache_payload.get("metadata", {})
                    cached_kind = cache_metadata.get("cache_kind", "block_output")
                    if geometry_transport_mode == "cached_virtual_token_bias":
                        required_kind = "attention_kv"
                    elif geometry_transport_mode in {
                        "cached_draft_value_delta",
                        "cached_draft_output_delta",
                        "cached_draft_output_delta_centered",
                    }:
                        required_kind = "attention_value_delta"
                    elif geometry_transport_mode == "cached_draft_value_delta_consensus":
                        required_kind = "attention_value_delta_consensus"
                    elif geometry_transport_mode in {
                            "cached_value_transport",
                            "cached_value_residual",
                            "cached_value_memory",
                            "cached_value_attention_blend",
                    }:
                        required_kind = "attention_value"
                    else:
                        required_kind = "block_output"
                    if cached_kind != required_kind:
                        raise ValueError(
                            "Draft feature cache kind does not match the transport mode: "
                            f"cache={cached_kind} required={required_kind}"
                        )
                    if (
                        cached_kind == "attention_kv"
                        and cache_metadata.get("format_version") != 2
                    ):
                        raise ValueError(
                            "Draft K/V cache has an unsupported format version: "
                            f"{cache_metadata.get('format_version')}"
                        )
                    if (
                        cached_kind == "attention_value_delta"
                        and cache_metadata.get("format_version") != 3
                    ):
                        raise ValueError(
                            "Draft value-delta cache has an unsupported format version: "
                            f"{cache_metadata.get('format_version')}"
                        )
                    if (
                        cached_kind == "attention_value_delta_consensus"
                        and cache_metadata.get("format_version") != 4
                    ):
                        raise ValueError(
                            "Draft consensus-delta cache has an unsupported format version: "
                            f"{cache_metadata.get('format_version')}"
                        )
                    if tuple(cache_metadata.get("token_grid", ())) != token_grid:
                        raise ValueError(
                            "Draft feature cache token grid does not match this Wan run: "
                            f"cache={cache_metadata.get('token_grid')} run={token_grid}"
                        )
                    if cache_metadata.get("num_inference_steps") != num_inference_steps:
                        raise ValueError(
                            "Draft feature cache inference-step count does not match this Wan run"
                        )
                    cached_fingerprint = cache_metadata.get("run_fingerprint")
                    if cached_fingerprint != draft_feature_cache_fingerprint:
                        raise ValueError(
                            "Draft feature cache run fingerprint does not match this Wan run: "
                            f"cache={cached_fingerprint} run={draft_feature_cache_fingerprint}"
                        )
                    cached_map_sha256 = cache_metadata.get(
                        "geometry_map_sha256"
                    )
                    if cached_map_sha256 != geometry_map_sha256:
                        raise ValueError(
                            "Draft feature cache geometry map does not match "
                            "this run: "
                            f"cache={cached_map_sha256} "
                            f"run={geometry_map_sha256}"
                        )
                    cached_anchor_times = list(cache_metadata.get("anchor_token_times", []))
                    if cached_anchor_times != anchor_token_times:
                        raise ValueError(
                            "Draft feature cache anchors do not match the geometry map: "
                            f"cache={cached_anchor_times} map={anchor_token_times}"
                        )
                    cached_layers = set(cache_metadata.get("layers", []))
                    missing_layers = set(geometry_transport_layers) - cached_layers
                    if missing_layers:
                        raise ValueError(f"Draft feature cache is missing layers {sorted(missing_layers)}")
                    cached_features = cache_payload["features"]
                    if cached_kind == "attention_kv":
                        cached_interval = cache_metadata.get(
                            "geometry_transport_interval"
                        )
                        expected_interval = [
                            geometry_transport_start,
                            geometry_transport_end,
                        ]
                        if cached_interval != expected_interval:
                            raise ValueError(
                                "Draft K/V cache interval does not match this run: "
                                f"cache={cached_interval} run={expected_interval}"
                            )
                        spatial_tokens = token_grid[1] * token_grid[2]
                        for step_index in range(
                            geometry_transport_start,
                            geometry_transport_end + 1,
                        ):
                            for layer_index in geometry_transport_layers:
                                suffix = f":{layer_index}:{step_index}"
                                matching_keys = [
                                    key
                                    for key in cached_features
                                    if key.endswith(suffix)
                                ]
                                if len(matching_keys) != 1:
                                    raise ValueError(
                                        "Draft K/V cache must contain exactly one active "
                                        f"transformer entry for {suffix}, found {matching_keys}"
                                    )
                                cache_entry = cached_features[matching_keys[0]]
                                if (
                                    not isinstance(cache_entry, dict)
                                    or set(cache_entry) != {"key", "value"}
                                ):
                                    raise ValueError(
                                        f"Draft K/V cache entry {matching_keys[0]} is incomplete"
                                    )
                                key_shape = cache_entry["key"].shape
                                value_shape = cache_entry["value"].shape
                                expected_prefix = (
                                    latents.shape[0],
                                    len(anchor_token_times),
                                    spatial_tokens,
                                )
                                if (
                                    key_shape != value_shape
                                    or key_shape[:3] != expected_prefix
                                ):
                                    raise ValueError(
                                        f"Draft K/V cache entry {matching_keys[0]} has "
                                        f"incompatible shapes key={tuple(key_shape)} "
                                        f"value={tuple(value_shape)} expected_prefix="
                                        f"{expected_prefix}"
                                    )
                    elif cached_kind == "attention_value_delta":
                        cached_interval = cache_metadata.get(
                            "geometry_transport_interval"
                        )
                        expected_interval = [
                            geometry_transport_start,
                            geometry_transport_end,
                        ]
                        if cached_interval != expected_interval:
                            raise ValueError(
                                "Draft value-delta cache interval does not match this run: "
                                f"cache={cached_interval} run={expected_interval}"
                            )
                        for step_index in range(
                            geometry_transport_start,
                            geometry_transport_end + 1,
                        ):
                            for layer_index in geometry_transport_layers:
                                suffix = f":{layer_index}:{step_index}"
                                matching_keys = [
                                    key
                                    for key in cached_features
                                    if key.endswith(suffix)
                                ]
                                if len(matching_keys) != 1:
                                    raise ValueError(
                                        "Draft value-delta cache must contain exactly "
                                        f"one active transformer entry for {suffix}, "
                                        f"found {matching_keys}"
                                    )
                                cache_entry = cached_features[matching_keys[0]]
                                if (
                                    not isinstance(cache_entry, dict)
                                    or set(cache_entry)
                                    != {"target_flat", "delta", "confidence"}
                                ):
                                    raise ValueError(
                                        "Draft value-delta cache entry "
                                        f"{matching_keys[0]} is incomplete"
                                    )
                                target_flat = cache_entry["target_flat"]
                                delta = cache_entry["delta"]
                                confidence = cache_entry["confidence"]
                                if (
                                    target_flat.ndim != 1
                                    or delta.ndim != 3
                                    or confidence.ndim != 1
                                    or delta.shape[1] != target_flat.numel()
                                    or confidence.numel() != target_flat.numel()
                                    or delta.shape[0] != latents.shape[0]
                                ):
                                    raise ValueError(
                                        "Draft value-delta cache entry "
                                        f"{matching_keys[0]} has incompatible shapes: "
                                        f"target_flat={tuple(target_flat.shape)} "
                                        f"delta={tuple(delta.shape)} "
                                        f"confidence={tuple(confidence.shape)}"
                                    )
                    elif cached_kind == "attention_value_delta_consensus":
                        cached_interval = cache_metadata.get(
                            "geometry_transport_interval"
                        )
                        expected_interval = [
                            geometry_transport_start,
                            geometry_transport_end,
                        ]
                        if cached_interval != expected_interval:
                            raise ValueError(
                                "Draft consensus-delta cache interval does not match this run: "
                                f"cache={cached_interval} run={expected_interval}"
                            )
                        required_fields = {
                            "target_flat",
                            "delta",
                            "confidence",
                            "agreement",
                            "unique_anchor_count",
                            "effective_anchor_count",
                        }
                        for step_index in range(
                            geometry_transport_start,
                            geometry_transport_end + 1,
                        ):
                            for layer_index in geometry_transport_layers:
                                suffix = f":{layer_index}:{step_index}"
                                matching_keys = [
                                    key
                                    for key in cached_features
                                    if key.endswith(suffix)
                                ]
                                if len(matching_keys) != 1:
                                    raise ValueError(
                                        "Draft consensus-delta cache must contain exactly "
                                        f"one active transformer entry for {suffix}, "
                                        f"found {matching_keys}"
                                    )
                                cache_entry = cached_features[matching_keys[0]]
                                if (
                                    not isinstance(cache_entry, dict)
                                    or set(cache_entry) != required_fields
                                ):
                                    raise ValueError(
                                        "Draft consensus-delta cache entry "
                                        f"{matching_keys[0]} is incomplete"
                                    )
                                target_flat = cache_entry["target_flat"]
                                delta = cache_entry["delta"]
                                token_count = target_flat.numel()
                                if (
                                    target_flat.ndim != 1
                                    or delta.ndim != 3
                                    or delta.shape[0] != latents.shape[0]
                                    or delta.shape[1] != token_count
                                    or any(
                                        cache_entry[name].ndim != 1
                                        or cache_entry[name].numel() != token_count
                                        for name in (
                                            "confidence",
                                            "agreement",
                                            "unique_anchor_count",
                                            "effective_anchor_count",
                                        )
                                    )
                                ):
                                    raise ValueError(
                                        "Draft consensus-delta cache entry "
                                        f"{matching_keys[0]} has incompatible shapes"
                                    )
                    draft_feature_cache_state["features"] = cached_features
                    draft_feature_cache_state["metadata"] = cache_metadata
                    draft_feature_cache_state["kind"] = cached_kind
                    draft_feature_cache_state["loaded"] = True
                    print(
                        f"[draft-feature-cache] loaded={draft_feature_cache_in_path} "
                        f"entries={len(draft_feature_cache_state['features'])} "
                        f"kind={cached_kind} anchors={anchor_token_times}",
                        flush=True,
                    )

                if draft_vector_anchor_outside_geometry:
                    support = transport.confidence.amax(dim=-1).gt(0).to(dtype=torch.float32)
                    support = torch.cat(
                        [torch.zeros_like(support[:1]), support],
                        dim=0,
                    )
                    support = support.unsqueeze(0).unsqueeze(0)
                    if draft_vector_mask_spatial_dilation or draft_vector_mask_temporal_dilation:
                        kernel = (
                            2 * draft_vector_mask_temporal_dilation + 1,
                            2 * draft_vector_mask_spatial_dilation + 1,
                            2 * draft_vector_mask_spatial_dilation + 1,
                        )
                        padding = (
                            draft_vector_mask_temporal_dilation,
                            draft_vector_mask_spatial_dilation,
                            draft_vector_mask_spatial_dilation,
                        )
                        support = torch.nn.functional.max_pool3d(
                            support,
                            kernel_size=kernel,
                            stride=1,
                            padding=padding,
                        )
                    support[:, :, 0] = 0
                    support = torch.nn.functional.interpolate(
                        support,
                        size=latents.shape[2:],
                        mode="nearest",
                    )
                    draft_vector_cache_state["geometry_mask"] = support.to(
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    print(
                        "[draft-vector-anchor] "
                        f"mask_coverage={support.mean().item():.4f} "
                        f"temporal_dilation={draft_vector_mask_temporal_dilation} "
                        f"spatial_dilation={draft_vector_mask_spatial_dilation}",
                        flush=True,
                    )

            def _current_geometry_alpha() -> float:
                if geometry_transport_end == geometry_transport_start:
                    progress = 0.0
                else:
                    progress = (
                        attn_avg_state["step"] - geometry_transport_start
                    ) / (geometry_transport_end - geometry_transport_start)
                progress = max(0.0, min(1.0, float(progress)))
                if geometry_transport_schedule == "constant":
                    multiplier = 1.0
                elif geometry_transport_schedule == "linear_decay":
                    multiplier = 1.0 - progress
                else:
                    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
                return geometry_transport_alpha * multiplier

            def _current_draft_vector_geometry_alpha() -> float:
                if geometry_transport_end == geometry_transport_start:
                    progress = 0.0
                else:
                    progress = (
                        attn_avg_state["step"] - geometry_transport_start
                    ) / (geometry_transport_end - geometry_transport_start)
                progress = max(0.0, min(1.0, float(progress)))
                if geometry_transport_schedule == "constant":
                    multiplier = 1.0
                elif geometry_transport_schedule == "linear_decay":
                    multiplier = 1.0 - progress
                else:
                    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
                return draft_vector_geometry_alpha * multiplier

            def _current_draft_clean_latent_alpha() -> float:
                if geometry_transport_end == geometry_transport_start:
                    progress = 0.0
                else:
                    progress = (
                        attn_avg_state["step"] - geometry_transport_start
                    ) / (geometry_transport_end - geometry_transport_start)
                progress = max(0.0, min(1.0, float(progress)))
                if geometry_transport_schedule == "constant":
                    multiplier = 1.0
                elif geometry_transport_schedule == "linear_decay":
                    multiplier = 1.0 - progress
                else:
                    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
                return draft_clean_latent_alpha * multiplier

            def _matched_clean_latent_residual(
                current_x0: torch.Tensor,
                return_target: bool = False,
                spatial_dilation: int = 0,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Transport clean anchor latents to geometry-matched target tokens."""
                if not draft_clean_latent_state["loaded"]:
                    raise RuntimeError("Clean latent transport was enabled without a loaded cache")

                p_t, p_h, p_w = patch_size
                batch_size, channels = current_x0.shape[:2]
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                clean_latents = draft_clean_latent_state["latents"].to(
                    device=current_x0.device,
                    dtype=current_x0.dtype,
                )

                def _patchify_latents(latent: torch.Tensor) -> torch.Tensor:
                    expected_shape = (
                        num_tokens_t * p_t,
                        height_tokens * p_h,
                        width_tokens * p_w,
                    )
                    if tuple(latent.shape[2:]) != expected_shape:
                        raise ValueError(
                            "Clean latent transport token grid does not match the "
                            f"latent shape: latent={list(latent.shape[2:])} "
                            f"expected={list(expected_shape)}"
                        )
                    return (
                        latent.float()
                        .reshape(
                            batch_size,
                            channels,
                            num_tokens_t,
                            p_t,
                            height_tokens,
                            p_h,
                            width_tokens,
                            p_w,
                        )
                        .permute(0, 2, 4, 6, 1, 3, 5, 7)
                        .reshape(
                            batch_size,
                            num_tokens_t,
                            spatial_tokens,
                            channels * p_t * p_h * p_w,
                        )
                    )

                if draft_clean_latent_transport_mode == "full_patch":
                    current_flat = _patchify_latents(current_x0)
                    clean_flat = _patchify_latents(clean_latents)
                    transport_channels = channels * p_t * p_h * p_w
                else:
                    current_tokens = torch.nn.functional.avg_pool3d(
                        current_x0.float(),
                        kernel_size=(p_t, p_h, p_w),
                        stride=(p_t, p_h, p_w),
                    )
                    clean_tokens = torch.nn.functional.avg_pool3d(
                        clean_latents.float(),
                        kernel_size=(p_t, p_h, p_w),
                        stride=(p_t, p_h, p_w),
                    )
                    if (
                        tuple(current_tokens.shape[2:]) != token_grid
                        or tuple(clean_tokens.shape[2:]) != token_grid
                    ):
                        raise ValueError(
                            "Clean latent transport token grid does not match the geometry map"
                        )
                    current_flat = current_tokens.permute(0, 2, 3, 4, 1).reshape(
                        batch_size, num_tokens_t, spatial_tokens, channels
                    )
                    clean_flat = clean_tokens.permute(0, 2, 3, 4, 1).reshape(
                        batch_size, num_tokens_t, spatial_tokens, channels
                    )
                    transport_channels = channels
                all_sources = clean_flat.reshape(
                    batch_size,
                    num_tokens_t * spatial_tokens,
                    transport_channels,
                )
                matched = torch.zeros_like(current_flat[:, 1:])
                target_gate = torch.zeros(
                    (num_tokens_t - 1, spatial_tokens, 1),
                    device=current_x0.device,
                    dtype=torch.float32,
                )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = confidence.shape[-1]

                if spatial_dilation > 0:
                    # Expand only unsupported target tokens. The strongest nearby
                    # correspondence supplies a local translation, so neighboring
                    # target tokens keep their relative 2D layout in source space
                    # instead of all collapsing onto the donor's source token.
                    target_steps = num_tokens_t - 1
                    source_time_grid = source_time.reshape(
                        target_steps,
                        height_tokens,
                        width_tokens,
                        memory_slots,
                    )
                    source_index_grid = source_index.reshape(
                        target_steps,
                        height_tokens,
                        width_tokens,
                        memory_slots,
                    )
                    confidence_grid = confidence.reshape(
                        target_steps,
                        height_tokens,
                        width_tokens,
                        memory_slots,
                    )
                    support_score = confidence_grid.amax(dim=-1)
                    pooled_score, donor_index = torch.nn.functional.max_pool2d(
                        support_score.unsqueeze(1),
                        kernel_size=2 * spatial_dilation + 1,
                        stride=1,
                        padding=spatial_dilation,
                        return_indices=True,
                    )
                    pooled_score = pooled_score.squeeze(1)
                    donor_index = donor_index.squeeze(1)
                    fill_support = (support_score <= 0) & (pooled_score > 0)

                    if fill_support.any():
                        donor_flat = donor_index.reshape(
                            target_steps,
                            spatial_tokens,
                            1,
                        ).expand(-1, -1, memory_slots)

                        def _gather_donor(values: torch.Tensor) -> torch.Tensor:
                            return values.reshape(
                                target_steps,
                                spatial_tokens,
                                memory_slots,
                            ).gather(1, donor_flat).reshape(
                                target_steps,
                                height_tokens,
                                width_tokens,
                                memory_slots,
                            )

                        donor_source_time = _gather_donor(source_time_grid)
                        donor_source_index = _gather_donor(source_index_grid)
                        donor_confidence = _gather_donor(confidence_grid)

                        target_y = torch.arange(
                            height_tokens,
                            device=confidence.device,
                        ).view(1, height_tokens, 1)
                        target_x = torch.arange(
                            width_tokens,
                            device=confidence.device,
                        ).view(1, 1, width_tokens)
                        donor_y = torch.div(
                            donor_index,
                            width_tokens,
                            rounding_mode="floor",
                        )
                        donor_x = donor_index.remainder(width_tokens)
                        offset_y = (target_y - donor_y).unsqueeze(-1)
                        offset_x = (target_x - donor_x).unsqueeze(-1)

                        donor_source_y = torch.div(
                            donor_source_index,
                            width_tokens,
                            rounding_mode="floor",
                        )
                        donor_source_x = donor_source_index.remainder(width_tokens)
                        propagated_source_y = (
                            donor_source_y + offset_y
                        ).clamp(0, height_tokens - 1)
                        propagated_source_x = (
                            donor_source_x + offset_x
                        ).clamp(0, width_tokens - 1)
                        propagated_source_index = (
                            propagated_source_y * width_tokens
                            + propagated_source_x
                        )

                        fill_slots = fill_support.unsqueeze(-1)
                        source_time = torch.where(
                            fill_slots,
                            donor_source_time,
                            source_time_grid,
                        ).reshape_as(source_time)
                        source_index = torch.where(
                            fill_slots,
                            propagated_source_index,
                            source_index_grid,
                        ).reshape_as(source_index)
                        confidence = torch.where(
                            fill_slots,
                            donor_confidence,
                            confidence_grid,
                        ).reshape_as(confidence)

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(
                        spatial_tokens, memory_slots
                    )
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue
                    source_flat = (
                        source_time[target_time - 1].reshape(
                            spatial_tokens, memory_slots
                        )
                        * spatial_tokens
                        + source_index[target_time - 1].reshape(
                            spatial_tokens, memory_slots
                        )
                    ).reshape(-1)
                    source_values = all_sources[:, source_flat].reshape(
                        batch_size,
                        spatial_tokens,
                        memory_slots,
                        transport_channels,
                    )
                    weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1)
                    matched[:, target_time - 1] = (
                        weights * source_values
                    ).sum(dim=2)
                    max_confidence = confidence_t.amax(dim=-1, keepdim=True)
                    if draft_clean_latent_gate_mode == "binary_support":
                        max_confidence = (max_confidence > 0).to(
                            max_confidence.dtype
                        )
                    target_gate[target_time - 1] = max_confidence

                transported = torch.zeros_like(current_flat)
                if return_target:
                    transported[:, 1:] = matched
                else:
                    if draft_clean_latent_reference_mode in {
                        "draft_delta_centered",
                        "draft_delta_highpass",
                        "draft_delta_lowpass",
                    }:
                        residual = matched.float() - clean_flat[:, 1:].float()
                        filtered = torch.zeros_like(residual)
                        for target_idx in range(num_tokens_t - 1):
                            gate_t = target_gate[target_idx, :, 0]
                            active = gate_t > 0
                            if not active.any():
                                continue

                            if (
                                draft_clean_latent_reference_mode
                                in {
                                    "draft_delta_highpass",
                                    "draft_delta_lowpass",
                                }
                            ):
                                radius = draft_clean_latent_lowpass_radius
                                kernel_size = 2 * radius + 1
                                residual_grid = residual[:, target_idx].reshape(
                                    batch_size,
                                    height_tokens,
                                    width_tokens,
                                    transport_channels,
                                ).permute(0, 3, 1, 2)
                                gate_grid = gate_t.reshape(
                                    1, 1, height_tokens, width_tokens
                                ).expand(batch_size, -1, -1, -1)
                                local_sum = torch.nn.functional.avg_pool2d(
                                    residual_grid * gate_grid,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=radius,
                                )
                                local_weight = torch.nn.functional.avg_pool2d(
                                    gate_grid,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=radius,
                                )
                                local_mean = local_sum / local_weight.clamp_min(1e-8)
                                if (
                                    draft_clean_latent_reference_mode
                                    == "draft_delta_lowpass"
                                ):
                                    filtered_grid = local_mean
                                else:
                                    filtered_grid = residual_grid - local_mean
                                filtered_t = filtered_grid.permute(
                                    0, 2, 3, 1
                                ).reshape(
                                    batch_size,
                                    spatial_tokens,
                                    transport_channels,
                                )
                            else:
                                weights_t = gate_t[active].reshape(1, -1, 1)
                                active_residual = residual[:, target_idx, active]
                                center = (
                                    active_residual * weights_t
                                ).sum(dim=1, keepdim=True) / weights_t.sum(
                                    dim=1, keepdim=True
                                ).clamp_min(1e-8)
                                filtered_t = residual[:, target_idx] - center

                            if draft_clean_latent_delta_quantile > 0.0:
                                magnitude = filtered_t.square().mean(dim=-1).sqrt()
                                threshold = torch.quantile(
                                    magnitude[:, active],
                                    draft_clean_latent_delta_quantile,
                                    dim=1,
                                    keepdim=True,
                                )
                                keep = magnitude >= threshold
                                filtered_t = filtered_t * keep.unsqueeze(-1)
                            filtered[:, target_idx] = filtered_t
                        residual = filtered.to(current_flat.dtype)
                    elif draft_clean_latent_reference_mode == "draft_delta":
                        # Inject only the correction measured inside the baseline
                        # draft, preserving the current sample's target-view content.
                        residual = matched - clean_flat[:, 1:]
                    else:
                        residual = matched - current_flat[:, 1:]
                    transported[:, 1:] = (
                        residual
                    ) * target_gate.reshape(
                        1,
                        num_tokens_t - 1,
                        spatial_tokens,
                        1,
                    )
                transported = transported.reshape(
                    batch_size,
                    num_tokens_t,
                    height_tokens,
                    width_tokens,
                    transport_channels,
                )
                gate = torch.cat(
                    [torch.zeros_like(target_gate[:1]), target_gate],
                    dim=0,
                ).reshape(1, 1, num_tokens_t, height_tokens, width_tokens)
                if draft_clean_latent_transport_mode == "full_patch":
                    transported = (
                        transported.reshape(
                            batch_size,
                            num_tokens_t,
                            height_tokens,
                            width_tokens,
                            channels,
                            p_t,
                            p_h,
                            p_w,
                        )
                        .permute(0, 4, 1, 5, 2, 6, 3, 7)
                        .reshape_as(current_x0)
                        .to(current_x0.dtype)
                    )
                else:
                    transported = torch.nn.functional.interpolate(
                        transported.permute(0, 4, 1, 2, 3),
                        size=current_x0.shape[2:],
                        mode="nearest",
                    ).to(current_x0.dtype)
                gate = torch.nn.functional.interpolate(
                    gate,
                    size=current_x0.shape[2:],
                    mode="nearest",
                ).to(current_x0.dtype)
                return transported, gate

            def _matched_draft_vector_residual(
                draft_prediction: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Transport the cached baseline solver vector along offline 3D correspondences."""
                p_t, p_h, p_w = patch_size
                batch_size, channels = draft_prediction.shape[:2]
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens

                draft_tokens = torch.nn.functional.avg_pool3d(
                    draft_prediction.float(),
                    kernel_size=(p_t, p_h, p_w),
                    stride=(p_t, p_h, p_w),
                )
                if tuple(draft_tokens.shape[2:]) != token_grid:
                    raise ValueError(
                        "Draft vector token grid does not match the geometry map: "
                        f"vector={list(draft_tokens.shape[2:])} map={token_grid}"
                    )
                draft_flat = draft_tokens.permute(0, 2, 3, 4, 1).reshape(
                    batch_size, num_tokens_t, spatial_tokens, channels
                )
                all_sources = draft_flat.reshape(
                    batch_size, num_tokens_t * spatial_tokens, channels
                )
                matched = torch.zeros_like(draft_flat[:, 1:])
                target_gate = torch.zeros(
                    (num_tokens_t - 1, spatial_tokens, 1),
                    device=draft_prediction.device,
                    dtype=torch.float32,
                )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = confidence.shape[-1]

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(
                        spatial_tokens, memory_slots
                    )
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue
                    source_flat = (
                        source_time[target_time - 1].reshape(spatial_tokens, memory_slots)
                        * spatial_tokens
                        + source_index[target_time - 1].reshape(spatial_tokens, memory_slots)
                    ).reshape(-1)
                    source_values = all_sources[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, channels
                    )
                    weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1)
                    matched[:, target_time - 1] = (
                        weights * source_values
                    ).sum(dim=2)
                    max_confidence = confidence_t.amax(dim=-1, keepdim=True)
                    if draft_vector_geometry_gate_mode == "binary_support":
                        max_confidence = (max_confidence > 0).to(max_confidence.dtype)
                    target_gate[target_time - 1] = max_confidence

                residual = torch.zeros_like(draft_flat)
                residual[:, 1:] = (
                    matched - draft_flat[:, 1:]
                ) * target_gate.reshape(
                    1, num_tokens_t - 1, spatial_tokens, 1
                )
                residual = residual.reshape(
                    batch_size, num_tokens_t, height_tokens, width_tokens, channels
                ).permute(0, 4, 1, 2, 3)
                gate = torch.cat(
                    [
                        torch.zeros_like(target_gate[:1]),
                        target_gate,
                    ],
                    dim=0,
                ).reshape(1, 1, num_tokens_t, height_tokens, width_tokens)
                residual = torch.nn.functional.interpolate(
                    residual,
                    size=draft_prediction.shape[2:],
                    mode="nearest",
                ).to(draft_prediction.dtype)
                gate = torch.nn.functional.interpolate(
                    gate,
                    size=draft_prediction.shape[2:],
                    mode="nearest",
                ).to(draft_prediction.dtype)
                return residual, gate

            def _geometry_support_gate(confidence: torch.Tensor) -> torch.Tensor:
                """Separate correspondence support from intervention strength."""
                if geometry_transport_gate_mode == "binary_support":
                    return (confidence > 0).to(confidence.dtype)
                return confidence

            def _preserve_conditioning_frame(reference: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
                if attn_avg_preserve_first_frame:
                    reference = reference.clone()
                    reference[:, :1] = original[:, :1]
                return reference

            def _temporal_window_mean(hidden: torch.Tensor) -> torch.Tensor:
                num_tokens_t = hidden.shape[1]
                accumulator = torch.zeros_like(hidden)
                count = torch.zeros((1, num_tokens_t, 1, 1, 1), device=hidden.device, dtype=hidden.dtype)
                for offset in range(-attn_avg_temporal_radius, attn_avg_temporal_radius + 1):
                    dst_lo = max(0, offset)
                    dst_hi = min(num_tokens_t, num_tokens_t + offset)
                    src_lo = max(0, -offset)
                    src_hi = min(num_tokens_t, num_tokens_t - offset)
                    accumulator[:, dst_lo:dst_hi] += hidden[:, src_lo:src_hi]
                    count[:, dst_lo:dst_hi] += 1
                return accumulator / count

            def _reduced_descriptors(hidden: torch.Tensor) -> torch.Tensor:
                channels = hidden.shape[-1]
                descriptor_dim = min(attn_avg_descriptor_dim, channels)
                group_size = channels // descriptor_dim
                usable_channels = descriptor_dim * group_size
                descriptor = hidden[..., :usable_channels].float().reshape(*hidden.shape[:-1], descriptor_dim, group_size)
                descriptor = descriptor.mean(dim=-1)
                return F.normalize(descriptor, dim=-1, eps=1e-6)

            def _match_previous_indices(input_hidden: torch.Tensor):
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                if num_tokens_t < 2:
                    return None, None, None, None

                descriptors = _reduced_descriptors(input_hidden)
                descriptor_dim = descriptors.shape[-1]
                kernel_size = 2 * attn_avg_match_radius + 1

                previous_descriptors = descriptors[:, :-1]
                current_descriptors = descriptors[:, 1:]
                previous_4d = previous_descriptors.permute(0, 1, 4, 2, 3).reshape(
                    batch * (num_tokens_t - 1), descriptor_dim, height_tokens, width_tokens
                )
                patches = F.unfold(previous_4d, kernel_size=kernel_size, padding=attn_avg_match_radius)
                patches = patches.reshape(
                    batch, num_tokens_t - 1, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                ).permute(0, 1, 4, 5, 3, 2)
                similarity = (current_descriptors.unsqueeze(-2) * patches).sum(dim=-1)
                best_similarity, best_index = similarity.max(dim=-1)

                offset_y = torch.div(best_index, kernel_size, rounding_mode="floor") - attn_avg_match_radius
                offset_x = best_index.remainder(kernel_size) - attn_avg_match_radius
                base_y = torch.arange(height_tokens, device=input_hidden.device).view(1, 1, height_tokens, 1)
                base_x = torch.arange(width_tokens, device=input_hidden.device).view(1, 1, 1, width_tokens)
                source_y = (base_y + offset_y).clamp(0, height_tokens - 1)
                source_x = (base_x + offset_x).clamp(0, width_tokens - 1)
                source_index = (source_y * width_tokens + source_x).reshape(batch, num_tokens_t - 1, -1)

                reciprocal = None
                if attn_avg_match_mutual:
                    # A static, unoccluded point should map back to its current token under the same local search.
                    # This rejects ambiguous repeated texture and many occlusion-boundary matches before K/V transport.
                    current_4d = current_descriptors.permute(0, 1, 4, 2, 3).reshape(
                        batch * (num_tokens_t - 1), descriptor_dim, height_tokens, width_tokens
                    )
                    reverse_patches = F.unfold(current_4d, kernel_size=kernel_size, padding=attn_avg_match_radius)
                    reverse_patches = reverse_patches.reshape(
                        batch, num_tokens_t - 1, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                    ).permute(0, 1, 4, 5, 3, 2)
                    reverse_similarity = (previous_descriptors.unsqueeze(-2) * reverse_patches).sum(dim=-1)
                    _, reverse_best_index = reverse_similarity.max(dim=-1)

                    reverse_offset_y = torch.div(reverse_best_index, kernel_size, rounding_mode="floor") - attn_avg_match_radius
                    reverse_offset_x = reverse_best_index.remainder(kernel_size) - attn_avg_match_radius
                    reverse_y = (base_y + reverse_offset_y).clamp(0, height_tokens - 1)
                    reverse_x = (base_x + reverse_offset_x).clamp(0, width_tokens - 1)
                    reverse_target_index = (reverse_y * width_tokens + reverse_x).reshape(batch, num_tokens_t - 1, -1)

                    returned_target = reverse_target_index.gather(2, source_index)
                    current_index = torch.arange(
                        height_tokens * width_tokens, device=input_hidden.device
                    ).view(1, 1, -1)
                    reciprocal = returned_target.eq(current_index).reshape(
                        batch, num_tokens_t - 1, height_tokens, width_tokens
                    )

                confidence = (
                    (best_similarity - attn_avg_match_confidence)
                    / (1.0 - attn_avg_match_confidence + 1e-6)
                ).clamp(0.0, 1.0).unsqueeze(-1)
                if reciprocal is not None:
                    confidence = confidence * reciprocal.unsqueeze(-1).to(confidence.dtype)
                return source_index, confidence, best_similarity, reciprocal

            def _c2f_match_previous_indices(input_hidden: torch.Tensor):
                """Globally match coarse tokens, then refine each match in a small fine-scale neighborhood."""
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                if num_tokens_t < 2:
                    return None, None, None, None

                factor = attn_avg_coarse_factor
                if height_tokens % factor or width_tokens % factor:
                    raise ValueError(
                        "c2f_match_prev requires token-grid dimensions divisible by attn_avg_coarse_factor; "
                        f"got {(height_tokens, width_tokens)} and factor={factor}"
                    )

                descriptors = _reduced_descriptors(input_hidden)
                descriptor_dim = descriptors.shape[-1]
                height_coarse, width_coarse = height_tokens // factor, width_tokens // factor
                coarse_tokens = height_coarse * width_coarse
                frame_pairs = batch * (num_tokens_t - 1)

                descriptor_4d = descriptors.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_tokens_t, descriptor_dim, height_tokens, width_tokens
                )
                coarse = F.avg_pool2d(descriptor_4d, kernel_size=factor, stride=factor)
                coarse = coarse.reshape(batch, num_tokens_t, descriptor_dim, height_coarse, width_coarse)
                coarse = F.normalize(coarse.permute(0, 1, 3, 4, 2), dim=-1, eps=1e-6)

                previous_coarse = coarse[:, :-1].reshape(frame_pairs, coarse_tokens, descriptor_dim)
                current_coarse = coarse[:, 1:].reshape(frame_pairs, coarse_tokens, descriptor_dim)
                coarse_similarity = torch.bmm(current_coarse, previous_coarse.transpose(1, 2))
                _, source_coarse_index = coarse_similarity.max(dim=-1)

                reciprocal = None
                if attn_avg_match_mutual:
                    reverse_best_index = coarse_similarity.argmax(dim=1)
                    returned_target = reverse_best_index.gather(1, source_coarse_index)
                    current_index = torch.arange(coarse_tokens, device=input_hidden.device).view(1, -1)
                    reciprocal = returned_target.eq(current_index).reshape(
                        batch, num_tokens_t - 1, height_coarse, width_coarse
                    )

                source_coarse_y = torch.div(source_coarse_index, width_coarse, rounding_mode="floor").reshape(
                    batch, num_tokens_t - 1, height_coarse, width_coarse
                )
                source_coarse_x = source_coarse_index.remainder(width_coarse).reshape(
                    batch, num_tokens_t - 1, height_coarse, width_coarse
                )
                source_base_y = source_coarse_y.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                source_base_x = source_coarse_x.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                source_base_y = source_base_y * factor + factor // 2
                source_base_x = source_base_x * factor + factor // 2

                radius = attn_avg_match_radius
                offsets_y, offsets_x = torch.meshgrid(
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    indexing="ij",
                )
                offsets_y = offsets_y.reshape(1, 1, 1, 1, -1)
                offsets_x = offsets_x.reshape(1, 1, 1, 1, -1)
                candidate_y = (source_base_y.unsqueeze(-1) + offsets_y).clamp(0, height_tokens - 1)
                candidate_x = (source_base_x.unsqueeze(-1) + offsets_x).clamp(0, width_tokens - 1)
                candidate_index = (candidate_y * width_tokens + candidate_x).reshape(
                    batch, num_tokens_t - 1, height_tokens * width_tokens, -1
                )

                spatial_tokens = height_tokens * width_tokens
                candidates_per_token = candidate_index.shape[-1]
                previous_fine = descriptors[:, :-1].reshape(frame_pairs, spatial_tokens, descriptor_dim)
                current_fine = descriptors[:, 1:].reshape(
                    batch, num_tokens_t - 1, spatial_tokens, descriptor_dim
                )
                gathered_previous = previous_fine.gather(
                    1,
                    candidate_index.reshape(frame_pairs, spatial_tokens * candidates_per_token)
                    .unsqueeze(-1)
                    .expand(-1, -1, descriptor_dim),
                ).reshape(batch, num_tokens_t - 1, spatial_tokens, candidates_per_token, descriptor_dim)
                fine_similarity = (current_fine.unsqueeze(-2) * gathered_previous).sum(dim=-1)
                fine_best_similarity, fine_best_index = fine_similarity.max(dim=-1)
                source_index = candidate_index.gather(-1, fine_best_index.unsqueeze(-1)).squeeze(-1)

                confidence = (
                    (fine_best_similarity - attn_avg_match_confidence)
                    / (1.0 - attn_avg_match_confidence + 1e-6)
                ).clamp(0.0, 1.0).reshape(batch, num_tokens_t - 1, height_tokens, width_tokens, 1)
                if reciprocal is not None:
                    reciprocal_fine = reciprocal.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                    confidence = confidence * reciprocal_fine.unsqueeze(-1).to(confidence.dtype)
                return source_index, confidence, fine_best_similarity.reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                ), reciprocal

            def _c2f_match_anchor_indices(input_hidden: torch.Tensor):
                """Match every later frame to the first conditioned-frame token grid.

                This is deliberately a read-only canonical anchor: unlike previous-frame
                propagation, it cannot accumulate an incorrect match from one frame to the next.
                Mutual coarse matching gates tokens that are no longer visible after camera motion.
                """
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                if num_tokens_t < 2:
                    return None, None, None, None

                factor = attn_avg_coarse_factor
                if height_tokens % factor or width_tokens % factor:
                    raise ValueError(
                        "c2f_value_residual_anchor requires token-grid dimensions divisible by "
                        f"attn_avg_coarse_factor; got {(height_tokens, width_tokens)} and factor={factor}"
                    )

                descriptors = _reduced_descriptors(input_hidden)
                descriptor_dim = descriptors.shape[-1]
                height_coarse, width_coarse = height_tokens // factor, width_tokens // factor
                coarse_tokens = height_coarse * width_coarse
                frame_pairs = batch * (num_tokens_t - 1)

                descriptor_4d = descriptors.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_tokens_t, descriptor_dim, height_tokens, width_tokens
                )
                coarse = F.avg_pool2d(descriptor_4d, kernel_size=factor, stride=factor)
                coarse = coarse.reshape(batch, num_tokens_t, descriptor_dim, height_coarse, width_coarse)
                coarse = F.normalize(coarse.permute(0, 1, 3, 4, 2), dim=-1, eps=1e-6)

                anchor_coarse = coarse[:, :1].expand(-1, num_tokens_t - 1, -1, -1, -1).reshape(
                    frame_pairs, coarse_tokens, descriptor_dim
                )
                current_coarse = coarse[:, 1:].reshape(frame_pairs, coarse_tokens, descriptor_dim)
                coarse_similarity = torch.bmm(current_coarse, anchor_coarse.transpose(1, 2))
                _, source_coarse_index = coarse_similarity.max(dim=-1)

                reciprocal = None
                if attn_avg_match_mutual:
                    reverse_best_index = coarse_similarity.argmax(dim=1)
                    returned_target = reverse_best_index.gather(1, source_coarse_index)
                    current_index = torch.arange(coarse_tokens, device=input_hidden.device).view(1, -1)
                    reciprocal = returned_target.eq(current_index).reshape(
                        batch, num_tokens_t - 1, height_coarse, width_coarse
                    )

                source_coarse_y = torch.div(source_coarse_index, width_coarse, rounding_mode="floor").reshape(
                    batch, num_tokens_t - 1, height_coarse, width_coarse
                )
                source_coarse_x = source_coarse_index.remainder(width_coarse).reshape(
                    batch, num_tokens_t - 1, height_coarse, width_coarse
                )
                source_base_y = source_coarse_y.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                source_base_x = source_coarse_x.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                source_base_y = source_base_y * factor + factor // 2
                source_base_x = source_base_x * factor + factor // 2

                radius = attn_avg_match_radius
                offsets_y, offsets_x = torch.meshgrid(
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    indexing="ij",
                )
                offsets_y = offsets_y.reshape(1, 1, 1, 1, -1)
                offsets_x = offsets_x.reshape(1, 1, 1, 1, -1)
                candidate_y = (source_base_y.unsqueeze(-1) + offsets_y).clamp(0, height_tokens - 1)
                candidate_x = (source_base_x.unsqueeze(-1) + offsets_x).clamp(0, width_tokens - 1)
                candidate_index = (candidate_y * width_tokens + candidate_x).reshape(
                    batch, num_tokens_t - 1, height_tokens * width_tokens, -1
                )

                spatial_tokens = height_tokens * width_tokens
                candidates_per_token = candidate_index.shape[-1]
                anchor_fine = descriptors[:, :1].expand(-1, num_tokens_t - 1, -1, -1, -1).reshape(
                    frame_pairs, spatial_tokens, descriptor_dim
                )
                current_fine = descriptors[:, 1:].reshape(
                    batch, num_tokens_t - 1, spatial_tokens, descriptor_dim
                )
                gathered_anchor = anchor_fine.gather(
                    1,
                    candidate_index.reshape(frame_pairs, spatial_tokens * candidates_per_token)
                    .unsqueeze(-1)
                    .expand(-1, -1, descriptor_dim),
                ).reshape(batch, num_tokens_t - 1, spatial_tokens, candidates_per_token, descriptor_dim)
                fine_similarity = (current_fine.unsqueeze(-2) * gathered_anchor).sum(dim=-1)
                fine_best_similarity, fine_best_index = fine_similarity.max(dim=-1)
                source_index = candidate_index.gather(-1, fine_best_index.unsqueeze(-1)).squeeze(-1)

                confidence = (
                    (fine_best_similarity - attn_avg_match_confidence)
                    / (1.0 - attn_avg_match_confidence + 1e-6)
                ).clamp(0.0, 1.0).reshape(batch, num_tokens_t - 1, height_tokens, width_tokens, 1)
                if reciprocal is not None:
                    reciprocal_fine = reciprocal.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                    confidence = confidence * reciprocal_fine.unsqueeze(-1).to(confidence.dtype)
                return source_index, confidence, fine_best_similarity.reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                ), reciprocal

            def _c2f_match_memory_indices(input_hidden: torch.Tensor):
                """Retrieve each token from the most reliable recent visible frame.

                The candidate bank is causal and bounded: a current token may use a
                source from one of the preceding `attn_avg_memory_lookback` frames.
                This avoids forcing a first-frame value into regions that have left
                the camera view, while mutual C2F matching gates ambiguous matches.
                """
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                if num_tokens_t < 2:
                    return None, None, None, None, None

                factor = attn_avg_coarse_factor
                if height_tokens % factor or width_tokens % factor:
                    raise ValueError(
                        "c2f_value_residual_memory requires token-grid dimensions divisible by "
                        f"attn_avg_coarse_factor; got {(height_tokens, width_tokens)} and factor={factor}"
                    )

                descriptors = _reduced_descriptors(input_hidden)
                descriptor_dim = descriptors.shape[-1]
                height_coarse, width_coarse = height_tokens // factor, width_tokens // factor
                coarse_tokens = height_coarse * width_coarse
                spatial_tokens = height_tokens * width_tokens

                descriptor_4d = descriptors.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_tokens_t, descriptor_dim, height_tokens, width_tokens
                )
                coarse = F.avg_pool2d(descriptor_4d, kernel_size=factor, stride=factor)
                coarse = coarse.reshape(batch, num_tokens_t, descriptor_dim, height_coarse, width_coarse)
                coarse = F.normalize(coarse.permute(0, 1, 3, 4, 2), dim=-1, eps=1e-6)

                best_confidence = torch.zeros(
                    batch, num_tokens_t - 1, spatial_tokens, device=input_hidden.device, dtype=torch.float32
                )
                best_similarity = torch.full_like(best_confidence, -1.0)
                best_source_index = torch.zeros(
                    batch, num_tokens_t - 1, spatial_tokens, device=input_hidden.device, dtype=torch.long
                )
                best_source_time = torch.zeros_like(best_source_index)
                best_reciprocal = torch.zeros_like(best_source_index, dtype=torch.bool)

                radius = attn_avg_match_radius
                offsets_y, offsets_x = torch.meshgrid(
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    torch.arange(-radius, radius + 1, device=input_hidden.device),
                    indexing="ij",
                )
                offsets_y = offsets_y.reshape(1, 1, 1, 1, -1)
                offsets_x = offsets_x.reshape(1, 1, 1, 1, -1)
                max_lag = min(attn_avg_memory_lookback, num_tokens_t - 1)

                for lag in range(1, max_lag + 1):
                    frame_pairs = num_tokens_t - lag
                    reference_coarse = coarse[:, :-lag].reshape(frame_pairs * batch, coarse_tokens, descriptor_dim)
                    current_coarse = coarse[:, lag:].reshape(frame_pairs * batch, coarse_tokens, descriptor_dim)
                    coarse_similarity = torch.bmm(current_coarse, reference_coarse.transpose(1, 2))
                    _, source_coarse_index = coarse_similarity.max(dim=-1)

                    reciprocal_coarse = None
                    if attn_avg_match_mutual:
                        reverse_best_index = coarse_similarity.argmax(dim=1)
                        returned_target = reverse_best_index.gather(1, source_coarse_index)
                        current_index = torch.arange(coarse_tokens, device=input_hidden.device).view(1, -1)
                        reciprocal_coarse = returned_target.eq(current_index).reshape(
                            batch, frame_pairs, height_coarse, width_coarse
                        )

                    source_coarse_y = torch.div(source_coarse_index, width_coarse, rounding_mode="floor").reshape(
                        batch, frame_pairs, height_coarse, width_coarse
                    )
                    source_coarse_x = source_coarse_index.remainder(width_coarse).reshape(
                        batch, frame_pairs, height_coarse, width_coarse
                    )
                    source_base_y = source_coarse_y.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                    source_base_x = source_coarse_x.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                    source_base_y = source_base_y * factor + factor // 2
                    source_base_x = source_base_x * factor + factor // 2
                    candidate_y = (source_base_y.unsqueeze(-1) + offsets_y).clamp(0, height_tokens - 1)
                    candidate_x = (source_base_x.unsqueeze(-1) + offsets_x).clamp(0, width_tokens - 1)
                    candidate_index = (candidate_y * width_tokens + candidate_x).reshape(
                        batch, frame_pairs, spatial_tokens, -1
                    )

                    candidates_per_token = candidate_index.shape[-1]
                    reference_fine = descriptors[:, :-lag].reshape(frame_pairs * batch, spatial_tokens, descriptor_dim)
                    current_fine = descriptors[:, lag:].reshape(
                        batch, frame_pairs, spatial_tokens, descriptor_dim
                    )
                    gathered_reference = reference_fine.gather(
                        1,
                        candidate_index.reshape(frame_pairs * batch, spatial_tokens * candidates_per_token)
                        .unsqueeze(-1)
                        .expand(-1, -1, descriptor_dim),
                    ).reshape(batch, frame_pairs, spatial_tokens, candidates_per_token, descriptor_dim)
                    fine_similarity = (current_fine.unsqueeze(-2) * gathered_reference).sum(dim=-1)
                    fine_best_similarity, fine_best_index = fine_similarity.max(dim=-1)
                    source_index = candidate_index.gather(-1, fine_best_index.unsqueeze(-1)).squeeze(-1)
                    confidence = (
                        (fine_best_similarity - attn_avg_match_confidence)
                        / (1.0 - attn_avg_match_confidence + 1e-6)
                    ).clamp(0.0, 1.0)

                    reciprocal_fine = None
                    if reciprocal_coarse is not None:
                        reciprocal_fine = reciprocal_coarse.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
                        reciprocal_fine = reciprocal_fine.reshape(batch, frame_pairs, spatial_tokens)
                        confidence = confidence * reciprocal_fine.to(confidence.dtype)

                    target_slots = torch.arange(lag - 1, num_tokens_t - 1, device=input_hidden.device)
                    previous_confidence = best_confidence[:, target_slots]
                    replace = confidence > previous_confidence
                    source_time = torch.arange(frame_pairs, device=input_hidden.device).view(1, frame_pairs, 1)
                    source_time = source_time.expand(batch, -1, spatial_tokens)
                    best_confidence[:, target_slots] = torch.where(replace, confidence, previous_confidence)
                    best_similarity[:, target_slots] = torch.where(
                        replace, fine_best_similarity, best_similarity[:, target_slots]
                    )
                    best_source_index[:, target_slots] = torch.where(
                        replace, source_index, best_source_index[:, target_slots]
                    )
                    best_source_time[:, target_slots] = torch.where(
                        replace, source_time, best_source_time[:, target_slots]
                    )
                    if reciprocal_fine is not None:
                        best_reciprocal[:, target_slots] = torch.where(
                            replace, reciprocal_fine, best_reciprocal[:, target_slots]
                        )

                reciprocal = best_reciprocal.reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                ) if attn_avg_match_mutual else None
                return (
                    best_source_index,
                    best_source_time,
                    best_confidence.reshape(batch, num_tokens_t - 1, height_tokens, width_tokens, 1),
                    best_similarity.reshape(batch, num_tokens_t - 1, height_tokens, width_tokens),
                    reciprocal,
                )

            def _match_previous_frame(hidden: torch.Tensor, input_hidden: torch.Tensor):
                batch, num_tokens_t, height_tokens, width_tokens, channels = hidden.shape
                if attn_avg_mode == "c2f_match_prev":
                    source_index, confidence, best_similarity, _ = _c2f_match_previous_indices(input_hidden)
                else:
                    source_index, confidence, best_similarity, _ = _match_previous_indices(input_hidden)
                if source_index is None:
                    return hidden, None

                previous_output = hidden[:, :-1].reshape(batch, num_tokens_t - 1, height_tokens * width_tokens, channels)
                matched_output = previous_output.gather(
                    2, source_index.unsqueeze(-1).expand(-1, -1, -1, channels)
                ).reshape(batch, num_tokens_t - 1, height_tokens, width_tokens, channels)

                mixed = hidden.clone()
                blend = attn_avg_alpha * confidence
                mixed[:, 1:] = (
                    hidden[:, 1:].float() * (1.0 - blend) + matched_output.float() * blend
                ).to(hidden.dtype)
                return _preserve_conditioning_frame(mixed, hidden), (best_similarity, blend)

            def _transport_previous_qkv(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                input_hidden: torch.Tensor,
                transport_query: bool,
                transport_key: bool,
                transport_value: bool,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                batch_size, sequence_length, query_channels = query.shape
                key_channels = key.shape[-1]
                value_channels = value.shape[-1]
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                expected_sequence_length = num_tokens_t * spatial_tokens
                if sequence_length != expected_sequence_length:
                    raise ValueError(
                        f"K/V sequence length {sequence_length} does not match token grid {token_grid} = "
                        f"{expected_sequence_length}."
                    )

                input_grid = input_hidden.reshape(batch_size, num_tokens_t, height_tokens, width_tokens, -1)
                source_index, confidence, best_similarity, reciprocal = _match_previous_indices(input_grid)
                if source_index is None:
                    return query, key, value, None, None

                query_grid = query.reshape(batch_size, num_tokens_t, spatial_tokens, query_channels)
                key_grid = key.reshape(batch_size, num_tokens_t, spatial_tokens, key_channels)
                value_grid = value.reshape(batch_size, num_tokens_t, spatial_tokens, value_channels)

                # Mix unrotated Q/K/V, then apply the current token's RoPE below.
                # The transported feature moves to the current-frame coordinate rather than its old image location.
                blend = (attn_avg_alpha * confidence).reshape(batch_size, num_tokens_t - 1, spatial_tokens, 1)
                transported_query = query_grid.clone()
                transported_key = key_grid.clone()
                transported_value = value_grid.clone()

                if transport_query:
                    matched_query = query_grid[:, :-1].gather(
                        2, source_index.unsqueeze(-1).expand(-1, -1, -1, query_channels)
                    )
                    transported_query[:, 1:] = (
                        query_grid[:, 1:].float() * (1.0 - blend) + matched_query.float() * blend
                    ).to(query.dtype)
                if transport_key:
                    matched_key = key_grid[:, :-1].gather(
                        2, source_index.unsqueeze(-1).expand(-1, -1, -1, key_channels)
                    )
                    transported_key[:, 1:] = (
                        key_grid[:, 1:].float() * (1.0 - blend) + matched_key.float() * blend
                    ).to(key.dtype)
                if transport_value:
                    matched_value = value_grid[:, :-1].gather(
                        2, source_index.unsqueeze(-1).expand(-1, -1, -1, value_channels)
                    )
                    transported_value[:, 1:] = (
                        value_grid[:, 1:].float() * (1.0 - blend) + matched_value.float() * blend
                    ).to(value.dtype)
                return (
                    transported_query.reshape_as(query),
                    transported_key.reshape_as(key),
                    transported_value.reshape_as(value),
                    best_similarity,
                    reciprocal,
                )

            def _matched_previous_values(
                value: torch.Tensor,
                input_hidden: torch.Tensor,
            ) -> tuple[
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]:
                """Return matched reference values without changing Q/K or the full attention map."""
                batch_size, sequence_length, value_channels = value.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                expected_sequence_length = num_tokens_t * spatial_tokens
                if sequence_length != expected_sequence_length:
                    raise ValueError(
                        f"Value sequence length {sequence_length} does not match token grid {token_grid} = "
                        f"{expected_sequence_length}."
                    )

                input_grid = input_hidden.reshape(batch_size, num_tokens_t, height_tokens, width_tokens, -1)
                if attn_avg_mode == "c2f_value_residual_anchor":
                    source_index, confidence, best_similarity, reciprocal = _c2f_match_anchor_indices(input_grid)
                    source_time = None
                elif attn_avg_mode == "c2f_value_residual_memory":
                    source_index, source_time, confidence, best_similarity, reciprocal = _c2f_match_memory_indices(input_grid)
                elif attn_avg_mode == "c2f_value_residual_prev":
                    source_index, confidence, best_similarity, reciprocal = _c2f_match_previous_indices(input_grid)
                    source_time = None
                else:
                    source_index, confidence, best_similarity, reciprocal = _match_previous_indices(input_grid)
                    source_time = None
                if source_index is None:
                    return None, None, None, None, None, None

                value_grid = value.reshape(batch_size, num_tokens_t, spatial_tokens, value_channels)
                if attn_avg_mode == "c2f_value_residual_memory":
                    source_flat_index = source_time * spatial_tokens + source_index
                    matched_values = value_grid.reshape(batch_size, num_tokens_t * spatial_tokens, value_channels).gather(
                        1,
                        source_flat_index.reshape(batch_size, -1)
                        .unsqueeze(-1)
                        .expand(-1, -1, value_channels),
                    ).reshape(batch_size, num_tokens_t - 1, spatial_tokens, value_channels)
                else:
                    reference_values = (
                        value_grid[:, :1].expand(-1, num_tokens_t - 1, -1, -1)
                        if attn_avg_mode == "c2f_value_residual_anchor"
                        else value_grid[:, :-1]
                    )
                    matched_values = reference_values.gather(
                        2, source_index.unsqueeze(-1).expand(-1, -1, -1, value_channels)
                    )
                return matched_values, confidence, best_similarity, reciprocal, source_index, source_time

            def _matched_geometry_values(
                value: torch.Tensor,
            ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                """Gather one z-buffered, geometry-validated previous value per target token."""
                if not geometry_transport_state["ready"]:
                    return None, None, None, None
                batch_size, sequence_length, value_channels = value.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Geometry transport received an unexpected Wan token sequence length")
                source_time = geometry_transport_state["source_time"][..., 0]
                source_index = geometry_transport_state["source_index"][..., 0]
                confidence = geometry_transport_state["confidence"][..., 0].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                gate_confidence = _geometry_support_gate(confidence)
                source_flat = source_time * spatial_tokens + source_index
                values_flat = value.reshape(batch_size, num_tokens_t * spatial_tokens, value_channels)
                matched = values_flat.gather(
                    1,
                    source_flat.reshape(1, -1).expand(batch_size, -1).unsqueeze(-1).expand(-1, -1, value_channels),
                ).reshape(batch_size, num_tokens_t - 1, spatial_tokens, value_channels)
                return (
                    matched,
                    gate_confidence.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
                    source_index,
                    source_time,
                )

            def _apply_rotary_at_target(
                states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ) -> torch.Tensor:
                """Encode a source K at its *projected target* RoPE coordinate."""
                x1, x2 = states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                rotated = torch.empty_like(states)
                rotated[..., 0::2] = x1 * cos - x2 * sin
                rotated[..., 1::2] = x1 * sin + x2 * cos
                return rotated.type_as(states)

            def _apply_geometry_qk_transport(
                query: torch.Tensor,
                key: torch.Tensor,
                transport_key: bool,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Transport clean-anchor Q/K content to its reprojected target token.

                Q/K are mixed before RoPE. The transported content therefore uses
                the target token's spatiotemporal coordinate below instead of
                retaining the anchor token's image coordinate.
                """
                if not geometry_transport_state["ready"]:
                    return query, key
                batch_size, sequence_length, heads, head_dim = query.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Geometry Q/K transport received an unexpected Wan token sequence length")

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                query_grid = query.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                key_grid = key.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                query_flat = query_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
                key_flat = key_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
                transported_query = query_grid.clone()
                transported_key = key_grid.clone()
                effective_alpha = _current_geometry_alpha()

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(spatial_tokens, memory_slots)
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue

                    source_flat = (
                        source_time[target_time - 1] * spatial_tokens
                        + source_index[target_time - 1]
                    ).reshape(-1)
                    source_query = query_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, heads, head_dim
                    )
                    source_key = key_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, heads, head_dim
                    )
                    candidate_weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1, 1)
                    matched_query = (candidate_weights * source_query.float()).sum(dim=2)
                    matched_key = (candidate_weights * source_key.float()).sum(dim=2)

                    # Candidate confidence controls whether this token is altered;
                    # alpha controls the global intervention strength.
                    gate = (
                        effective_alpha * _geometry_support_gate(confidence_t.amax(dim=-1))
                    ).reshape(1, spatial_tokens, 1, 1)
                    transported_query[:, target_time] = (
                        query_grid[:, target_time].float() * (1.0 - gate)
                        + matched_query * gate
                    ).to(query.dtype)
                    if geometry_transport_preserve_qk_rms:
                        query_reference = query_grid[:, target_time].float()
                        query_mixed = transported_query[:, target_time].float()
                        query_reference_rms = query_reference.square().mean(
                            dim=-1, keepdim=True
                        ).sqrt()
                        query_mixed_rms = query_mixed.square().mean(
                            dim=-1, keepdim=True
                        ).sqrt()
                        query_rescaled = (
                            query_mixed
                            * query_reference_rms
                            / query_mixed_rms.clamp_min(1e-8)
                        )
                        transported_query[:, target_time] = torch.where(
                            gate > 0,
                            query_rescaled,
                            query_mixed,
                        ).to(query.dtype)
                    if transport_key:
                        transported_key[:, target_time] = (
                            key_grid[:, target_time].float() * (1.0 - gate)
                            + matched_key * gate
                        ).to(key.dtype)
                        if geometry_transport_preserve_qk_rms:
                            key_reference = key_grid[:, target_time].float()
                            key_mixed = transported_key[:, target_time].float()
                            key_reference_rms = key_reference.square().mean(
                                dim=-1, keepdim=True
                            ).sqrt()
                            key_mixed_rms = key_mixed.square().mean(
                                dim=-1, keepdim=True
                            ).sqrt()
                            key_rescaled = (
                                key_mixed
                                * key_reference_rms
                                / key_mixed_rms.clamp_min(1e-8)
                            )
                            transported_key[:, target_time] = torch.where(
                                gate > 0,
                                key_rescaled,
                                key_mixed,
                            ).to(key.dtype)

                    if geometry_transport_debug and (
                        "qk", target_time
                    ) not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(("qk", target_time))
                        print(
                            f"[geometry-transport] target_latent={target_time} "
                            f"mode={'qk_transport' if transport_key else 'query_transport'} "
                            f"coverage={(confidence_sum > 0).float().mean().item():.3f} "
                            f"mean_gate={gate.float().mean().item():.5f} "
                            f"max_gate={gate.float().max().item():.5f} "
                            f"alpha={effective_alpha:.5f}",
                            flush=True,
                        )

                return transported_query.reshape_as(query), transported_key.reshape_as(key)

            def _apply_geometry_memory(
                query_with_rope: torch.Tensor,
                key_before_rope: torch.Tensor,
                value_heads: torch.Tensor,
                attended: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
                memory_mode: str,
            ) -> torch.Tensor:
                """Geometry-gated memory over only validated source candidates.

                The native full self-attention is retained. This adds a small
                per-target cross-attention residual over up to K reprojected
                source tokens plus a learned-free null option, so new/occluded
                regions can decline memory rather than receiving copied content.
                ``kv_memory`` transports raw V features; ``attn_output_memory``
                transports the native post-attention context at the source token.
                ``geometry_attention_bias`` uses geometry only to nominate source
                candidates, then lets their Q/K compatibility compete against the
                target token's own self-attention score before transporting V.
                ``geometry_attention_output_bias`` uses the same conservative
                competition but transports the source token's native full-attention
                context rather than its raw V.
                """
                if not geometry_transport_state["ready"]:
                    return attended
                if rotary_emb is None:
                    raise ValueError("geometry memory requires Wan rotary embeddings")
                if memory_mode not in {
                    "kv_memory",
                    "attn_output_memory",
                    "geometry_attention_bias",
                    "geometry_attention_output_bias",
                }:
                    raise ValueError(f"Unexpected geometry memory mode: {memory_mode}")
                batch_size, sequence_length, heads, head_dim = query_with_rope.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Geometry K/V memory received an unexpected Wan token sequence length")

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                query_grid = query_with_rope.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                key_grid = key_before_rope.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                value_grid = value_heads.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                attended_grid = attended.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                native_attended_grid = attended_grid.clone()
                key_flat = key_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
                memory_grid = (
                    native_attended_grid
                    if memory_mode in {
                        "attn_output_memory",
                        "geometry_attention_output_bias",
                    }
                    else value_grid
                )
                memory_flat = memory_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
                freqs_cos, freqs_sin = rotary_emb

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1]
                    if not (confidence_t > 0).any():
                        continue
                    source_flat = source_time[target_time - 1] * spatial_tokens + source_index[target_time - 1]
                    source_flat = source_flat.reshape(-1)
                    source_key = key_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, heads, head_dim
                    )
                    source_memory = memory_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, heads, head_dim
                    )

                    target_sequence_indices = target_time * spatial_tokens + torch.arange(
                        spatial_tokens, device=attended.device
                    )
                    target_cos = freqs_cos[:, target_sequence_indices].unsqueeze(2)
                    target_sin = freqs_sin[:, target_sequence_indices].unsqueeze(2)
                    source_key = _apply_rotary_at_target(source_key, target_cos, target_sin)
                    query_target = query_grid[:, target_time].unsqueeze(2)
                    scores = (query_target.float() * source_key.float()).sum(dim=-1) / (head_dim**0.5)
                    scores = scores.permute(0, 1, 3, 2)
                    confidence_candidates = confidence_t.reshape(
                        1, spatial_tokens, 1, memory_slots
                    )
                    if geometry_transport_gate_mode == "binary_support":
                        confidence_candidates = (
                            confidence_candidates > 0
                        ).to(confidence_candidates.dtype)
                    if memory_mode in {
                        "geometry_attention_bias",
                        "geometry_attention_output_bias",
                    }:
                        confidence_candidates = confidence_candidates / confidence_candidates.sum(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(1e-8)
                    confidence_logits = torch.where(
                        confidence_candidates > 0,
                        confidence_candidates.clamp_min(1e-8).log(),
                        torch.full_like(confidence_candidates, float("-inf")),
                    )
                    scores = scores + confidence_logits
                    if memory_mode in {
                        "geometry_attention_bias",
                        "geometry_attention_output_bias",
                    }:
                        target_key = key_grid[:, target_time].unsqueeze(2)
                        target_key = _apply_rotary_at_target(
                            target_key,
                            target_cos,
                            target_sin,
                        ).squeeze(2)
                        self_score = (
                            query_grid[:, target_time].float() * target_key.float()
                        ).sum(dim=-1) / (head_dim**0.5)
                        null_logits = self_score.unsqueeze(-1)
                        scores = scores + geometry_transport_source_logit_bias
                    else:
                        null_logits = torch.full_like(
                            scores[..., :1],
                            geometry_transport_null_logit,
                        )
                    weights = torch.softmax(torch.cat([scores, null_logits], dim=-1), dim=-1)[..., :-1]
                    source_memory = source_memory.permute(0, 1, 3, 2, 4)
                    memory_value = (weights.unsqueeze(-1) * source_memory.float()).sum(dim=3)
                    memory_mass = weights.sum(dim=-1).unsqueeze(-1)
                    current_memory = memory_grid[:, target_time].float()
                    effective_alpha = _current_geometry_alpha()
                    transport_delta = effective_alpha * (
                        memory_value - memory_mass * current_memory
                    )
                    attended_grid[:, target_time] = (
                        native_attended_grid[:, target_time].float()
                        + transport_delta
                    ).to(attended.dtype)

                    debug_key = (
                        (memory_mode, attn_avg_state["step"], target_time)
                        if memory_mode in {
                            "geometry_attention_bias",
                            "geometry_attention_output_bias",
                        }
                        else target_time
                    )
                    if geometry_transport_debug and debug_key not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(debug_key)
                        memory_mass_no_channel = memory_mass.squeeze(-1)
                        support = (confidence_t.amax(dim=-1) > 0).reshape(-1)
                        support_mass = memory_mass_no_channel[:, support]
                        support_delta = transport_delta[:, support]
                        support_native = native_attended_grid[:, target_time].float()[:, support]
                        if support_mass.numel() > 0:
                            mass_quantiles = torch.quantile(
                                support_mass.float(),
                                torch.tensor(
                                    [0.5, 0.9],
                                    device=support_mass.device,
                                ),
                            )
                            mean_gate = support_mass.mean().item()
                            accepted = (support_mass > 0.5).float().mean().item()
                            max_gate = support_mass.max().item()
                            residual_ratio = (
                                support_delta.float().square().mean().sqrt()
                                / support_native.float().square().mean().sqrt().clamp_min(1e-8)
                            ).item()
                        else:
                            mass_quantiles = torch.zeros(2, device=weights.device)
                            mean_gate = 0.0
                            accepted = 0.0
                            max_gate = 0.0
                            residual_ratio = 0.0
                        print(
                            f"[geometry-transport] step={attn_avg_state['step']} "
                            f"target_latent={target_time} mode={memory_mode} "
                            f"coverage={(confidence_t.amax(dim=-1) > 0).float().mean().item():.3f} "
                            f"roi_memory_mass_mean={mean_gate:.3f} "
                            f"roi_memory_mass_p50={mass_quantiles[0].item():.3f} "
                            f"roi_memory_mass_p90={mass_quantiles[1].item():.3f} "
                            f"roi_memory_mass_max={max_gate:.3f} "
                            f"roi_memory_mass_gt_half={accepted:.3f} "
                            f"roi_residual_rms_ratio={residual_ratio:.5f} "
                            f"source_logit_bias={geometry_transport_source_logit_bias:.3f} "
                            f"alpha={effective_alpha:.5f}",
                            flush=True,
                        )
                return attended_grid.reshape_as(attended)

            def _apply_geometry_sparse_logit_bias(
                query_with_rope: torch.Tensor,
                key_with_rope: torch.Tensor,
                key_before_rope: torch.Tensor,
                value_heads: torch.Tensor,
                attended: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
                virtual_token: bool,
                cached_virtual_token: bool,
                cache_key: str | None,
                key_normalizer,
            ) -> torch.Tensor:
                """Promote mapped sources inside the native attention distribution.

                The normal full 3D attention output is kept as the reference. For
                geometry-supported target tokens only, this function recomputes the
                native attention log-normalizer in key chunks. The sparse-logit
                mode promotes the real source token at its native coordinate. The
                virtual-token mode instead re-encodes the source K at the target
                (t, y, x) RoPE coordinate, representing an explicit geometry-warped
                correspondence. Confidence is normalized over K candidates, so
                duplicate candidates cannot silently strengthen the intervention.
                """
                effective_alpha = _current_geometry_alpha()
                if (
                    not geometry_transport_state["ready"]
                    or effective_alpha <= 0.0
                    or geometry_transport_sparse_logit_boost <= 0.0
                ):
                    return attended

                batch_size, sequence_length, heads, head_dim = query_with_rope.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError(
                        "Geometry sparse logit bias received an unexpected Wan token sequence length"
                    )
                if (virtual_token or cached_virtual_token) and rotary_emb is None:
                    raise ValueError("Geometry virtual token bias requires Wan rotary embeddings")
                if cached_virtual_token and (
                    not draft_feature_cache_state["loaded"]
                    or cache_key not in draft_feature_cache_state["features"]
                ):
                    raise KeyError(f"Draft K/V cache has no entry for {cache_key}")

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                output = attended.clone()
                score_scale = head_dim**-0.5
                boost_multiplier = math.expm1(geometry_transport_sparse_logit_boost)
                key_chunk_size = 2048
                debug_native_mass = []
                debug_extra_mass = []
                debug_delta_sq = torch.zeros((), device=attended.device, dtype=torch.float32)
                debug_native_sq = torch.zeros((), device=attended.device, dtype=torch.float32)
                debug_values = 0
                active_targets = 0
                cached_key_grid = None
                cached_value_grid = None
                source_lookup = None
                if cached_virtual_token:
                    cached_entry = draft_feature_cache_state["features"][cache_key]
                    if not isinstance(cached_entry, dict) or set(cached_entry) != {
                        "key",
                        "value",
                    }:
                        raise ValueError(
                            f"Draft K/V cache entry {cache_key} must contain key and value tensors"
                        )
                    cached_key = cached_entry["key"].to(
                        device=attended.device,
                        dtype=key_before_rope.dtype,
                    )
                    cached_value = cached_entry["value"].to(
                        device=attended.device,
                        dtype=value_heads.dtype,
                    )
                    if cached_key.shape != cached_value.shape:
                        raise ValueError(
                            f"Draft K/V shapes differ for {cache_key}: "
                            f"key={tuple(cached_key.shape)} value={tuple(cached_value.shape)}"
                        )
                    expected_prefix = (
                        batch_size,
                        len(draft_feature_cache_state["anchor_token_times"]),
                        spatial_tokens,
                    )
                    if cached_key.shape[:3] != expected_prefix:
                        raise ValueError(
                            f"Draft K/V shape {tuple(cached_key.shape)} is incompatible with "
                            f"batch={batch_size}, anchors="
                            f"{draft_feature_cache_state['anchor_token_times']}, "
                            f"spatial_tokens={spatial_tokens}"
                        )
                    cached_channels = cached_key.shape[-1]
                    if cached_channels != heads * head_dim:
                        raise ValueError(
                            f"Draft K/V channels {cached_channels} do not match "
                            f"attention channels {heads * head_dim}"
                        )
                    cached_key = key_normalizer(
                        cached_key.reshape(batch_size, -1, cached_channels)
                    ).unflatten(2, (heads, head_dim))
                    cached_key_grid = cached_key.reshape(
                        batch_size,
                        len(draft_feature_cache_state["anchor_token_times"]),
                        spatial_tokens,
                        heads,
                        head_dim,
                    )
                    cached_value_grid = cached_value.unflatten(
                        -1, (heads, head_dim)
                    )
                    source_lookup = torch.full(
                        (num_tokens_t,),
                        -1,
                        dtype=torch.long,
                        device=attended.device,
                    )
                    for cache_index, token_time in enumerate(
                        draft_feature_cache_state["anchor_token_times"]
                    ):
                        source_lookup[token_time] = cache_index

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(
                        spatial_tokens, memory_slots
                    )
                    support = confidence_t.amax(dim=-1) > 0
                    if not support.any():
                        continue

                    target_spatial = support.nonzero(as_tuple=False).squeeze(-1)
                    target_flat = target_time * spatial_tokens + target_spatial
                    confidence_active = confidence_t[target_spatial]
                    if geometry_transport_gate_mode == "binary_support":
                        confidence_active = (confidence_active > 0).to(
                            confidence_active.dtype
                        )
                    confidence_active = confidence_active / confidence_active.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)

                    source_flat = (
                        source_time[target_time - 1, target_spatial] * spatial_tokens
                        + source_index[target_time - 1, target_spatial]
                    )
                    query_target = query_with_rope[:, target_flat].float()

                    # Recompute only the per-target native log-normalizer. Chunking
                    # avoids materializing the full N x N attention matrix.
                    log_normalizer = torch.full(
                        (batch_size, target_flat.numel(), heads),
                        float("-inf"),
                        device=query_target.device,
                        dtype=torch.float32,
                    )
                    for key_start in range(0, sequence_length, key_chunk_size):
                        key_stop = min(sequence_length, key_start + key_chunk_size)
                        key_chunk = key_with_rope[:, key_start:key_stop].float()
                        scores_chunk = torch.einsum(
                            "bphd,bkhd->bphk",
                            query_target,
                            key_chunk,
                        ) * score_scale
                        log_normalizer = torch.logaddexp(
                            log_normalizer,
                            torch.logsumexp(scores_chunk, dim=-1),
                        )

                    cached_source_value = None
                    if cached_virtual_token:
                        source_time_active = source_time[
                            target_time - 1, target_spatial
                        ]
                        source_cache_index = source_lookup[source_time_active]
                        invalid = (source_cache_index < 0) & (
                            confidence_active > 0
                        )
                        if invalid.any():
                            bad_times = sorted(
                                set(source_time_active[invalid].tolist())
                            )
                            raise ValueError(
                                f"Geometry map references uncached anchor times {bad_times}"
                            )
                        source_cache_flat = (
                            source_cache_index.clamp_min(0) * spatial_tokens
                            + source_index[target_time - 1, target_spatial]
                        )
                        source_key = cached_key_grid.reshape(
                            batch_size, -1, heads, head_dim
                        )[:, source_cache_flat.reshape(-1)].reshape(
                            batch_size,
                            target_flat.numel(),
                            memory_slots,
                            heads,
                            head_dim,
                        )
                        cached_source_value = cached_value_grid.reshape(
                            batch_size, -1, heads, head_dim
                        )[:, source_cache_flat.reshape(-1)].reshape(
                            batch_size,
                            target_flat.numel(),
                            memory_slots,
                            heads,
                            head_dim,
                        )
                        freqs_cos, freqs_sin = rotary_emb
                        target_cos = freqs_cos[:, target_flat].unsqueeze(2)
                        target_sin = freqs_sin[:, target_flat].unsqueeze(2)
                        source_key = _apply_rotary_at_target(
                            source_key,
                            target_cos,
                            target_sin,
                        )
                    elif virtual_token:
                        source_key = key_before_rope[:, source_flat.reshape(-1)].reshape(
                            batch_size,
                            target_flat.numel(),
                            memory_slots,
                            heads,
                            head_dim,
                        )
                        freqs_cos, freqs_sin = rotary_emb
                        target_cos = freqs_cos[:, target_flat].unsqueeze(2)
                        target_sin = freqs_sin[:, target_flat].unsqueeze(2)
                        source_key = _apply_rotary_at_target(
                            source_key,
                            target_cos,
                            target_sin,
                        )
                    else:
                        source_key = key_with_rope[:, source_flat.reshape(-1)].reshape(
                            batch_size,
                            target_flat.numel(),
                            memory_slots,
                            heads,
                            head_dim,
                        )
                    source_scores = torch.einsum(
                        "bphd,bpmhd->bphm",
                        query_target,
                        source_key.float(),
                    ) * score_scale
                    source_probability = torch.exp(
                        source_scores - log_normalizer.unsqueeze(-1)
                    )

                    confidence_view = confidence_active.reshape(
                        1, target_flat.numel(), 1, memory_slots
                    )
                    if virtual_token or cached_virtual_token:
                        confidence_log = torch.where(
                            confidence_view > 0,
                            confidence_view.clamp_min(1e-8).log(),
                            torch.full_like(confidence_view, float("-inf")),
                        )
                        extra_weight = torch.exp(
                            source_scores
                            + confidence_log
                            + geometry_transport_sparse_logit_boost
                            - log_normalizer.unsqueeze(-1)
                        )
                    else:
                        # For K=1 this is exactly an additive logit boost. For
                        # K>1, confidence distributes one total positive boost
                        # budget over candidates, avoiding candidate-count bias.
                        extra_weight = (
                            boost_multiplier
                            * confidence_view
                            * source_probability
                        )
                    extra_mass = extra_weight.sum(dim=-1)
                    if cached_source_value is not None:
                        source_value = cached_source_value.permute(0, 1, 3, 2, 4)
                    else:
                        source_value = value_heads[
                            :, source_flat.reshape(-1)
                        ].reshape(
                            batch_size,
                            target_flat.numel(),
                            memory_slots,
                            heads,
                            head_dim,
                        ).permute(0, 1, 3, 2, 4)
                    extra_value = (
                        extra_weight.unsqueeze(-1) * source_value.float()
                    ).sum(dim=3)

                    native_target = attended[:, target_flat].float()
                    biased_target = (
                        native_target + extra_value
                    ) / (1.0 + extra_mass.unsqueeze(-1))
                    transport_delta = effective_alpha * (
                        biased_target - native_target
                    )
                    output[:, target_flat] = (
                        native_target + transport_delta
                    ).to(attended.dtype)

                    if geometry_transport_debug:
                        valid_candidate = confidence_active > 0
                        native_mass = (
                            source_probability
                            * valid_candidate.reshape(
                                1, target_flat.numel(), 1, memory_slots
                            )
                        ).sum(dim=-1)
                        debug_native_mass.append(native_mass.detach().reshape(-1))
                        debug_extra_mass.append(extra_mass.detach().reshape(-1))
                        debug_delta_sq += transport_delta.float().square().sum()
                        debug_native_sq += native_target.float().square().sum()
                        debug_values += transport_delta.numel()
                        active_targets += target_flat.numel()

                debug_key = (
                    (
                        "geometry_virtual_token_bias"
                        if virtual_token
                        else "geometry_sparse_logit_bias"
                    ),
                    attn_avg_state["step"],
                )
                if cached_virtual_token:
                    debug_key = ("cached_virtual_token_bias", attn_avg_state["step"])
                if (
                    geometry_transport_debug
                    and debug_key not in geometry_transport_state["printed"]
                    and debug_extra_mass
                ):
                    geometry_transport_state["printed"].add(debug_key)
                    native_mass = torch.cat(debug_native_mass)
                    extra_mass = torch.cat(debug_extra_mass)
                    extra_quantiles = torch.quantile(
                        extra_mass,
                        torch.tensor([0.5, 0.9], device=extra_mass.device),
                    )
                    residual_ratio = (
                        (debug_delta_sq / max(debug_values, 1)).sqrt()
                        / (debug_native_sq / max(debug_values, 1)).sqrt().clamp_min(1e-8)
                    )
                    print(
                        f"[geometry-transport] step={attn_avg_state['step']} "
                        f"mode={debug_key[0]} "
                        f"active_targets={active_targets} "
                        f"native_candidate_mass_mean={native_mass.mean().item():.6f} "
                        f"extra_mass_mean={extra_mass.mean().item():.6f} "
                        f"extra_mass_p50={extra_quantiles[0].item():.6f} "
                        f"extra_mass_p90={extra_quantiles[1].item():.6f} "
                        f"roi_residual_rms_ratio={residual_ratio.item():.6f} "
                        f"logit_boost={geometry_transport_sparse_logit_boost:.3f} "
                        f"alpha={effective_alpha:.5f}",
                        flush=True,
                    )
                return output

            def _apply_geometry_surface_graph_logit_bias(
                query_with_rope: torch.Tensor,
                key_with_rope: torch.Tensor,
                value_heads: torch.Tensor,
                attended: torch.Tensor,
            ) -> torch.Tensor:
                """Strengthen live attention along geometry-preserving surface edges.

                The draft geometry map supplies only correspondence identities. A
                source-grid four-neighborhood is converted once into target-frame
                graph edges. During sampling, each active target query receives an
                additive logit boost toward its current-frame graph neighbors. This
                preserves local surface connectivity without copying cached draft
                features or early-frame appearance.
                """
                effective_alpha = _current_geometry_alpha()
                graph_neighbors = geometry_transport_state[
                    "surface_graph_neighbors"
                ]
                graph_confidence = geometry_transport_state[
                    "surface_graph_confidence"
                ]
                signed_graph = (
                    geometry_transport_mode
                    == "geometry_signed_surface_graph_logit_bias"
                )
                boundary_neighbors = geometry_transport_state[
                    "surface_boundary_neighbors"
                ]
                boundary_confidence = geometry_transport_state[
                    "surface_boundary_confidence"
                ]
                if (
                    not geometry_transport_state["ready"]
                    or graph_neighbors is None
                    or effective_alpha <= 0.0
                    or geometry_transport_sparse_logit_boost <= 0.0
                    or (
                        signed_graph
                        and (
                            boundary_neighbors is None
                            or geometry_transport_surface_boundary_logit_suppress <= 0.0
                        )
                    )
                ):
                    return attended

                batch_size, sequence_length, heads, head_dim = (
                    query_with_rope.shape
                )
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError(
                        "Geometry surface graph received an unexpected Wan token sequence length"
                    )
                if graph_neighbors.shape != (
                    num_tokens_t - 1,
                    spatial_tokens,
                    4,
                ):
                    raise ValueError(
                        "Geometry surface graph has an incompatible neighbor shape"
                    )

                output = attended.clone()
                score_scale = head_dim**-0.5
                positive_multiplier = math.expm1(
                    geometry_transport_sparse_logit_boost
                )
                negative_multiplier = math.expm1(
                    -geometry_transport_surface_boundary_logit_suppress
                )
                key_chunk_size = 2048
                debug_native_mass = []
                debug_extra_mass = []
                debug_positive_mass = []
                debug_removed_mass = []
                debug_normalizer = []
                debug_degree = []
                debug_delta_sq = torch.zeros(
                    (),
                    device=attended.device,
                    dtype=torch.float32,
                )
                debug_native_sq = torch.zeros(
                    (),
                    device=attended.device,
                    dtype=torch.float32,
                )
                debug_values = 0
                active_targets = 0
                directed_edges = 0

                for target_time in range(1, num_tokens_t):
                    neighbor_t = graph_neighbors[target_time - 1]
                    confidence_t = graph_confidence[target_time - 1].clone()
                    valid_edge = neighbor_t >= 0
                    boundary_t = None
                    boundary_confidence_t = None
                    valid_boundary = None
                    if signed_graph:
                        boundary_t = boundary_neighbors[target_time - 1]
                        boundary_confidence_t = boundary_confidence[
                            target_time - 1
                        ].clone()
                        valid_boundary = boundary_t >= 0
                    if geometry_transport_min_confidence > 0:
                        valid_edge &= (
                            confidence_t >= geometry_transport_min_confidence
                        )
                        if signed_graph:
                            valid_boundary &= (
                                boundary_confidence_t
                                >= geometry_transport_min_confidence
                            )
                    support = valid_edge.any(dim=-1)
                    if signed_graph:
                        support |= valid_boundary.any(dim=-1)
                    if not support.any():
                        continue

                    target_spatial = support.nonzero(
                        as_tuple=False
                    ).squeeze(-1)
                    target_flat = (
                        target_time * spatial_tokens + target_spatial
                    )
                    neighbor_active = neighbor_t[target_spatial]
                    valid_active = valid_edge[target_spatial]
                    confidence_active = confidence_t[target_spatial]
                    edge_multipliers = torch.full(
                        (4,),
                        positive_multiplier,
                        device=attended.device,
                        dtype=torch.float32,
                    )
                    if signed_graph:
                        neighbor_active = torch.cat(
                            [
                                neighbor_active,
                                boundary_t[target_spatial],
                            ],
                            dim=-1,
                        )
                        valid_active = torch.cat(
                            [
                                valid_active,
                                valid_boundary[target_spatial],
                            ],
                            dim=-1,
                        )
                        confidence_active = torch.cat(
                            [
                                confidence_active,
                                boundary_confidence_t[target_spatial],
                            ],
                            dim=-1,
                        )
                        edge_multipliers = torch.cat(
                            [
                                edge_multipliers,
                                torch.full(
                                    (4,),
                                    negative_multiplier,
                                    device=attended.device,
                                    dtype=torch.float32,
                                ),
                            ],
                        )
                    edge_slots = neighbor_active.shape[-1]
                    neighbor_flat = (
                        target_time * spatial_tokens
                        + neighbor_active.clamp_min(0)
                    )
                    query_target = query_with_rope[:, target_flat].float()

                    log_normalizer = torch.full(
                        (batch_size, target_flat.numel(), heads),
                        float("-inf"),
                        device=query_target.device,
                        dtype=torch.float32,
                    )
                    for key_start in range(
                        0,
                        sequence_length,
                        key_chunk_size,
                    ):
                        key_stop = min(
                            sequence_length,
                            key_start + key_chunk_size,
                        )
                        key_chunk = key_with_rope[
                            :,
                            key_start:key_stop,
                        ].float()
                        scores_chunk = torch.einsum(
                            "bphd,bkhd->bphk",
                            query_target,
                            key_chunk,
                        ) * score_scale
                        log_normalizer = torch.logaddexp(
                            log_normalizer,
                            torch.logsumexp(scores_chunk, dim=-1),
                        )

                    neighbor_key = key_with_rope[
                        :,
                        neighbor_flat.reshape(-1),
                    ].reshape(
                        batch_size,
                        target_flat.numel(),
                        edge_slots,
                        heads,
                        head_dim,
                    )
                    neighbor_scores = torch.einsum(
                        "bphd,bpmhd->bphm",
                        query_target,
                        neighbor_key.float(),
                    ) * score_scale
                    neighbor_probability = torch.exp(
                        neighbor_scores - log_normalizer.unsqueeze(-1)
                    )
                    edge_gate = valid_active.reshape(
                        1,
                        target_flat.numel(),
                        1,
                        edge_slots,
                    ).to(neighbor_probability.dtype)
                    if geometry_transport_gate_mode == "confidence":
                        edge_gate = edge_gate * confidence_active.reshape(
                            1,
                            target_flat.numel(),
                            1,
                            edge_slots,
                        )
                    extra_weight = (
                        edge_multipliers.reshape(1, 1, 1, edge_slots)
                        * edge_gate
                        * neighbor_probability
                    )
                    extra_mass = extra_weight.sum(dim=-1)
                    neighbor_value = value_heads[
                        :,
                        neighbor_flat.reshape(-1),
                    ].reshape(
                        batch_size,
                        target_flat.numel(),
                        edge_slots,
                        heads,
                        head_dim,
                    ).permute(0, 1, 3, 2, 4)
                    extra_value = (
                        extra_weight.unsqueeze(-1)
                        * neighbor_value.float()
                    ).sum(dim=3)

                    native_target = attended[:, target_flat].float()
                    reweight_normalizer = 1.0 + extra_mass
                    if (reweight_normalizer <= 0.05).any():
                        raise RuntimeError(
                            "Signed surface-graph attention reweighting produced "
                            "an unsafe normalizer <= 0.05; reduce boundary suppression"
                        )
                    biased_target = (
                        native_target + extra_value
                    ) / reweight_normalizer.unsqueeze(-1)
                    transport_delta = effective_alpha * (
                        biased_target - native_target
                    )
                    output[:, target_flat] = (
                        native_target + transport_delta
                    ).to(attended.dtype)

                    if geometry_transport_debug:
                        native_mass = (
                            neighbor_probability * edge_gate
                        ).sum(dim=-1)
                        debug_native_mass.append(
                            native_mass.detach().reshape(-1)
                        )
                        debug_extra_mass.append(
                            extra_mass.detach().reshape(-1)
                        )
                        positive_mass = (
                            extra_weight[..., :4].sum(dim=-1)
                        )
                        removed_mass = (
                            -extra_weight[..., 4:].sum(dim=-1)
                            if signed_graph
                            else torch.zeros_like(positive_mass)
                        )
                        debug_positive_mass.append(
                            positive_mass.detach().reshape(-1)
                        )
                        debug_removed_mass.append(
                            removed_mass.detach().reshape(-1)
                        )
                        debug_normalizer.append(
                            reweight_normalizer.detach().reshape(-1)
                        )
                        debug_degree.append(
                            valid_active.sum(dim=-1).detach().float()
                        )
                        debug_delta_sq += (
                            transport_delta.float().square().sum()
                        )
                        debug_native_sq += (
                            native_target.float().square().sum()
                        )
                        debug_values += transport_delta.numel()
                        active_targets += target_flat.numel()
                        directed_edges += valid_active.sum().item()

                debug_key = (
                    geometry_transport_mode,
                    attn_avg_state["step"],
                )
                if (
                    geometry_transport_debug
                    and debug_key not in geometry_transport_state["printed"]
                    and debug_extra_mass
                ):
                    geometry_transport_state["printed"].add(debug_key)
                    native_mass = torch.cat(debug_native_mass)
                    extra_mass = torch.cat(debug_extra_mass)
                    positive_mass = torch.cat(debug_positive_mass)
                    removed_mass = torch.cat(debug_removed_mass)
                    normalizer = torch.cat(debug_normalizer)
                    degree = torch.cat(debug_degree)
                    extra_quantiles = torch.quantile(
                        extra_mass,
                        torch.tensor(
                            [0.5, 0.9],
                            device=extra_mass.device,
                        ),
                    )
                    residual_ratio = (
                        (debug_delta_sq / max(debug_values, 1)).sqrt()
                        / (
                            debug_native_sq / max(debug_values, 1)
                        ).sqrt().clamp_min(1e-8)
                    )
                    print(
                        f"[draft-geometry] step={attn_avg_state['step']} "
                        f"mode={geometry_transport_mode} "
                        f"active_targets={active_targets} "
                        f"directed_edges={directed_edges} "
                        f"mean_degree={degree.mean().item():.3f} "
                        f"native_neighbor_mass_mean={native_mass.mean().item():.6f} "
                        f"extra_mass_mean={extra_mass.mean().item():.6f} "
                        f"extra_mass_p50={extra_quantiles[0].item():.6f} "
                        f"extra_mass_p90={extra_quantiles[1].item():.6f} "
                        f"positive_mass_mean={positive_mass.mean().item():.6f} "
                        f"removed_mass_mean={removed_mass.mean().item():.6f} "
                        f"normalizer_min={normalizer.min().item():.6f} "
                        f"roi_residual_rms_ratio={residual_ratio.item():.6f} "
                        f"logit_boost={geometry_transport_sparse_logit_boost:.3f} "
                        f"alpha={effective_alpha:.5f}",
                        flush=True,
                    )
                return output

            def _apply_geometry_relative_edge_output(
                attended: torch.Tensor,
            ) -> torch.Tensor:
                """Match source-anchor and target relative features on surface edges.

                For every undirected same-surface edge, this applies one normalized
                graph-Laplacian step to the target endpoints. The source edge
                difference is read from the immutable live attention output, so the
                update preserves relative structure without copying absolute source
                appearance.
                """
                effective_alpha = _current_geometry_alpha()
                edge_records = geometry_transport_state["surface_relative_edges"]
                if (
                    not geometry_transport_state["ready"]
                    or edge_records is None
                    or effective_alpha <= 0.0
                ):
                    return attended

                batch_size, sequence_length, heads, head_dim = attended.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError(
                        "Geometry relative-edge output received an unexpected "
                        "Wan token sequence length"
                    )
                if len(edge_records) != num_tokens_t - 1:
                    raise ValueError(
                        "Geometry relative-edge graph has an incompatible time dimension"
                    )

                native = attended.float()
                output = attended.clone()
                debug_edge_error_before_sq = torch.zeros(
                    (), device=attended.device, dtype=torch.float32
                )
                debug_edge_error_after_sq = torch.zeros_like(
                    debug_edge_error_before_sq
                )
                debug_delta_sq = torch.zeros_like(debug_edge_error_before_sq)
                debug_native_sq = torch.zeros_like(debug_edge_error_before_sq)
                debug_edge_values = 0
                debug_delta_values = 0
                debug_edges = 0
                debug_active_nodes = 0
                debug_max_degree = 0.0
                debug_cap_scale = 1.0

                for target_time in range(1, num_tokens_t):
                    record = edge_records[target_time - 1]
                    confidence = record["confidence"]
                    scale_weight = record["scale_weight"]
                    valid = confidence > 0
                    if geometry_transport_min_confidence > 0:
                        valid &= confidence >= geometry_transport_min_confidence
                    if not valid.any():
                        continue

                    target_i_local = record["target_i"][valid]
                    target_j_local = record["target_j"][valid]
                    source_i = record["source_i"][valid]
                    source_j = record["source_j"][valid]
                    confidence_weight = (
                        confidence[valid]
                        if geometry_transport_gate_mode == "confidence"
                        else torch.ones_like(confidence[valid])
                    )
                    edge_weight = scale_weight[valid] * confidence_weight
                    target_i = target_time * spatial_tokens + target_i_local
                    target_j = target_time * spatial_tokens + target_j_local

                    source_difference = (
                        native[:, source_j] - native[:, source_i]
                    )
                    target_difference = (
                        native[:, target_j] - native[:, target_i]
                    )
                    edge_error = source_difference - target_difference
                    weighted_error = edge_error * edge_weight.reshape(
                        1, -1, 1, 1
                    )

                    target_delta = torch.zeros(
                        (
                            batch_size,
                            spatial_tokens,
                            heads,
                            head_dim,
                        ),
                        device=attended.device,
                        dtype=torch.float32,
                    )
                    scatter_shape = (
                        batch_size,
                        target_i_local.numel(),
                        heads,
                        head_dim,
                    )
                    target_delta.scatter_add_(
                        1,
                        target_i_local.reshape(1, -1, 1, 1).expand(
                            scatter_shape
                        ),
                        -0.5 * weighted_error,
                    )
                    target_delta.scatter_add_(
                        1,
                        target_j_local.reshape(1, -1, 1, 1).expand(
                            scatter_shape
                        ),
                        0.5 * weighted_error,
                    )

                    weighted_degree = torch.zeros(
                        spatial_tokens,
                        device=attended.device,
                        dtype=torch.float32,
                    )
                    weighted_degree.scatter_add_(
                        0,
                        target_i_local,
                        edge_weight,
                    )
                    weighted_degree.scatter_add_(
                        0,
                        target_j_local,
                        edge_weight,
                    )
                    max_degree = weighted_degree.max().clamp_min(1.0)
                    target_delta = target_delta / max_degree
                    target_slice = slice(
                        target_time * spatial_tokens,
                        (target_time + 1) * spatial_tokens,
                    )
                    native_target = native[:, target_slice]
                    transport_delta = effective_alpha * target_delta
                    active_nodes = weighted_degree > 0
                    if active_nodes.any() and geometry_transport_max_relative_rms > 0:
                        delta_rms = transport_delta[:, active_nodes].square().mean().sqrt()
                        native_rms = native_target[:, active_nodes].square().mean().sqrt().clamp_min(1e-8)
                        relative_rms = delta_rms / native_rms
                        if relative_rms > geometry_transport_max_relative_rms:
                            cap_scale = (
                                geometry_transport_max_relative_rms
                                / relative_rms
                            )
                            transport_delta = transport_delta * cap_scale
                            debug_cap_scale = min(
                                debug_cap_scale,
                                float(cap_scale.item()),
                            )

                    output[:, target_slice] = (
                        native_target + transport_delta
                    ).to(attended.dtype)

                    if geometry_transport_debug:
                        updated_target = output[:, target_slice].float()
                        updated_difference = (
                            updated_target[:, target_j_local]
                            - updated_target[:, target_i_local]
                        )
                        updated_error = source_difference - updated_difference
                        debug_edge_error_before_sq += edge_error.square().sum()
                        debug_edge_error_after_sq += updated_error.square().sum()
                        debug_delta_sq += transport_delta[:, active_nodes].square().sum()
                        debug_native_sq += native_target[:, active_nodes].square().sum()
                        debug_edge_values += edge_error.numel()
                        debug_delta_values += transport_delta[:, active_nodes].numel()
                        debug_edges += target_i_local.numel()
                        debug_active_nodes += active_nodes.sum().item()
                        debug_max_degree = max(
                            debug_max_degree,
                            float(max_degree.item()),
                        )

                debug_key = (
                    "geometry_relative_edge_output",
                    attn_avg_state["step"],
                )
                if (
                    geometry_transport_debug
                    and debug_key not in geometry_transport_state["printed"]
                    and debug_edge_values > 0
                ):
                    geometry_transport_state["printed"].add(debug_key)
                    edge_error_before = (
                        debug_edge_error_before_sq
                        / debug_edge_values
                    ).sqrt()
                    edge_error_after = (
                        debug_edge_error_after_sq
                        / debug_edge_values
                    ).sqrt()
                    residual_ratio = (
                        (debug_delta_sq / max(debug_delta_values, 1)).sqrt()
                        / (
                            debug_native_sq
                            / max(debug_delta_values, 1)
                        ).sqrt().clamp_min(1e-8)
                    )
                    print(
                        f"[draft-geometry] step={attn_avg_state['step']} "
                        "mode=geometry_relative_edge_output "
                        f"undirected_edges={debug_edges} "
                        f"active_nodes={debug_active_nodes} "
                        f"max_weighted_degree={debug_max_degree:.4f} "
                        f"edge_error_before={edge_error_before.item():.6f} "
                        f"edge_error_after={edge_error_after.item():.6f} "
                        f"roi_residual_rms_ratio={residual_ratio.item():.6f} "
                        f"cap_scale={debug_cap_scale:.6f} "
                        f"alpha={effective_alpha:.5f}",
                        flush=True,
                    )
                return output

            def _apply_cached_value_memory(
                query_with_rope: torch.Tensor,
                key_before_rope: torch.Tensor,
                value_heads: torch.Tensor,
                attended: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
                cache_key: str,
                layer_idx: int,
            ) -> torch.Tensor:
                """Attend to geometry candidates and read frozen draft V features."""
                if not draft_feature_cache_state["loaded"]:
                    return attended
                if cache_key not in draft_feature_cache_state["features"]:
                    raise KeyError(f"Draft value cache has no entry for {cache_key}")
                if rotary_emb is None:
                    raise ValueError("Cached value memory requires Wan rotary embeddings")

                batch_size, sequence_length, heads, head_dim = query_with_rope.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                value_channels = heads * head_dim
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError(
                        "Cached value memory received an unexpected Wan token sequence length"
                    )

                cached = draft_feature_cache_state["features"][cache_key].to(
                    device=attended.device,
                    dtype=value_heads.dtype,
                )
                if cached.shape[0] == 1 and batch_size > 1:
                    cached = cached.expand(batch_size, -1, -1, -1)
                expected_prefix = (
                    batch_size,
                    len(draft_feature_cache_state["anchor_token_times"]),
                    spatial_tokens,
                )
                if cached.shape[:3] != expected_prefix or cached.shape[-1] != value_channels:
                    raise ValueError(
                        f"Cached attention value shape {tuple(cached.shape)} is incompatible with "
                        f"batch={batch_size}, anchors={draft_feature_cache_state['anchor_token_times']}, "
                        f"spatial_tokens={spatial_tokens}, channels={value_channels}"
                    )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                source_lookup = torch.full(
                    (num_tokens_t,),
                    -1,
                    dtype=torch.long,
                    device=attended.device,
                )
                for cache_index, token_time in enumerate(
                    draft_feature_cache_state["anchor_token_times"]
                ):
                    source_lookup[token_time] = cache_index

                query_grid = query_with_rope.reshape(
                    batch_size,
                    num_tokens_t,
                    spatial_tokens,
                    heads,
                    head_dim,
                )
                key_grid = key_before_rope.reshape(
                    batch_size,
                    num_tokens_t,
                    spatial_tokens,
                    heads,
                    head_dim,
                )
                value_grid = value_heads.reshape(
                    batch_size,
                    num_tokens_t,
                    spatial_tokens,
                    heads,
                    head_dim,
                )
                cached_memory_grid = cached
                current_memory_grid = value_grid
                if geometry_transport_value_lowpass_radius > 0:
                    kernel_size = 2 * geometry_transport_value_lowpass_radius + 1
                    cached_memory_grid = F.avg_pool2d(
                        cached.reshape(
                            batch_size * cached.shape[1],
                            height_tokens,
                            width_tokens,
                            value_channels,
                        ).permute(0, 3, 1, 2),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=geometry_transport_value_lowpass_radius,
                        count_include_pad=False,
                    ).permute(0, 2, 3, 1).reshape_as(cached)
                    current_memory_grid = F.avg_pool2d(
                        value_grid.reshape(
                            batch_size * num_tokens_t,
                            height_tokens,
                            width_tokens,
                            value_channels,
                        ).permute(0, 3, 1, 2),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=geometry_transport_value_lowpass_radius,
                        count_include_pad=False,
                    ).permute(0, 2, 3, 1).reshape_as(value_grid)
                attended_grid = attended.reshape_as(value_grid)
                native_attended_grid = attended_grid.clone()
                key_flat = key_grid.reshape(
                    batch_size,
                    num_tokens_t * spatial_tokens,
                    heads,
                    head_dim,
                )
                cached_flat = cached_memory_grid.reshape(
                    batch_size,
                    -1,
                    heads,
                    head_dim,
                )
                freqs_cos, freqs_sin = rotary_emb
                effective_alpha = _current_geometry_alpha()

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(
                        spatial_tokens,
                        memory_slots,
                    )
                    if not (confidence_t > 0).any():
                        continue
                    source_time_t = source_time[target_time - 1].reshape(
                        spatial_tokens,
                        memory_slots,
                    )
                    source_cache_index = source_lookup[source_time_t]
                    invalid = (source_cache_index < 0) & (confidence_t > 0)
                    if invalid.any():
                        bad_times = sorted(set(source_time_t[invalid].tolist()))
                        raise ValueError(
                            f"Geometry map references uncached anchor times {bad_times}"
                        )

                    source_spatial_index = source_index[target_time - 1].reshape(
                        spatial_tokens,
                        memory_slots,
                    )
                    source_current_flat = (
                        source_time_t * spatial_tokens + source_spatial_index
                    ).reshape(-1)
                    source_cache_flat = (
                        source_cache_index.clamp_min(0) * spatial_tokens
                        + source_spatial_index
                    ).reshape(-1)
                    source_key = key_flat[:, source_current_flat].reshape(
                        batch_size,
                        spatial_tokens,
                        memory_slots,
                        heads,
                        head_dim,
                    )
                    source_value = cached_flat[:, source_cache_flat].reshape(
                        batch_size,
                        spatial_tokens,
                        memory_slots,
                        heads,
                        head_dim,
                    )
                    consensus_value = None
                    consensus_gate = None
                    consensus_score = None
                    if geometry_transport_consensus_threshold >= 0.0:
                        valid_candidates = (confidence_t > 0).reshape(
                            1,
                            spatial_tokens,
                            memory_slots,
                            1,
                            1,
                        )
                        candidate_count = valid_candidates.sum(dim=2).clamp_min(1)
                        consensus_value = (
                            source_value.float()
                            * valid_candidates.to(source_value.dtype)
                        ).sum(dim=2) / candidate_count

                        normalized_candidates = F.normalize(
                            source_value.float().flatten(-2),
                            dim=-1,
                        )
                        pairwise_similarity = torch.einsum(
                            "bskc,bsjc->bskj",
                            normalized_candidates,
                            normalized_candidates,
                        )
                        valid_flat = valid_candidates[..., 0, 0]
                        pair_mask = (
                            valid_flat.unsqueeze(-1)
                            & valid_flat.unsqueeze(-2)
                        )
                        pair_mask = torch.triu(pair_mask, diagonal=1)
                        pair_count = pair_mask.sum(dim=(-2, -1))
                        consensus_score = (
                            pairwise_similarity
                            * pair_mask.to(pairwise_similarity.dtype)
                        ).sum(dim=(-2, -1)) / pair_count.clamp_min(1)
                        consensus_gate = (
                            (pair_count > 0)
                            & (
                                consensus_score
                                >= geometry_transport_consensus_threshold
                            )
                        ).reshape(batch_size, spatial_tokens, 1, 1)

                    target_sequence_indices = (
                        target_time * spatial_tokens
                        + torch.arange(spatial_tokens, device=attended.device)
                    )
                    target_cos = freqs_cos[:, target_sequence_indices].unsqueeze(2)
                    target_sin = freqs_sin[:, target_sequence_indices].unsqueeze(2)
                    source_key = _apply_rotary_at_target(
                        source_key,
                        target_cos,
                        target_sin,
                    )
                    query_target = query_grid[:, target_time].unsqueeze(2)
                    scores = (
                        query_target.float() * source_key.float()
                    ).sum(dim=-1) / (head_dim**0.5)
                    scores = scores.permute(0, 1, 3, 2)
                    confidence_candidates = confidence_t.reshape(
                        1,
                        spatial_tokens,
                        1,
                        memory_slots,
                    )
                    if geometry_transport_gate_mode == "binary_support":
                        confidence_candidates = (
                            confidence_candidates > 0
                        ).to(confidence_candidates.dtype)
                    confidence_logits = torch.where(
                        confidence_candidates > 0,
                        confidence_candidates.clamp_min(1e-8).log(),
                        torch.full_like(
                            confidence_candidates,
                            float("-inf"),
                        ),
                    )
                    scores = scores + confidence_logits
                    null_logits = torch.full_like(
                        scores[..., :1],
                        geometry_transport_null_logit,
                    )
                    weights = torch.softmax(
                        torch.cat([scores, null_logits], dim=-1),
                        dim=-1,
                    )[..., :-1]
                    source_value = source_value.permute(0, 1, 3, 2, 4)
                    memory_mass = weights.sum(dim=-1).unsqueeze(-1)
                    if consensus_value is None:
                        memory_value = (
                            weights.unsqueeze(-1) * source_value.float()
                        ).sum(dim=3)
                    else:
                        memory_value = consensus_value * memory_mass
                        memory_value = (
                            memory_value
                            * consensus_gate.to(memory_value.dtype)
                        )
                        memory_mass = (
                            memory_mass
                            * consensus_gate.to(memory_mass.dtype)
                        )
                    if geometry_transport_mode == "cached_value_attention_blend":
                        reference_value = native_attended_grid[:, target_time].float()
                    else:
                        reference_value = current_memory_grid[:, target_time].float()
                    attended_grid[:, target_time] = (
                        native_attended_grid[:, target_time].float()
                        + effective_alpha
                        * (memory_value - memory_mass * reference_value)
                    ).to(attended.dtype)

                    mode_label = (
                        "cached_value_attention_blend"
                        if geometry_transport_mode == "cached_value_attention_blend"
                        else "cached_value_memory"
                    )
                    print_key = (
                        mode_label,
                        layer_idx,
                        attn_avg_state["step"],
                        target_time,
                    )
                    if (
                        geometry_transport_debug
                        and print_key not in geometry_transport_state["printed"]
                    ):
                        geometry_transport_state["printed"].add(print_key)
                        active = confidence_t.amax(dim=-1) > 0
                        target_mass = memory_mass.float().mean(dim=2).squeeze(-1)
                        mean_mass = (
                            target_mass[:, active].mean().item()
                            if active.any()
                            else 0.0
                        )
                        consensus_text = ""
                        if consensus_gate is not None:
                            accepted = consensus_gate[..., 0, 0]
                            mean_consensus = (
                                consensus_score[accepted].mean().item()
                                if accepted.any()
                                else 0.0
                            )
                            consensus_text = (
                                f" consensus_accept={accepted.float().mean().item():.3f}"
                                f" consensus_mean={mean_consensus:.3f}"
                            )
                        print(
                            f"[geometry-transport] target_latent={target_time} "
                            f"mode={mode_label} layer={layer_idx} "
                            f"coverage={active.float().mean().item():.3f} "
                            f"memory_mass_active={mean_mass:.5f} "
                            f"alpha={effective_alpha:.5f}"
                            f" lowpass_radius={geometry_transport_value_lowpass_radius}"
                            f"{consensus_text}",
                            flush=True,
                        )

                return attended_grid.reshape_as(attended)

            def _apply_cached_block_transport(
                hidden_states: torch.Tensor,
                cache_key: str,
                layer_idx: int,
            ) -> torch.Tensor:
                """Blend target block features with frozen draft-anchor features."""
                if not draft_feature_cache_state["loaded"]:
                    return hidden_states
                if cache_key not in draft_feature_cache_state["features"]:
                    raise KeyError(f"Draft feature cache has no entry for {cache_key}")

                batch_size, sequence_length, channels = hidden_states.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Cached block transport received an unexpected Wan token sequence length")

                cached = draft_feature_cache_state["features"][cache_key].to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if cached.shape[0] == 1 and batch_size > 1:
                    cached = cached.expand(batch_size, -1, -1, -1)
                if cached.shape[:3] != (
                    batch_size,
                    len(draft_feature_cache_state["anchor_token_times"]),
                    spatial_tokens,
                ):
                    raise ValueError(
                        f"Cached block feature shape {tuple(cached.shape)} is incompatible with "
                        f"batch={batch_size}, anchors={draft_feature_cache_state['anchor_token_times']}, "
                        f"spatial_tokens={spatial_tokens}"
                    )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                source_lookup = torch.full(
                    (num_tokens_t,),
                    -1,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                for cache_index, token_time in enumerate(draft_feature_cache_state["anchor_token_times"]):
                    source_lookup[token_time] = cache_index

                hidden_grid = hidden_states.reshape(
                    batch_size, num_tokens_t, spatial_tokens, channels
                )
                mixed_grid = hidden_grid.clone()
                cached_flat = cached.reshape(batch_size, -1, channels)
                effective_alpha = _current_geometry_alpha()

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(spatial_tokens, memory_slots)
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue
                    source_time_t = source_time[target_time - 1].reshape(spatial_tokens, memory_slots)
                    source_cache_index = source_lookup[source_time_t]
                    invalid = (source_cache_index < 0) & (confidence_t > 0)
                    if invalid.any():
                        bad_times = sorted(set(source_time_t[invalid].tolist()))
                        raise ValueError(f"Geometry map references uncached anchor times {bad_times}")
                    source_flat = (
                        source_cache_index.clamp_min(0) * spatial_tokens
                        + source_index[target_time - 1].reshape(spatial_tokens, memory_slots)
                    ).reshape(-1)
                    source_feature = cached_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, channels
                    )
                    candidate_weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1)
                    matched_feature = (candidate_weights * source_feature.float()).sum(dim=2)
                    gate = (
                        effective_alpha * _geometry_support_gate(confidence_t.amax(dim=-1))
                    ).reshape(1, spatial_tokens, 1)
                    mixed_grid[:, target_time] = (
                        hidden_grid[:, target_time].float() * (1.0 - gate)
                        + matched_feature * gate
                    ).to(hidden_states.dtype)

                    print_key = ("cached-block", layer_idx, target_time)
                    if geometry_transport_debug and print_key not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(print_key)
                        print(
                            f"[geometry-transport] target_latent={target_time} "
                            f"mode=cached_block_transport layer={layer_idx} "
                            f"coverage={(confidence_sum > 0).float().mean().item():.3f} "
                            f"mean_gate={gate.float().mean().item():.5f} "
                            f"alpha={effective_alpha:.5f}",
                            flush=True,
                        )

                return mixed_grid.reshape_as(hidden_states)

            def _apply_cached_value_transport(
                value: torch.Tensor,
                cache_key: str,
                layer_idx: int,
            ) -> torch.Tensor:
                """Blend only attention V at geometry-matched target tokens.

                Current-run Q/K and therefore the attention lookup remain unchanged.
                The cached draft contributes identity/content at locations validated by
                the offline geometry map.
                """
                if not draft_feature_cache_state["loaded"]:
                    return value
                if cache_key not in draft_feature_cache_state["features"]:
                    raise KeyError(f"Draft value cache has no entry for {cache_key}")

                batch_size, sequence_length, value_channels = value.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Cached value transport received an unexpected Wan token sequence length")

                cached = draft_feature_cache_state["features"][cache_key].to(
                    device=value.device,
                    dtype=value.dtype,
                )
                if cached.shape[0] == 1 and batch_size > 1:
                    cached = cached.expand(batch_size, -1, -1, -1)
                expected_prefix = (
                    batch_size,
                    len(draft_feature_cache_state["anchor_token_times"]),
                    spatial_tokens,
                )
                if cached.shape[:3] != expected_prefix or cached.shape[-1] != value_channels:
                    raise ValueError(
                        f"Cached attention value shape {tuple(cached.shape)} is incompatible with "
                        f"batch={batch_size}, anchors={draft_feature_cache_state['anchor_token_times']}, "
                        f"spatial_tokens={spatial_tokens}, channels={value_channels}"
                    )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                source_lookup = torch.full(
                    (num_tokens_t,),
                    -1,
                    dtype=torch.long,
                    device=value.device,
                )
                for cache_index, token_time in enumerate(draft_feature_cache_state["anchor_token_times"]):
                    source_lookup[token_time] = cache_index

                value_grid = value.reshape(batch_size, num_tokens_t, spatial_tokens, value_channels)
                mixed_grid = value_grid.clone()
                cached_flat = cached.reshape(batch_size, -1, value_channels)
                effective_alpha = _current_geometry_alpha()

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(spatial_tokens, memory_slots)
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue
                    source_time_t = source_time[target_time - 1].reshape(spatial_tokens, memory_slots)
                    source_cache_index = source_lookup[source_time_t]
                    invalid = (source_cache_index < 0) & (confidence_t > 0)
                    if invalid.any():
                        bad_times = sorted(set(source_time_t[invalid].tolist()))
                        raise ValueError(f"Geometry map references uncached anchor times {bad_times}")
                    source_flat = (
                        source_cache_index.clamp_min(0) * spatial_tokens
                        + source_index[target_time - 1].reshape(spatial_tokens, memory_slots)
                    ).reshape(-1)
                    source_value = cached_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, value_channels
                    )
                    candidate_weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1)
                    matched_value = (candidate_weights * source_value.float()).sum(dim=2)
                    gate = (
                        effective_alpha * _geometry_support_gate(confidence_t.amax(dim=-1))
                    ).reshape(1, spatial_tokens, 1)
                    mixed_grid[:, target_time] = (
                        value_grid[:, target_time].float() * (1.0 - gate)
                        + matched_value * gate
                    ).to(value.dtype)

                    print_key = ("cached-value", layer_idx, target_time)
                    if geometry_transport_debug and print_key not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(print_key)
                        print(
                            f"[geometry-transport] target_latent={target_time} "
                            f"mode=cached_value_transport layer={layer_idx} "
                            f"coverage={(confidence_sum > 0).float().mean().item():.3f} "
                            f"mean_gate={gate.float().mean().item():.5f} "
                            f"alpha={effective_alpha:.5f}",
                            flush=True,
                        )

                return mixed_grid.reshape_as(value)

            def _apply_cached_draft_value_delta(
                value: torch.Tensor,
                cache_key: str,
                layer_idx: int,
                mode_label: str = "cached_draft_value_delta",
                center_by_target_time: bool = False,
                consensus_gate: bool = False,
            ) -> torch.Tensor:
                """Apply the sparse correction measured in the frozen baseline draft.

                Unlike source-value replacement, this preserves the guided run's
                current target value and adds only the counterfactual baseline
                direction from its deformed target toward the matched stable surface.
                """
                if not draft_feature_cache_state["loaded"]:
                    return value
                if cache_key not in draft_feature_cache_state["features"]:
                    raise KeyError(
                        f"Draft value-delta cache has no entry for {cache_key}"
                    )

                cache_entry = draft_feature_cache_state["features"][cache_key]
                expected_fields = {
                    "target_flat",
                    "delta",
                    "confidence",
                }
                if consensus_gate:
                    expected_fields.update(
                        {
                            "agreement",
                            "unique_anchor_count",
                            "effective_anchor_count",
                        }
                    )
                if (
                    not isinstance(cache_entry, dict)
                    or set(cache_entry) != expected_fields
                ):
                    raise ValueError(
                        f"Draft value-delta cache entry {cache_key} is incomplete"
                    )
                target_flat = cache_entry["target_flat"].to(
                    device=value.device,
                    dtype=torch.long,
                )
                delta = cache_entry["delta"].to(
                    device=value.device,
                    dtype=value.dtype,
                )
                confidence = cache_entry["confidence"].to(
                    device=value.device,
                    dtype=torch.float32,
                )
                if (
                    target_flat.ndim != 1
                    or delta.ndim != 3
                    or confidence.ndim != 1
                    or delta.shape[0] != value.shape[0]
                    or delta.shape[1] != target_flat.numel()
                    or delta.shape[2] != value.shape[2]
                    or confidence.numel() != target_flat.numel()
                ):
                    raise ValueError(
                        f"Draft value-delta cache entry {cache_key} does not match "
                        f"current value shape {tuple(value.shape)}"
                    )
                if target_flat.numel() == 0:
                    return value
                if (
                    target_flat.min().item() < 0
                    or target_flat.max().item() >= value.shape[1]
                ):
                    raise ValueError(
                        f"Draft value-delta cache entry {cache_key} has "
                        "out-of-range target indices"
                    )

                confidence = confidence.clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                gate = _geometry_support_gate(confidence).reshape(1, -1, 1)
                agreement = None
                consensus_weight = None
                if consensus_gate:
                    agreement = cache_entry["agreement"].to(
                        device=value.device,
                        dtype=torch.float32,
                    ).clamp(0, 1)
                    unique_anchor_count = cache_entry["unique_anchor_count"].to(
                        device=value.device,
                    )
                    effective_anchor_count = cache_entry[
                        "effective_anchor_count"
                    ].to(
                        device=value.device,
                        dtype=torch.float32,
                    )
                    valid_consensus = (
                        (unique_anchor_count >= 2)
                        & (effective_anchor_count >= 1.8)
                    )
                    consensus_weight = (
                        (
                            agreement - geometry_transport_consensus_threshold
                        )
                        / (
                            1.0
                            - geometry_transport_consensus_threshold
                        )
                    ).clamp(0, 1)
                    consensus_weight = (
                        consensus_weight * valid_consensus.to(torch.float32)
                    )
                    gate = gate * consensus_weight.reshape(1, -1, 1)
                effective_alpha = _current_geometry_alpha()
                delta_float = delta.float()
                if center_by_target_time:
                    spatial_tokens = token_grid[1] * token_grid[2]
                    target_times = torch.div(
                        target_flat,
                        spatial_tokens,
                        rounding_mode="floor",
                    )
                    delta_float = delta_float.clone()
                    for target_time in target_times.unique():
                        time_mask = target_times == target_time
                        active_mask = time_mask & (confidence > 0)
                        if active_mask.any():
                            delta_float[:, time_mask] -= delta_float[
                                :, active_mask
                            ].mean(dim=1, keepdim=True)
                update = effective_alpha * gate * delta_float
                corrected = value.clone()
                corrected[:, target_flat] = (
                    value[:, target_flat].float() + update
                ).to(value.dtype)

                print_key = (
                    mode_label,
                    layer_idx,
                    attn_avg_state["step"],
                )
                if (
                    geometry_transport_debug
                    and print_key not in geometry_transport_state["printed"]
                ):
                    geometry_transport_state["printed"].add(print_key)
                    current_target = value[:, target_flat].float()
                    delta_rms = delta_float.square().mean().sqrt()
                    current_rms = current_target.square().mean().sqrt()
                    update_rms = update.square().mean().sqrt()
                    print(
                        f"[geometry-transport] step={attn_avg_state['step']} "
                        f"mode={mode_label} layer={layer_idx} "
                        f"active_targets={(confidence > 0).sum().item()} "
                        f"cached_delta_rms_ratio="
                        f"{(delta_rms / current_rms.clamp_min(1e-8)).item():.6f} "
                        f"applied_update_rms_ratio="
                        f"{(update_rms / current_rms.clamp_min(1e-8)).item():.6f} "
                        + (
                            f"agreement_mean={agreement.mean().item():.4f} "
                            f"consensus_gate_mean={consensus_weight.mean().item():.4f} "
                            f"consensus_active={(consensus_weight > 0).sum().item()} "
                            if consensus_gate
                            else ""
                        )
                        + f"alpha={effective_alpha:.5f}",
                        flush=True,
                    )
                return corrected

            def _matched_cached_geometry_values(
                value: torch.Tensor,
                cache_key: str,
                layer_idx: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Gather cached draft V while keeping the correction target-local."""
                if cache_key not in draft_feature_cache_state["features"]:
                    raise KeyError(f"Draft value cache has no entry for {cache_key}")

                batch_size, sequence_length, value_channels = value.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Cached value residual received an unexpected Wan token sequence length")

                cached = draft_feature_cache_state["features"][cache_key].to(
                    device=value.device,
                    dtype=value.dtype,
                )
                if cached.shape[0] == 1 and batch_size > 1:
                    cached = cached.expand(batch_size, -1, -1, -1)
                expected_prefix = (
                    batch_size,
                    len(draft_feature_cache_state["anchor_token_times"]),
                    spatial_tokens,
                )
                if cached.shape[:3] != expected_prefix or cached.shape[-1] != value_channels:
                    raise ValueError(
                        f"Cached attention value shape {tuple(cached.shape)} is incompatible with "
                        f"batch={batch_size}, anchors={draft_feature_cache_state['anchor_token_times']}, "
                        f"spatial_tokens={spatial_tokens}, channels={value_channels}"
                    )

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"].clone()
                confidence[confidence < geometry_transport_min_confidence] = 0
                memory_slots = source_time.shape[-1]
                source_lookup = torch.full(
                    (num_tokens_t,),
                    -1,
                    dtype=torch.long,
                    device=value.device,
                )
                for cache_index, token_time in enumerate(draft_feature_cache_state["anchor_token_times"]):
                    source_lookup[token_time] = cache_index

                cached_flat = cached.reshape(batch_size, -1, value_channels)
                matched = torch.zeros(
                    (batch_size, num_tokens_t - 1, spatial_tokens, value_channels),
                    device=value.device,
                    dtype=value.dtype,
                )
                target_confidence = torch.zeros(
                    (num_tokens_t - 1, spatial_tokens, 1),
                    device=value.device,
                    dtype=torch.float32,
                )

                for target_time in range(1, num_tokens_t):
                    confidence_t = confidence[target_time - 1].reshape(spatial_tokens, memory_slots)
                    confidence_sum = confidence_t.sum(dim=-1, keepdim=True)
                    if not (confidence_sum > 0).any():
                        continue
                    source_time_t = source_time[target_time - 1].reshape(spatial_tokens, memory_slots)
                    source_cache_index = source_lookup[source_time_t]
                    invalid = (source_cache_index < 0) & (confidence_t > 0)
                    if invalid.any():
                        bad_times = sorted(set(source_time_t[invalid].tolist()))
                        raise ValueError(f"Geometry map references uncached anchor times {bad_times}")
                    source_flat = (
                        source_cache_index.clamp_min(0) * spatial_tokens
                        + source_index[target_time - 1].reshape(spatial_tokens, memory_slots)
                    ).reshape(-1)
                    source_value = cached_flat[:, source_flat].reshape(
                        batch_size, spatial_tokens, memory_slots, value_channels
                    )
                    candidate_weights = (
                        confidence_t / confidence_sum.clamp_min(1e-8)
                    ).reshape(1, spatial_tokens, memory_slots, 1)
                    matched[:, target_time - 1] = (
                        candidate_weights * source_value.float()
                    ).sum(dim=2).to(value.dtype)
                    target_confidence[target_time - 1] = _geometry_support_gate(
                        confidence_t.amax(dim=-1, keepdim=True)
                    )

                    print_key = ("cached-value-residual", layer_idx, target_time)
                    if geometry_transport_debug and print_key not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(print_key)
                        print(
                            f"[geometry-transport] target_latent={target_time} "
                            f"mode=cached_value_residual layer={layer_idx} "
                            f"coverage={(confidence_sum > 0).float().mean().item():.3f} "
                            f"mean_confidence={target_confidence[target_time - 1].mean().item():.5f} "
                            f"alpha={_current_geometry_alpha():.5f}",
                            flush=True,
                        )

                confidence_out = target_confidence.reshape(
                    1, num_tokens_t - 1, height_tokens, width_tokens, 1
                ).expand(batch_size, -1, -1, -1, -1)
                return matched, confidence_out

            class _CorrespondenceKVProcessor:
                def __init__(self, base_processor, layer_idx: int, model_name: str):
                    self.base_processor = base_processor
                    self.layer_idx = layer_idx
                    self.model_name = model_name

                def __call__(
                    self,
                    attn,
                    hidden_states: torch.Tensor,
                    encoder_hidden_states: torch.Tensor | None = None,
                    attention_mask: torch.Tensor | None = None,
                    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
                ) -> torch.Tensor:
                    feature_match_active = (
                        attn_avg_alpha > 0.0
                        and attn_avg_start <= attn_avg_state["step"] <= attn_avg_end
                        and (not attn_avg_cond_only or attn_avg_state["branch"] == "cond")
                    )
                    geometry_active = (
                        geometry_transport_mode != "shadow"
                        and geometry_transport_state["ready"]
                        and geometry_transport_start <= attn_avg_state["step"] <= geometry_transport_end
                        and (not geometry_transport_cond_only or attn_avg_state["branch"] == "cond")
                        and (
                            geometry_transport_mode not in {
                                "geometry_attention_bias",
                                "geometry_attention_output_bias",
                                "geometry_sparse_logit_bias",
                                "geometry_virtual_token_bias",
                                "geometry_surface_graph_logit_bias",
                                "geometry_signed_surface_graph_logit_bias",
                                "geometry_relative_edge_output",
                                "cached_virtual_token_bias",
                            }
                            or geometry_transport_alpha > 0.0
                        )
                    )
                    if not feature_match_active and not geometry_active:
                        return self.base_processor(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

                    # attn1 is self-attention. Never silently alter a cross-attention path.
                    if encoder_hidden_states is not None or attn.add_k_proj is not None:
                        return self.base_processor(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

                    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                    if geometry_active and geometry_transport_mode == "cached_value_transport":
                        cache_key = f"{self.model_name}:{self.layer_idx}:{attn_avg_state['step']}"
                        value = _apply_cached_value_transport(value, cache_key, self.layer_idx)
                    if (
                        geometry_active
                        and geometry_transport_mode
                        in {
                            "cached_draft_value_delta",
                            "cached_draft_value_delta_consensus",
                        }
                    ):
                        cache_key = f"{self.model_name}:{self.layer_idx}:{attn_avg_state['step']}"
                        value = _apply_cached_draft_value_delta(
                            value,
                            cache_key,
                            self.layer_idx,
                            mode_label=geometry_transport_mode,
                            consensus_gate=(
                                geometry_transport_mode
                                == "cached_draft_value_delta_consensus"
                            ),
                        )
                    matched_values = None
                    match_confidence = None
                    match_similarity = None
                    reciprocal = None
                    match_source_index = None
                    match_source_time = None
                    geometry_value_residual = geometry_active and geometry_transport_mode in {
                        "value_residual",
                        "cached_value_residual",
                    }
                    geometry_memory = geometry_active and geometry_transport_mode in {
                        "kv_memory",
                        "attn_output_memory",
                        "geometry_attention_bias",
                        "geometry_attention_output_bias",
                        "geometry_sparse_logit_bias",
                        "geometry_virtual_token_bias",
                        "geometry_surface_graph_logit_bias",
                        "geometry_signed_surface_graph_logit_bias",
                        "geometry_relative_edge_output",
                        "cached_virtual_token_bias",
                        "cached_value_memory",
                        "cached_value_attention_blend",
                    }
                    geometry_qk_transport = geometry_active and geometry_transport_mode in {
                        "query_transport",
                        "qk_transport",
                    }
                    if geometry_value_residual:
                        if geometry_transport_mode == "cached_value_residual":
                            cache_key = f"{self.model_name}:{self.layer_idx}:{attn_avg_state['step']}"
                            matched_values, match_confidence = _matched_cached_geometry_values(
                                value,
                                cache_key,
                                self.layer_idx,
                            )
                        else:
                            (
                                matched_values,
                                match_confidence,
                                match_source_index,
                                match_source_time,
                            ) = _matched_geometry_values(value)
                    elif feature_match_active and attn_avg_mode in {"query_match_prev", "key_match_prev", "kv_match_prev"}:
                        query, key, value, match_similarity, reciprocal = _transport_previous_qkv(
                            query,
                            key,
                            value,
                            hidden_states,
                            transport_query=attn_avg_mode == "query_match_prev",
                            transport_key=attn_avg_mode in {"key_match_prev", "kv_match_prev"},
                            transport_value=attn_avg_mode == "kv_match_prev",
                        )
                    elif feature_match_active and attn_avg_mode in {
                        "value_residual_prev", "c2f_value_residual_prev", "c2f_value_residual_anchor",
                        "c2f_value_residual_memory",
                    }:
                        (
                            matched_values,
                            match_confidence,
                            match_similarity,
                            reciprocal,
                            match_source_index,
                            match_source_time,
                        ) = _matched_previous_values(
                            value,
                            hidden_states,
                        )

                    query = attn.norm_q(query)
                    key = attn.norm_k(key)
                    query = query.unflatten(2, (attn.heads, -1))
                    key = key.unflatten(2, (attn.heads, -1))
                    value = value.unflatten(2, (attn.heads, -1))
                    if geometry_qk_transport:
                        query, key = _apply_geometry_qk_transport(
                            query,
                            key,
                            transport_key=geometry_transport_mode == "qk_transport",
                        )
                    key_before_rope = key

                    if rotary_emb is not None:
                        def _apply_rotary_emb(states, freqs_cos, freqs_sin):
                            x1, x2 = states.unflatten(-1, (-1, 2)).unbind(-1)
                            cos = freqs_cos[..., 0::2]
                            sin = freqs_sin[..., 1::2]
                            rotated = torch.empty_like(states)
                            rotated[..., 0::2] = x1 * cos - x2 * sin
                            rotated[..., 1::2] = x1 * sin + x2 * cos
                            return rotated.type_as(states)

                        query = _apply_rotary_emb(query, *rotary_emb)
                        key = _apply_rotary_emb(key, *rotary_emb)

                    hidden_states = dispatch_attention_fn(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=getattr(self.base_processor, "_attention_backend", None),
                        parallel_config=getattr(self.base_processor, "_parallel_config", None),
                    )

                    if (
                        geometry_active
                        and geometry_transport_mode
                        in {
                            "cached_draft_output_delta",
                            "cached_draft_output_delta_centered",
                        }
                    ):
                        cache_key = (
                            f"{self.model_name}:{self.layer_idx}:"
                            f"{attn_avg_state['step']}"
                        )
                        attended_flat = hidden_states.flatten(2, 3)
                        attended_flat = _apply_cached_draft_value_delta(
                            attended_flat,
                            cache_key,
                            self.layer_idx,
                            mode_label=geometry_transport_mode,
                            center_by_target_time=(
                                geometry_transport_mode
                                == "cached_draft_output_delta_centered"
                            ),
                        )
                        hidden_states = attended_flat.unflatten(
                            2,
                            (attn.heads, -1),
                        )

                    if geometry_memory:
                        if (
                            geometry_transport_mode
                            in {
                                "geometry_surface_graph_logit_bias",
                                "geometry_signed_surface_graph_logit_bias",
                            }
                        ):
                            if attention_mask is not None:
                                raise ValueError(
                                    f"{geometry_transport_mode} currently "
                                    "requires unmasked Wan self-attention"
                                )
                            hidden_states = (
                                _apply_geometry_surface_graph_logit_bias(
                                    query,
                                    key,
                                    value,
                                    hidden_states,
                                )
                            )
                        elif (
                            geometry_transport_mode
                            == "geometry_relative_edge_output"
                        ):
                            hidden_states = (
                                _apply_geometry_relative_edge_output(
                                    hidden_states
                                )
                            )
                        elif geometry_transport_mode in {
                            "geometry_sparse_logit_bias",
                            "geometry_virtual_token_bias",
                            "cached_virtual_token_bias",
                        }:
                            if attention_mask is not None:
                                raise ValueError(
                                    f"{geometry_transport_mode} currently requires unmasked Wan self-attention"
                                )
                            cache_key = (
                                f"{self.model_name}:{self.layer_idx}:"
                                f"{attn_avg_state['step']}"
                            )
                            hidden_states = _apply_geometry_sparse_logit_bias(
                                query,
                                key,
                                key_before_rope,
                                value,
                                hidden_states,
                                rotary_emb,
                                virtual_token=geometry_transport_mode
                                == "geometry_virtual_token_bias",
                                cached_virtual_token=geometry_transport_mode
                                == "cached_virtual_token_bias",
                                cache_key=cache_key,
                                key_normalizer=attn.norm_k,
                            )
                        elif geometry_transport_mode in {
                            "cached_value_memory",
                            "cached_value_attention_blend",
                        }:
                            cache_key = (
                                f"{self.model_name}:{self.layer_idx}:"
                                f"{attn_avg_state['step']}"
                            )
                            hidden_states = _apply_cached_value_memory(
                                query,
                                key_before_rope,
                                value,
                                hidden_states,
                                rotary_emb,
                                cache_key,
                                self.layer_idx,
                            )
                        else:
                            hidden_states = _apply_geometry_memory(
                                query,
                                key_before_rope,
                                value,
                                hidden_states,
                                rotary_emb,
                                geometry_transport_mode,
                            )

                    if matched_values is not None and match_confidence is not None:
                        num_tokens_t, height_tokens, width_tokens = token_grid
                        spatial_tokens = height_tokens * width_tokens
                        attended_grid = hidden_states.reshape(
                            batch_size, num_tokens_t, spatial_tokens, attn.heads, -1
                        )
                        matched_value_grid = matched_values.unflatten(-1, (attn.heads, -1))
                        transport_alpha = _current_geometry_alpha() if geometry_value_residual else attn_avg_alpha
                        blend = (
                            transport_alpha * match_confidence.reshape(
                                batch_size, num_tokens_t - 1, spatial_tokens, 1, 1
                            )
                        )
                        mixed_grid = attended_grid.clone()
                        if geometry_value_residual:
                            current_value_grid = value.reshape(
                                batch_size, num_tokens_t, spatial_tokens, attn.heads, -1
                            )
                            mixed_grid[:, 1:] = (
                                attended_grid[:, 1:].float()
                                + blend
                                * (
                                    matched_value_grid.float()
                                    - current_value_grid[:, 1:].float()
                                )
                            ).to(hidden_states.dtype)
                        else:
                            mixed_grid[:, 1:] = (
                                attended_grid[:, 1:].float() * (1.0 - blend)
                                + matched_value_grid.float() * blend
                            ).to(hidden_states.dtype)
                        hidden_states = mixed_grid.reshape_as(hidden_states)

                    hidden_states = hidden_states.flatten(2, 3).type_as(query)
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)

                    if attn_avg_debug and self.layer_idx == attn_avg_layers[0] and match_similarity is not None:
                        print_key = (attn_avg_state["step"], attn_avg_state["branch"])
                        if print_key not in attn_avg_state["printed"]:
                            attn_avg_state["printed"].add(print_key)
                            reciprocal_info = (
                                f" reciprocal_keep={reciprocal.float().mean().item():.3f}"
                                if reciprocal is not None
                                else ""
                            )
                            residual_gate_info = (
                                f"residual_gate_mean={match_confidence.float().mean().item():.3f} "
                                if match_confidence is not None
                                else ""
                            )
                            displacement_info = ""
                            if match_source_index is not None and match_confidence is not None:
                                _, height_tokens, width_tokens = token_grid
                                spatial_tokens = height_tokens * width_tokens
                                current_index = torch.arange(
                                    spatial_tokens, device=match_source_index.device
                                ).view(1, 1, -1)
                                current_y = torch.div(current_index, width_tokens, rounding_mode="floor")
                                current_x = current_index.remainder(width_tokens)
                                source_y = torch.div(match_source_index, width_tokens, rounding_mode="floor")
                                source_x = match_source_index.remainder(width_tokens)
                                delta_y = source_y - current_y
                                delta_x = source_x - current_x
                                active_match = match_confidence.reshape_as(match_source_index) > 0
                                if active_match.any():
                                    at_radius = (delta_y.abs() == attn_avg_match_radius) | (
                                        delta_x.abs() == attn_avg_match_radius
                                    )
                                    displacement_info = (
                                        f"match_abs_dy={delta_y.abs()[active_match].float().mean().item():.2f} "
                                        f"match_abs_dx={delta_x.abs()[active_match].float().mean().item():.2f} "
                                        f"match_at_radius={at_radius[active_match].float().mean().item():.3f} "
                                    )
                            memory_lag_info = ""
                            if match_source_time is not None and match_confidence is not None:
                                current_time = torch.arange(
                                    1, token_grid[0], device=match_source_time.device
                                ).view(1, -1, 1)
                                active_match = match_confidence.reshape_as(match_source_time) > 0
                                if active_match.any():
                                    memory_lag = current_time - match_source_time
                                    memory_lag_info = (
                                        f"memory_lag_mean={memory_lag[active_match].float().mean().item():.2f} "
                                    )
                            print(
                                f"[attn-manip] step={attn_avg_state['step']} branch={attn_avg_state['branch']} "
                                f"mode={attn_avg_mode} alpha={attn_avg_alpha:.3f} "
                                f"match_cos_mean={match_similarity.float().mean().item():.3f} "
                                f"match_cos_p10={torch.quantile(match_similarity.float(), 0.1).item():.3f} "
                                f"match_cos_p90={torch.quantile(match_similarity.float(), 0.9).item():.3f} "
                                f"{residual_gate_info}"
                                f"{reciprocal_info}"
                                f"{displacement_info}"
                                f"{memory_lag_info}"
                            )
                    return hidden_states

            def _make_attention_hook(layer_idx: int):
                def _attention_feature_hook(module, inputs, output):
                    if output is None or attn_avg_state["step"] < attn_avg_start or attn_avg_state["step"] > attn_avg_end:
                        return output
                    if attn_avg_cond_only and attn_avg_state["branch"] != "cond":
                        return output
                    if output.ndim != 3:
                        raise ValueError(f"Attention output must be [B, N, C], got shape {tuple(output.shape)}")

                    batch_size, seq_len, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    expected_seq_len = num_tokens_t * height_tokens * width_tokens
                    if seq_len != expected_seq_len:
                        raise ValueError(
                            f"Attention output seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                        )

                    hidden = output.reshape(batch_size, num_tokens_t, height_tokens, width_tokens, channels)
                    correspondence = None
                    if attn_avg_mode == "global":
                        target = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
                        mixed = (hidden.float() * (1.0 - attn_avg_alpha) + target.float() * attn_avg_alpha).to(output.dtype)
                        mixed = _preserve_conditioning_frame(mixed, hidden)
                    elif attn_avg_mode == "local":
                        target = _temporal_window_mean(hidden)
                        mixed = (hidden.float() * (1.0 - attn_avg_alpha) + target.float() * attn_avg_alpha).to(output.dtype)
                        mixed = _preserve_conditioning_frame(mixed, hidden)
                    elif attn_avg_mode == "anchor":
                        target = hidden[:, :1].expand_as(hidden)
                        mixed = (hidden.float() * (1.0 - attn_avg_alpha) + target.float() * attn_avg_alpha).to(output.dtype)
                        mixed = _preserve_conditioning_frame(mixed, hidden)
                    else:
                        if not inputs or inputs[0].ndim != 3:
                            raise ValueError("match_prev requires the attention input to have shape [B, N, C]")
                        input_hidden = inputs[0].reshape(batch_size, num_tokens_t, height_tokens, width_tokens, channels)
                        mixed, correspondence = _match_previous_frame(hidden, input_hidden)

                    if attn_avg_debug and layer_idx == attn_avg_layers[0]:
                        print_key = (attn_avg_state["step"], attn_avg_state["branch"])
                        if print_key not in attn_avg_state["printed"]:
                            attn_avg_state["printed"].add(print_key)
                            if correspondence is None:
                                print(
                                    f"[attn-manip] step={attn_avg_state['step']} branch={attn_avg_state['branch']} "
                                    f"mode={attn_avg_mode} alpha={attn_avg_alpha:.3f}"
                                )
                            else:
                                best_similarity, blend = correspondence
                                print(
                                    f"[attn-manip] step={attn_avg_state['step']} branch={attn_avg_state['branch']} "
                                    f"mode={attn_avg_mode} alpha={attn_avg_alpha:.3f} "
                                    f"match_cos_mean={best_similarity.float().mean().item():.3f} "
                                    f"match_cos_p10={torch.quantile(best_similarity.float(), 0.1).item():.3f} "
                                    f"match_cos_p90={torch.quantile(best_similarity.float(), 0.9).item():.3f} "
                                    f"active_fraction={(blend > 0).float().mean().item():.3f} "
                                    f"mean_blend={blend.float().mean().item():.4f}"
                                )
                    return mixed.reshape(batch_size, seq_len, channels)

                return _attention_feature_hook

            def _make_draft_feature_hook(model_name: str, layer_idx: int):
                def _draft_feature_hook(module, inputs, output):
                    if output is None or attn_avg_state["branch"] != "cond":
                        return output
                    if output.ndim != 3:
                        raise ValueError(f"Wan block output must be [B, N, C], got {tuple(output.shape)}")

                    batch_size, sequence_length, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    spatial_tokens = height_tokens * width_tokens
                    if sequence_length != num_tokens_t * spatial_tokens:
                        raise ValueError("Draft feature hook received an unexpected Wan token sequence length")
                    cache_key = f"{model_name}:{layer_idx}:{attn_avg_state['step']}"

                    if (
                        draft_feature_cache_state["capture"]
                        and geometry_transport_start
                        <= attn_avg_state["step"]
                        <= geometry_transport_end
                    ):
                        hidden_grid = output.reshape(
                            batch_size, num_tokens_t, spatial_tokens, channels
                        )
                        anchor_times = draft_feature_cache_state["anchor_token_times"]
                        draft_feature_cache_state["features"][cache_key] = (
                            hidden_grid[:, anchor_times]
                            .detach()
                            .to(device="cpu", dtype=torch.bfloat16)
                            .contiguous()
                        )

                    transport_active = (
                        geometry_transport_mode == "cached_block_transport"
                        and draft_feature_cache_state["loaded"]
                        and geometry_transport_start <= attn_avg_state["step"] <= geometry_transport_end
                    )
                    if transport_active:
                        return _apply_cached_block_transport(output, cache_key, layer_idx)
                    return output

                return _draft_feature_hook

            def _make_draft_value_hook(model_name: str, layer_idx: int):
                def _draft_value_hook(module, inputs, output):
                    if output is None or attn_avg_state["branch"] != "cond":
                        return output
                    if output.ndim != 3:
                        raise ValueError(f"Wan attention value must be [B, N, C], got {tuple(output.shape)}")
                    if not (
                        draft_feature_cache_state["capture"]
                        and geometry_transport_start
                        <= attn_avg_state["step"]
                        <= geometry_transport_end
                    ):
                        return output

                    batch_size, sequence_length, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    spatial_tokens = height_tokens * width_tokens
                    if sequence_length != num_tokens_t * spatial_tokens:
                        raise ValueError("Draft value hook received an unexpected Wan token sequence length")
                    cache_key = f"{model_name}:{layer_idx}:{attn_avg_state['step']}"
                    value_grid = output.reshape(
                        batch_size, num_tokens_t, spatial_tokens, channels
                    )
                    anchor_times = draft_feature_cache_state["anchor_token_times"]
                    draft_feature_cache_state["features"][cache_key] = (
                        value_grid[:, anchor_times]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    return output

                return _draft_value_hook

            def _make_draft_value_delta_hook(model_name: str, layer_idx: int):
                def _draft_value_delta_hook(module, inputs, output):
                    if output is None or attn_avg_state["branch"] != "cond":
                        return output
                    if output.ndim != 3:
                        raise ValueError(
                            "Wan attention value must be [B, N, C], "
                            f"got {tuple(output.shape)}"
                        )
                    if not (
                        draft_feature_cache_state["capture"]
                        and geometry_transport_start
                        <= attn_avg_state["step"]
                        <= geometry_transport_end
                    ):
                        return output

                    batch_size, sequence_length, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    spatial_tokens = height_tokens * width_tokens
                    if sequence_length != num_tokens_t * spatial_tokens:
                        raise ValueError(
                            "Draft value-delta hook received an unexpected "
                            "Wan token sequence length"
                        )

                    value_grid = output.reshape(
                        batch_size,
                        num_tokens_t,
                        spatial_tokens,
                        channels,
                    )
                    value_flat = value_grid.reshape(
                        batch_size,
                        num_tokens_t * spatial_tokens,
                        channels,
                    )
                    source_time = geometry_transport_state["source_time"]
                    source_index = geometry_transport_state["source_index"]
                    confidence = geometry_transport_state["confidence"]
                    memory_slots = source_time.shape[-1]
                    target_flat_parts = []
                    delta_parts = []
                    confidence_parts = []

                    for target_time in range(1, num_tokens_t):
                        confidence_t = confidence[target_time - 1].reshape(
                            spatial_tokens,
                            memory_slots,
                        )
                        confidence_sum = confidence_t.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        support = confidence_sum[:, 0] > 0
                        if not support.any():
                            continue
                        target_spatial = support.nonzero(
                            as_tuple=False
                        ).squeeze(-1)
                        source_flat = (
                            source_time[
                                target_time - 1,
                                target_spatial,
                            ]
                            * spatial_tokens
                            + source_index[
                                target_time - 1,
                                target_spatial,
                            ]
                        )
                        source_value = value_flat[
                            :, source_flat.reshape(-1)
                        ].reshape(
                            batch_size,
                            target_spatial.numel(),
                            memory_slots,
                            channels,
                        )
                        candidate_confidence = confidence_t[target_spatial]
                        candidate_weights = (
                            candidate_confidence
                            / candidate_confidence.sum(
                                dim=-1,
                                keepdim=True,
                            ).clamp_min(1e-8)
                        ).reshape(
                            1,
                            target_spatial.numel(),
                            memory_slots,
                            1,
                        )
                        matched_source = (
                            candidate_weights * source_value.float()
                        ).sum(dim=2)
                        target_value = value_grid[
                            :, target_time, target_spatial
                        ].float()
                        target_flat_parts.append(
                            target_time * spatial_tokens + target_spatial
                        )
                        delta_parts.append(matched_source - target_value)
                        confidence_parts.append(
                            candidate_confidence.amax(dim=-1)
                        )

                    cache_key = (
                        f"{model_name}:{layer_idx}:"
                        f"{attn_avg_state['step']}"
                    )
                    if target_flat_parts:
                        target_flat = torch.cat(target_flat_parts)
                        delta = torch.cat(delta_parts, dim=1)
                        target_confidence = torch.cat(confidence_parts)
                    else:
                        target_flat = torch.empty(
                            0,
                            dtype=torch.long,
                            device=output.device,
                        )
                        delta = output.new_empty(batch_size, 0, channels)
                        target_confidence = torch.empty(
                            0,
                            dtype=torch.float32,
                            device=output.device,
                        )
                    draft_feature_cache_state["features"][cache_key] = {
                        "target_flat": target_flat.detach().to(
                            device="cpu",
                            dtype=torch.long,
                        ).contiguous(),
                        "delta": delta.detach().to(
                            device="cpu",
                            dtype=torch.bfloat16,
                        ).contiguous(),
                        "confidence": target_confidence.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        ).contiguous(),
                    }
                    return output

                return _draft_value_delta_hook

            def _make_draft_kv_hook(
                model_name: str,
                layer_idx: int,
                projection_name: str,
            ):
                if projection_name not in {"key", "value"}:
                    raise ValueError(f"Unsupported draft K/V projection: {projection_name}")

                def _draft_kv_hook(module, inputs, output):
                    if output is None or attn_avg_state["branch"] != "cond":
                        return output
                    if output.ndim != 3:
                        raise ValueError(
                            f"Wan attention {projection_name} must be [B, N, C], "
                            f"got {tuple(output.shape)}"
                        )
                    if not (
                        draft_feature_cache_state["capture"]
                        and geometry_transport_start
                        <= attn_avg_state["step"]
                        <= geometry_transport_end
                    ):
                        return output

                    batch_size, sequence_length, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    spatial_tokens = height_tokens * width_tokens
                    if sequence_length != num_tokens_t * spatial_tokens:
                        raise ValueError(
                            "Draft K/V hook received an unexpected Wan token sequence length"
                        )
                    cache_key = f"{model_name}:{layer_idx}:{attn_avg_state['step']}"
                    projected_grid = output.reshape(
                        batch_size, num_tokens_t, spatial_tokens, channels
                    )
                    anchor_times = draft_feature_cache_state["anchor_token_times"]
                    cached_projection = (
                        projected_grid[:, anchor_times]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    cache_entry = draft_feature_cache_state["features"].setdefault(
                        cache_key, {}
                    )
                    if not isinstance(cache_entry, dict):
                        raise ValueError(
                            f"Draft K/V cache entry {cache_key} was already written "
                            "with an incompatible cache kind"
                        )
                    cache_entry[projection_name] = cached_projection
                    return output

                return _draft_kv_hook

            processor_layers = set()
            if attn_avg_alpha > 0.0 and attn_avg_layers and attn_avg_mode in {
                "query_match_prev", "key_match_prev", "kv_match_prev", "value_residual_prev",
                "c2f_value_residual_prev", "c2f_value_residual_anchor", "c2f_value_residual_memory",
            }:
                processor_layers.update(attn_avg_layers)
            attention_geometry_modes = {
                "value_residual",
                "kv_memory",
                "attn_output_memory",
                "geometry_attention_bias",
                "geometry_attention_output_bias",
                "geometry_sparse_logit_bias",
                "geometry_virtual_token_bias",
                "geometry_surface_graph_logit_bias",
                "geometry_signed_surface_graph_logit_bias",
                "geometry_relative_edge_output",
                "cached_virtual_token_bias",
                "query_transport",
                "qk_transport",
                "cached_value_transport",
                "cached_value_residual",
                "cached_value_memory",
                "cached_value_attention_blend",
                "cached_draft_value_delta",
                "cached_draft_value_delta_consensus",
                "cached_draft_output_delta",
                "cached_draft_output_delta_centered",
            }
            if geometry_transport_mode in attention_geometry_modes:
                processor_layers.update(geometry_transport_layers)
            hook_layers = set(attn_avg_layers or []) if attn_avg_alpha > 0.0 else set()

            for model_name, model in (
                ("transformer", self.transformer),
                ("transformer_2", self.transformer_2),
            ):
                if model is None:
                    continue
                for layer_idx in sorted(processor_layers | hook_layers):
                    if layer_idx < 0 or layer_idx >= len(model.blocks):
                        raise IndexError(f"attn_avg layer {layer_idx} is out of range for {model.__class__.__name__}")
                    attention_layer = model.blocks[layer_idx].attn1
                    if layer_idx in processor_layers:
                        if attention_layer.add_k_proj is not None:
                            raise ValueError("correspondence attention manipulation only supports Wan self-attention without added image K/V")
                        attn_avg_processor_backups.append((attention_layer, attention_layer.processor))
                        attention_layer.set_processor(
                            _CorrespondenceKVProcessor(attention_layer.processor, layer_idx, model_name)
                        )
                    elif layer_idx in hook_layers:
                        handle = attention_layer.register_forward_hook(_make_attention_hook(layer_idx))
                        attn_avg_hook_handles.append(handle)
                block_cache_active = (
                    draft_feature_cache_state["capture"]
                    and draft_feature_cache_kind == "block_output"
                ) or geometry_transport_mode == "cached_block_transport"
                value_cache_capture = (
                    draft_feature_cache_state["capture"]
                    and draft_feature_cache_kind == "attention_value"
                )
                value_delta_cache_capture = (
                    draft_feature_cache_state["capture"]
                    and draft_feature_cache_kind == "attention_value_delta"
                )
                kv_cache_capture = (
                    draft_feature_cache_state["capture"]
                    and draft_feature_cache_kind == "attention_kv"
                )
                if (
                    block_cache_active
                    or value_cache_capture
                    or value_delta_cache_capture
                    or kv_cache_capture
                ):
                    for layer_idx in geometry_transport_layers:
                        if layer_idx < 0 or layer_idx >= len(model.blocks):
                            raise IndexError(
                                f"draft feature layer {layer_idx} is out of range for {model.__class__.__name__}"
                            )
                        if kv_cache_capture:
                            attention_layer = model.blocks[layer_idx].attn1
                            key_handle = attention_layer.to_k.register_forward_hook(
                                _make_draft_kv_hook(
                                    model_name,
                                    layer_idx,
                                    "key",
                                )
                            )
                            value_handle = attention_layer.to_v.register_forward_hook(
                                _make_draft_kv_hook(
                                    model_name,
                                    layer_idx,
                                    "value",
                                )
                            )
                            draft_feature_hook_handles.extend(
                                [key_handle, value_handle]
                            )
                            continue
                        if value_delta_cache_capture:
                            handle = model.blocks[
                                layer_idx
                            ].attn1.to_v.register_forward_hook(
                                _make_draft_value_delta_hook(
                                    model_name,
                                    layer_idx,
                                )
                            )
                            draft_feature_hook_handles.append(handle)
                            continue
                        if value_cache_capture:
                            handle = model.blocks[layer_idx].attn1.to_v.register_forward_hook(
                                _make_draft_value_hook(model_name, layer_idx)
                            )
                        else:
                            handle = model.blocks[layer_idx].register_forward_hook(
                                _make_draft_feature_hook(model_name, layer_idx)
                            )
                        draft_feature_hook_handles.append(handle)

        base_condition = condition
        base_first_frame_mask = first_frame_mask if self.config.expand_timesteps else None
        geometry_condition = None
        geometry_condition_weight_gate = None
        geometry_condition_debug_steps: set[int] = set()

        if draft_geometry_condition_alpha > 0.0:
            if condition.shape[2] == 1 and latents.shape[2] > 1:
                condition = condition.expand(
                    -1,
                    -1,
                    latents.shape[2],
                    -1,
                    -1,
                ).clone()
            base_condition = condition
            base_first_frame_mask = first_frame_mask
            matched_condition, condition_gate = _matched_clean_latent_residual(
                condition.float(),
                return_target=True,
                spatial_dilation=draft_geometry_condition_spatial_dilation,
            )
            if matched_condition.shape != condition.shape:
                raise ValueError(
                    "Draft geometry condition shape does not match Wan condition: "
                    f"matched={list(matched_condition.shape)} "
                    f"condition={list(condition.shape)}"
                )
            condition_support = condition_gate > 0
            if draft_geometry_condition_reference_mode == "lowpass_delta":
                radius = draft_geometry_condition_lowpass_radius
                kernel_size = 2 * radius + 1
                support_float = condition_support.float()
                pool_weight = condition_gate.float().clamp_min(0)
                condition_delta = (
                    matched_condition.float() - condition.float()
                )
                local_delta_sum = torch.nn.functional.avg_pool3d(
                    condition_delta * pool_weight,
                    kernel_size=(1, kernel_size, kernel_size),
                    stride=1,
                    padding=(0, radius, radius),
                )
                local_support = torch.nn.functional.avg_pool3d(
                    pool_weight,
                    kernel_size=(1, kernel_size, kernel_size),
                    stride=1,
                    padding=(0, radius, radius),
                )
                lowpass_delta = (
                    local_delta_sum / local_support.clamp_min(1e-8)
                )
                condition_target = condition.float() + lowpass_delta
            else:
                condition_target = matched_condition.float()
            geometry_condition = torch.where(
                condition_support,
                condition_target.to(condition.dtype),
                condition,
            )
            if draft_geometry_condition_gate_mode == "binary_support":
                condition_weight_gate = condition_support.float()
            elif draft_geometry_condition_gate_mode == "target_p90":
                # Risk magnitudes are not calibrated across videos. Normalize
                # each target time by its active 90th percentile while keeping
                # the detector's spatial ranking and zero support unchanged.
                condition_weight_gate = torch.zeros_like(condition_gate.float())
                for target_idx in range(condition_gate.shape[2]):
                    target_gate = condition_gate[:, :, target_idx].float()
                    active_target_gate = target_gate[target_gate > 0]
                    if active_target_gate.numel() == 0:
                        continue
                    robust_scale = torch.quantile(
                        active_target_gate,
                        0.9,
                    ).clamp_min(1e-6)
                    condition_weight_gate[:, :, target_idx] = (
                        target_gate / robust_scale
                    ).clamp(0, 1)
            else:
                condition_weight_gate = condition_gate.float().clamp(0, 1)
            geometry_condition_weight_gate = condition_weight_gate.to(
                first_frame_mask.dtype
            )
            if draft_geometry_condition_debug:
                active_weight = geometry_condition_weight_gate[
                    geometry_condition_weight_gate > 0
                ]
                print(
                    "[draft-geometry-condition] "
                    f"gate_mode={draft_geometry_condition_gate_mode} "
                    f"spatial_dilation={draft_geometry_condition_spatial_dilation} "
                    f"reference_mode={draft_geometry_condition_reference_mode} "
                    f"lowpass_radius={draft_geometry_condition_lowpass_radius} "
                    f"support={condition_support.float().mean().item():.6f} "
                    f"gate_mean_active="
                    f"{active_weight.mean().item() if active_weight.numel() else 0.0:.6f} "
                    f"gate_max="
                    f"{active_weight.max().item() if active_weight.numel() else 0.0:.6f}",
                    flush=True,
                )

        def _geometry_condition_schedule_scale(step_index: int) -> float:
            if (
                draft_geometry_condition_alpha <= 0.0
                or step_index < draft_geometry_condition_start_step
            ):
                return 0.0
            if (
                draft_geometry_condition_schedule == "constant"
                or draft_geometry_condition_ramp_end_step
                <= draft_geometry_condition_start_step
                or step_index >= draft_geometry_condition_ramp_end_step
            ):
                return 1.0
            progress = (
                step_index - draft_geometry_condition_start_step
            ) / (
                draft_geometry_condition_ramp_end_step
                - draft_geometry_condition_start_step
            )
            if draft_geometry_condition_schedule == "linear_ramp":
                return float(progress)
            return float(0.5 - 0.5 * math.cos(math.pi * progress))

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
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

                if self.config.expand_timesteps:
                    condition_scale = _geometry_condition_schedule_scale(i)
                    if condition_scale > 0.0:
                        conditioning_weight = (
                            draft_geometry_condition_alpha
                            * condition_scale
                            * geometry_condition_weight_gate
                        ).to(base_first_frame_mask.dtype)
                        step_first_frame_mask = base_first_frame_mask * (
                            1.0 - conditioning_weight
                        )
                        step_condition = geometry_condition
                    else:
                        step_first_frame_mask = base_first_frame_mask
                        step_condition = base_condition
                    latent_model_input = (
                        (1 - step_first_frame_mask) * step_condition
                        + step_first_frame_mask * latents
                    )
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (
                        step_first_frame_mask[0][0][:, ::2, ::2] * t
                    ).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    if (
                        draft_geometry_condition_debug
                        and draft_geometry_condition_alpha > 0.0
                        and i not in geometry_condition_debug_steps
                        and i
                        in {
                            draft_geometry_condition_start_step,
                            draft_geometry_condition_ramp_end_step,
                            num_inference_steps - 1,
                        }
                    ):
                        geometry_condition_debug_steps.add(i)
                        debug_weight = (
                            draft_geometry_condition_alpha
                            * condition_scale
                            * geometry_condition_weight_gate
                        )
                        active_weight = debug_weight[
                            geometry_condition_weight_gate > 0
                        ]
                        print(
                            "[draft-geometry-condition-step] "
                            f"step={i} schedule="
                            f"{draft_geometry_condition_schedule} "
                            f"scale={condition_scale:.6f} "
                            f"weight_mean_active="
                            f"{active_weight.mean().item() if active_weight.numel() else 0.0:.6f} "
                            f"weight_max="
                            f"{active_weight.max().item() if active_weight.numel() else 0.0:.6f}",
                            flush=True,
                        )
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                attn_avg_state["step"] = i
                attn_avg_state["branch"] = "cond"
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    attn_avg_state["branch"] = "uncond"
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

                if draft_vector_cache_state["capture"]:
                    draft_vector_cache_state["predictions"][i] = (
                        noise_pred.detach().to(device="cpu").contiguous()
                    )

                if draft_vector_anchor_outside_geometry or draft_vector_geometry_alpha > 0.0:
                    draft_prediction = draft_vector_cache_state["predictions"][i].to(
                        device=noise_pred.device,
                        dtype=noise_pred.dtype,
                    )
                    if draft_prediction.shape != noise_pred.shape:
                        raise ValueError(
                            "Draft vector prediction shape does not match the current model output: "
                            f"cache={list(draft_prediction.shape)} run={list(noise_pred.shape)}"
                        )

                clean_latent_active = (
                    draft_clean_latent_alpha > 0.0
                    and geometry_transport_start <= i <= geometry_transport_end
                )
                if clean_latent_active:
                    prediction_type = getattr(
                        self.scheduler.config,
                        "prediction_type",
                        None,
                    )
                    if prediction_type != "flow_prediction":
                        raise ValueError(
                            "Clean x0 latent transport currently requires "
                            "scheduler prediction_type='flow_prediction'"
                        )
                    if self.scheduler.step_index is None:
                        self.scheduler._init_step_index(t)
                    sigma = self.scheduler.sigmas[self.scheduler.step_index].to(
                        device=latents.device,
                        dtype=torch.float32,
                    )
                    if sigma.item() <= 1e-6:
                        raise ValueError(
                            "Clean x0 latent transport cannot run at zero sigma"
                        )

                    noise_pred_float = noise_pred.float()
                    current_x0 = latents.float() - sigma * noise_pred_float
                    clean_residual, clean_support = _matched_clean_latent_residual(
                        current_x0
                    )
                    clean_alpha = _current_draft_clean_latent_alpha()
                    if draft_clean_latent_sigma_scaled:
                        x0_update = (
                            clean_alpha * sigma * clean_residual.float()
                        )
                    else:
                        x0_update = clean_alpha * clean_residual.float()
                    noise_update = -x0_update / sigma
                    noise_pred = (noise_pred_float + noise_update).to(
                        noise_pred.dtype
                    )

                    if (
                        draft_clean_latent_debug
                        and i not in draft_clean_latent_state["printed"]
                    ):
                        draft_clean_latent_state["printed"].add(i)
                        clean_alpha_value = (
                            (clean_alpha * sigma).item()
                            if draft_clean_latent_sigma_scaled
                            else clean_alpha
                        )
                        active_support = clean_support.float().expand_as(x0_update)
                        support_sum = active_support.gt(0).sum().clamp_min(1)
                        x0_update_inside = (
                            x0_update.abs() * active_support.gt(0)
                        ).sum() / support_sum
                        noise_update_inside = (
                            noise_update.abs() * active_support.gt(0)
                        ).sum() / support_sum
                        print(
                            f"[draft-clean-latent] step={i} "
                            f"sigma={sigma.item():.6f} "
                            f"alpha={clean_alpha_value:.6f} "
                            f"reference={draft_clean_latent_reference_mode} "
                            f"sigma_scaled={draft_clean_latent_sigma_scaled} "
                            f"delta_quantile={draft_clean_latent_delta_quantile:.3f} "
                            f"lowpass_radius={draft_clean_latent_lowpass_radius} "
                            f"support={clean_support.gt(0).float().mean().item():.6f} "
                            f"x0_update_inside={x0_update_inside.item():.8f} "
                            f"noise_update_inside={noise_update_inside.item():.8f}",
                            flush=True,
                        )

                vector_geometry_active = (
                    draft_vector_geometry_alpha > 0.0
                    and geometry_transport_start <= i <= geometry_transport_end
                )
                if vector_geometry_active:
                    vector_residual, vector_support = _matched_draft_vector_residual(
                        draft_prediction
                    )
                    vector_alpha = _current_draft_vector_geometry_alpha()
                    vector_update = vector_alpha * vector_residual
                    noise_pred = noise_pred + vector_update
                    if (
                        draft_vector_geometry_debug
                        and i not in draft_vector_cache_state["geometry_printed"]
                    ):
                        draft_vector_cache_state["geometry_printed"].add(i)
                        support_sum = vector_support.expand_as(vector_update).sum().clamp_min(1)
                        update_inside = (
                            vector_update.float().abs()
                            * vector_support.float()
                        ).sum() / support_sum.float()
                        print(
                            f"[draft-vector-geometry] step={i} "
                            f"alpha={vector_alpha:.6f} "
                            f"support={vector_support.float().mean().item():.6f} "
                            f"update_inside={update_inside.item():.8f}",
                            flush=True,
                        )

                if draft_vector_anchor_outside_geometry:
                    geometry_mask = draft_vector_cache_state["geometry_mask"].to(
                        device=noise_pred.device,
                        dtype=noise_pred.dtype,
                    )
                    guided_delta = noise_pred - draft_prediction
                    noise_pred = draft_prediction + geometry_mask * guided_delta
                    if draft_vector_anchor_debug and i not in draft_vector_cache_state["printed"]:
                        draft_vector_cache_state["printed"].add(i)
                        expanded_mask = geometry_mask.expand_as(guided_delta)
                        mask_sum = expanded_mask.sum().clamp_min(1)
                        inside_delta = (
                            guided_delta.float().abs() * expanded_mask.float()
                        ).sum() / mask_sum.float()
                        outside_mask = 1.0 - expanded_mask
                        outside_sum = outside_mask.sum().clamp_min(1)
                        outside_delta_before = (
                            guided_delta.float().abs() * outside_mask.float()
                        ).sum() / outside_sum.float()
                        outside_delta_after = (
                            (noise_pred - draft_prediction).float().abs() * outside_mask.float()
                        ).sum() / outside_sum.float()
                        print(
                            f"[draft-vector-anchor] step={i} "
                            f"inside_delta={inside_delta.item():.8f} "
                            f"outside_delta_before={outside_delta_before.item():.8f} "
                            f"outside_delta_after={outside_delta_after.item():.8f}",
                            flush=True,
                        )

                # compute the previous noisy sample x_t -> x_t-1
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

        for handle in attn_avg_hook_handles:
            handle.remove()
        for handle in draft_feature_hook_handles:
            handle.remove()
        for attention_layer, processor in attn_avg_processor_backups:
            attention_layer.set_processor(processor)
        if draft_feature_cache_out_path:
            if draft_feature_cache_kind in {
                "attention_kv",
                "attention_value_delta",
            }:
                if draft_feature_cache_kind == "attention_kv":
                    required_fields = {"key", "value"}
                    cache_label = "K/V"
                else:
                    required_fields = {
                        "target_flat",
                        "delta",
                        "confidence",
                    }
                    cache_label = "value-delta"
                incomplete_entries = {}
                for cache_key, cache_entry in draft_feature_cache_state["features"].items():
                    if not isinstance(cache_entry, dict):
                        incomplete_entries[cache_key] = sorted(required_fields)
                    elif set(cache_entry) != required_fields:
                        incomplete_entries[cache_key] = sorted(
                            required_fields - set(cache_entry)
                        )
                if incomplete_entries:
                    raise ValueError(
                        f"Draft {cache_label} cache capture produced incomplete entries: "
                        f"{incomplete_entries}"
                    )
                for step_index in range(
                    geometry_transport_start,
                    geometry_transport_end + 1,
                ):
                    for layer_index in geometry_transport_layers:
                        suffix = f":{layer_index}:{step_index}"
                        matching_keys = [
                            key
                            for key in draft_feature_cache_state["features"]
                            if key.endswith(suffix)
                        ]
                        if len(matching_keys) != 1:
                            raise ValueError(
                                f"Draft {cache_label} cache capture must produce exactly one "
                                f"active transformer entry for {suffix}, found "
                                f"{matching_keys}"
                            )
            cache_path = Path(draft_feature_cache_out_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_metadata = {
                "format_version": (
                    3
                    if draft_feature_cache_kind == "attention_value_delta"
                    else 2
                ),
                "cache_kind": draft_feature_cache_kind,
                "token_grid": list(token_grid),
                "anchor_token_times": draft_feature_cache_state["anchor_token_times"],
                "layers": geometry_transport_layers,
                "num_inference_steps": num_inference_steps,
                "run_fingerprint": draft_feature_cache_fingerprint,
                "geometry_map_sha256": geometry_transport_state[
                    "map_sha256"
                ],
                "geometry_transport_interval": [
                    geometry_transport_start,
                    geometry_transport_end,
                ],
            }
            torch.save(
                {
                    "metadata": cache_metadata,
                    "features": draft_feature_cache_state["features"],
                },
                cache_path,
            )
            print(
                f"[draft-feature-cache] saved={cache_path} "
                f"entries={len(draft_feature_cache_state['features'])}",
                flush=True,
            )
        if draft_vector_cache_out_path:
            vector_cache_path = Path(draft_vector_cache_out_path)
            vector_cache_path.parent.mkdir(parents=True, exist_ok=True)
            vector_cache_metadata = {
                "format_version": 1,
                "num_inference_steps": num_inference_steps,
                "latent_shape": list(latents.shape),
                "guidance_scale": float(guidance_scale),
                "prompt": prompt,
            }
            torch.save(
                {
                    "metadata": vector_cache_metadata,
                    "predictions": draft_vector_cache_state["predictions"],
                },
                vector_cache_path,
            )
            print(
                f"[draft-vector-cache] saved={vector_cache_path} "
                f"entries={len(draft_vector_cache_state['predictions'])}",
                flush=True,
            )

        self._current_timestep = None

        if self.config.expand_timesteps:
            if (
                draft_geometry_condition_alpha > 0.0
                and draft_geometry_condition_final_blend
            ):
                final_conditioning_weight = (
                    draft_geometry_condition_alpha
                    * geometry_condition_weight_gate
                ).to(base_first_frame_mask.dtype)
                final_first_frame_mask = base_first_frame_mask * (
                    1.0 - final_conditioning_weight
                )
                latents = (
                    (1 - final_first_frame_mask) * geometry_condition
                    + final_first_frame_mask * latents
                )
            else:
                latents = (
                    (1 - base_first_frame_mask) * base_condition
                    + base_first_frame_mask * latents
                )

        if draft_clean_latent_cache_out_path:
            clean_latent_path = Path(draft_clean_latent_cache_out_path)
            clean_latent_path.parent.mkdir(parents=True, exist_ok=True)
            clean_latent_metadata = {
                "format_version": 1,
                "num_inference_steps": num_inference_steps,
                "latent_shape": list(latents.shape),
                "guidance_scale": float(guidance_scale),
                "prompt": prompt,
            }
            torch.save(
                {
                    "metadata": clean_latent_metadata,
                    "latents": latents.detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous(),
                },
                clean_latent_path,
            )
            print(
                f"[draft-clean-latent] saved={clean_latent_path} "
                f"shape={list(latents.shape)}",
                flush=True,
            )

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
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
