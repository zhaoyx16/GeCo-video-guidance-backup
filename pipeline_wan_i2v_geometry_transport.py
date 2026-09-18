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
import time
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

from geometry_transport import build_geometry_transport_map


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
        # Geometry-validated transport is separate from the older feature-match modes.
        geometry_transport_alpha: float = 0.0,
        geometry_transport_layers: List[int] = None,
        geometry_transport_mode: str = "shadow",
        geometry_anchor_step: int = 30,
        geometry_transport_start: int | None = None,
        geometry_transport_end: int | None = None,
        geometry_frame_indices: List[int] = None,
        geometry_memory_lookback: int = 3,
        geometry_confidence_percentile: float = 20.0,
        geometry_confidence_floor: float = 0.2,
        geometry_depth_relative_threshold: float = 0.15,
        geometry_transport_cond_only: bool = True,
        geometry_transport_debug: bool = False,
        geometry_debug_dir: str | None = None,
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
        attn_avg_state = {"step": -1, "branch": "cond", "printed": set()}
        geometry_transport_layers = [] if geometry_transport_layers is None else sorted(set(geometry_transport_layers))
        geometry_transport_state = {
            "ready": False,
            "source_time": None,
            "source_index": None,
            "confidence": None,
            "pair_stats": [],
            "anchor_step": None,
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
        if geometry_transport_mode not in {"shadow", "value_residual", "kv_memory", "kv_transport"}:
            raise ValueError("geometry_transport_mode must be shadow, value_residual, kv_memory, or kv_transport")
        if geometry_memory_lookback < 1:
            raise ValueError("geometry_memory_lookback must be positive")
        if not 0.0 <= geometry_confidence_percentile <= 100.0:
            raise ValueError("geometry_confidence_percentile must lie in [0, 100]")
        if geometry_confidence_floor < 0.0:
            raise ValueError("geometry_confidence_floor must be non-negative")
        if geometry_depth_relative_threshold <= 0.0:
            raise ValueError("geometry_depth_relative_threshold must be positive")
        if geometry_transport_layers and not geometry_frame_indices:
            raise ValueError("geometry_frame_indices is required when geometry_transport_layers is set")
        if geometry_transport_layers and attn_avg_alpha > 0.0:
            raise ValueError("Do not combine geometry transport with feature-match transport in one run")
        if geometry_frame_indices is not None:
            geometry_frame_indices = [int(x) for x in geometry_frame_indices]
            if len(geometry_frame_indices) < 2 or geometry_frame_indices != sorted(set(geometry_frame_indices)):
                raise ValueError("geometry_frame_indices must be a strictly increasing list with at least two frames")
            if geometry_frame_indices[0] < 0 or geometry_frame_indices[-1] >= num_frames:
                raise ValueError("geometry_frame_indices must lie within the generated video")

        attn_avg_end = num_inference_steps - 1 if attn_avg_end is None else attn_avg_end
        if attn_avg_start < 0 or attn_avg_end < attn_avg_start or attn_avg_end >= num_inference_steps:
            raise ValueError("attn_avg_start/end must be a valid inclusive interval of sampling-step indices")

        geometry_transport_start = geometry_anchor_step + 1 if geometry_transport_start is None else geometry_transport_start
        geometry_transport_end = num_inference_steps - 1 if geometry_transport_end is None else geometry_transport_end
        if geometry_transport_layers:
            if geometry_anchor_step < 0 or geometry_anchor_step >= num_inference_steps:
                raise ValueError("geometry_anchor_step must be a valid sampling step")
            if geometry_transport_start < geometry_anchor_step + 1:
                raise ValueError("geometry transport must start after its geometry anchor step")
            if geometry_transport_end < geometry_transport_start or geometry_transport_end >= num_inference_steps:
                raise ValueError("geometry transport start/end must be a valid inclusive interval")

        attn_avg_processor_backups = []
        if (attn_avg_alpha > 0.0 and attn_avg_layers) or geometry_transport_layers:
            p_t, p_h, p_w = patch_size
            token_grid = (
                latents.shape[2] // p_t,
                latents.shape[3] // p_h,
                latents.shape[4] // p_w,
            )

            def _frames01_to_vggt_input(frames_01: torch.Tensor, target_width: int = 518, patch_size: int = 14) -> torch.Tensor:
                """Use the same resize/crop convention as the existing GeCo metric."""
                x = frames_01.permute(0, 3, 1, 2).contiguous().float()
                _, _, height_src, width_src = x.shape
                height_scaled = max(
                    patch_size,
                    int(round((height_src * target_width / width_src) / patch_size) * patch_size),
                )
                x = F.interpolate(x, size=(height_scaled, target_width), mode="bilinear", align_corners=False)
                if height_scaled > target_width:
                    top = (height_scaled - target_width) // 2
                    x = x[:, :, top : top + target_width, :]
                return x.unsqueeze(0)

            def _decode_geometry_frames(x0_pred: torch.Tensor) -> torch.Tensor:
                """Decode only sparse requested frames, without autograd or a full-video VAE pass."""
                temporal_scale = int(getattr(self, "vae_scale_factor_temporal", 4))
                latent_frames = x0_pred.shape[2]
                vae_dtype = self.vae.dtype
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(x0_pred.device, vae_dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(x0_pred.device, vae_dtype)
                decoded_frames = []
                for frame_index in geometry_frame_indices:
                    center_latent = 0 if frame_index == 0 else min(latent_frames - 1, (frame_index - 1) // temporal_scale + 1)
                    chunk_start = max(0, center_latent - 1)
                    chunk_end = min(latent_frames, center_latent + 2)
                    z_chunk = x0_pred[:, :, chunk_start:chunk_end].contiguous().to(vae_dtype)
                    z_chunk = z_chunk / latents_std + latents_mean
                    decoded_chunk = self.vae.decode(z_chunk, return_dict=False)[0]
                    frames_chunk = ((decoded_chunk.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0, 1)
                    chunk_first_frame = 0 if chunk_start == 0 else 1 + (chunk_start - 1) * temporal_scale
                    local_frame = int(max(0, min(frame_index - chunk_first_frame, frames_chunk.shape[1] - 1)))
                    decoded_frames.append(frames_chunk[0, local_frame].cpu())
                    del z_chunk, decoded_chunk, frames_chunk
                return torch.stack(decoded_frames, dim=0)

            def _build_geometry_transport_cache(noise_prediction: torch.Tensor, current_latents: torch.Tensor, timestep: torch.Tensor) -> None:
                """Estimate sparse static geometry once from a late x0 prediction and cache token maps."""
                from vggt.models.vggt import VGGT
                from utils import vggt_infer

                started = time.perf_counter()
                with torch.no_grad():
                    if self.scheduler.step_index is None:
                        self.scheduler._init_step_index(timestep)
                    x0_pred = self.scheduler.convert_model_output(
                        noise_prediction.float().detach(), sample=current_latents.float()
                    )
                    if self.config.expand_timesteps:
                        x0_pred = (1 - first_frame_mask.float()) * condition.float() + first_frame_mask.float() * x0_pred.float()
                    frames_01 = _decode_geometry_frames(x0_pred)
                    del x0_pred

                    if geometry_debug_dir:
                        debug_dir = Path(geometry_debug_dir)
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        thumbnails = []
                        for frame_index, frame_01 in zip(geometry_frame_indices, frames_01):
                            image = PIL.Image.fromarray(
                                frame_01.mul(255).round().to(torch.uint8).numpy()
                            )
                            image.thumbnail((320, 320))
                            thumbnails.append(image)
                            image.save(debug_dir / f"x0_step{attn_avg_state['step']:02d}_frame{frame_index:03d}.png")
                        if thumbnails:
                            width = max(image.width for image in thumbnails)
                            height = max(image.height for image in thumbnails)
                            sheet = PIL.Image.new("RGB", (width * len(thumbnails), height))
                            for column, image in enumerate(thumbnails):
                                sheet.paste(image, (column * width, 0))
                            sheet.save(debug_dir / f"x0_step{attn_avg_state['step']:02d}_contact_sheet.png")

                    device = current_latents.device
                    compute_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
                    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
                    geometry = vggt_infer(
                        vggt_model,
                        _frames01_to_vggt_input(frames_01),
                        upsample_size=tuple(frames_01.shape[1:3]),
                        point_prediction=False,
                        compute_dtype=compute_dtype,
                        device=device,
                        enable_grad=False,
                    )
                    transport = build_geometry_transport_map(
                        intrinsic=geometry["intrinsic"],
                        extrinsic=geometry["extrinsic"],
                        depth_map=geometry["depth_map"],
                        confidence_map=geometry["vggt_conf"],
                        selected_video_frames=geometry_frame_indices,
                        token_grid=token_grid,
                        temporal_scale=int(getattr(self, "vae_scale_factor_temporal", 4)),
                        confidence_percentile=geometry_confidence_percentile,
                        confidence_floor=geometry_confidence_floor,
                        depth_relative_threshold=geometry_depth_relative_threshold,
                        memory_lookback=geometry_memory_lookback,
                    )
                    geometry_transport_state["source_time"] = transport.source_time
                    geometry_transport_state["source_index"] = transport.source_index
                    geometry_transport_state["confidence"] = transport.confidence
                    geometry_transport_state["pair_stats"] = transport.pair_stats
                    geometry_transport_state["ready"] = True
                    del vggt_model, geometry, transport, frames_01
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                geometry_transport_state["build_seconds"] = time.perf_counter() - started
                coverage = geometry_transport_state["confidence"].amax(dim=-1).gt(0).float().mean().item()
                print(
                    f"[geometry-transport] cached at step={attn_avg_state['step']} "
                    f"coverage={coverage:.3f} pairs={len(geometry_transport_state['pair_stats'])} "
                    f"seconds={geometry_transport_state['build_seconds']:.2f}",
                    flush=True,
                )

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
                confidence = geometry_transport_state["confidence"][..., 0]
                source_flat = source_time * spatial_tokens + source_index
                values_flat = value.reshape(batch_size, num_tokens_t * spatial_tokens, value_channels)
                matched = values_flat.gather(
                    1,
                    source_flat.reshape(1, -1).expand(batch_size, -1).unsqueeze(-1).expand(-1, -1, value_channels),
                ).reshape(batch_size, num_tokens_t - 1, spatial_tokens, value_channels)
                return matched, confidence.unsqueeze(0).expand(batch_size, -1, -1, -1, -1), source_index, source_time

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

            def _apply_geometry_kv_memory(
                query_with_rope: torch.Tensor,
                key_before_rope: torch.Tensor,
                value_heads: torch.Tensor,
                attended: torch.Tensor,
                rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
            ) -> torch.Tensor:
                """Parallel K/V memory over only geometry-validated source candidates.

                The native full self-attention is retained. This adds a small
                per-target cross-attention residual over up to K reprojected
                source tokens plus a learned-free null option, so new/occluded
                regions can decline memory rather than receiving copied content.
                """
                if not geometry_transport_state["ready"]:
                    return attended
                if rotary_emb is None:
                    raise ValueError("geometry kv_memory requires Wan rotary embeddings")
                batch_size, sequence_length, heads, head_dim = query_with_rope.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Geometry K/V memory received an unexpected Wan token sequence length")

                source_time = geometry_transport_state["source_time"]
                source_index = geometry_transport_state["source_index"]
                confidence = geometry_transport_state["confidence"]
                memory_slots = source_time.shape[-1]
                query_grid = query_with_rope.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                key_grid = key_before_rope.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                value_grid = value_heads.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                attended_grid = attended.reshape(batch_size, num_tokens_t, spatial_tokens, heads, head_dim)
                key_flat = key_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
                value_flat = value_grid.reshape(batch_size, num_tokens_t * spatial_tokens, heads, head_dim)
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
                    source_value = value_flat[:, source_flat].reshape(
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
                    confidence_logits = confidence_t.reshape(1, spatial_tokens, 1, memory_slots).clamp_min(1e-8).log()
                    scores = scores + confidence_logits
                    null_logits = torch.zeros_like(scores[..., :1])
                    weights = torch.softmax(torch.cat([scores, null_logits], dim=-1), dim=-1)[..., :-1]
                    source_value = source_value.permute(0, 1, 3, 2, 4)
                    memory_value = (weights.unsqueeze(-1) * source_value.float()).sum(dim=3)
                    current_value = value_grid[:, target_time].float()
                    attended_grid[:, target_time] = (
                        attended_grid[:, target_time].float()
                        + geometry_transport_alpha * (memory_value - current_value)
                    ).to(attended.dtype)

                    if geometry_transport_debug and target_time not in geometry_transport_state["printed"]:
                        geometry_transport_state["printed"].add(target_time)
                        accepted = (weights.sum(dim=-1) > 0.5).float().mean().item()
                        print(
                            f"[geometry-transport] target_latent={target_time} mode=kv_memory "
                            f"coverage={(confidence_t.amax(dim=-1) > 0).float().mean().item():.3f} "
                            f"memory_gate={accepted:.3f}",
                            flush=True,
                        )
                return attended_grid.reshape_as(attended)

            def _transport_geometry_kv(
                key: torch.Tensor,
                value: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Mix visible source K/V into target coordinates before native attention.

                This differs from ``kv_memory``: it changes the K/V consumed
                by Wan's own full self-attention. Source K is mixed before
                RoPE, so the standard target RoPE below locates it at the
                reprojected target coordinate.
                """
                if not geometry_transport_state["ready"]:
                    return key, value
                batch_size, sequence_length, key_channels = key.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                spatial_tokens = height_tokens * width_tokens
                if sequence_length != num_tokens_t * spatial_tokens:
                    raise ValueError("Geometry K/V transport received an unexpected Wan token sequence length")

                source_time_all = geometry_transport_state["source_time"]
                source_index_all = geometry_transport_state["source_index"]
                confidence_all = geometry_transport_state["confidence"]
                best_slot = confidence_all.argmax(dim=-1)
                best_confidence = confidence_all.gather(-1, best_slot.unsqueeze(-1)).squeeze(-1)
                source_time = source_time_all.gather(
                    -1, best_slot.reshape(num_tokens_t - 1, spatial_tokens, 1)
                ).squeeze(-1)
                source_index = source_index_all.gather(
                    -1, best_slot.reshape(num_tokens_t - 1, spatial_tokens, 1)
                ).squeeze(-1)
                source_flat = source_time * spatial_tokens + source_index

                value_channels = value.shape[-1]
                key_grid = key.reshape(batch_size, num_tokens_t, spatial_tokens, key_channels)
                value_grid = value.reshape(batch_size, num_tokens_t, spatial_tokens, value_channels)
                key_flat = key_grid.reshape(batch_size, num_tokens_t * spatial_tokens, key_channels)
                value_flat = value_grid.reshape(batch_size, num_tokens_t * spatial_tokens, value_channels)
                gather_index = source_flat.reshape(1, -1).expand(batch_size, -1).unsqueeze(-1)
                source_key = key_flat.gather(1, gather_index.expand(-1, -1, key_channels)).reshape(
                    batch_size, num_tokens_t - 1, spatial_tokens, key_channels
                )
                source_value = value_flat.gather(1, gather_index.expand(-1, -1, value_channels)).reshape(
                    batch_size, num_tokens_t - 1, spatial_tokens, value_channels
                )
                blend = (geometry_transport_alpha * best_confidence).reshape(
                    1, num_tokens_t - 1, spatial_tokens, 1
                )
                mixed_key = key_grid.clone()
                mixed_value = value_grid.clone()
                mixed_key[:, 1:] = (
                    key_grid[:, 1:].float() * (1.0 - blend) + source_key.float() * blend
                ).to(key.dtype)
                mixed_value[:, 1:] = (
                    value_grid[:, 1:].float() * (1.0 - blend) + source_value.float() * blend
                ).to(value.dtype)
                if geometry_transport_debug and "kv_transport" not in geometry_transport_state["printed"]:
                    geometry_transport_state["printed"].add("kv_transport")
                    print(
                        "[geometry-transport] mode=kv_transport "
                        f"coverage={(best_confidence > 0).float().mean().item():.3f} "
                        f"mean_blend={blend.float().mean().item():.5f} "
                        f"max_blend={blend.float().amax().item():.5f}",
                        flush=True,
                    )
                return mixed_key.reshape_as(key), mixed_value.reshape_as(value)

            class _CorrespondenceKVProcessor:
                def __init__(self, base_processor, layer_idx: int):
                    self.base_processor = base_processor
                    self.layer_idx = layer_idx

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
                    )
                    if not feature_match_active and not geometry_active:
                        return self.base_processor(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

                    # attn1 is self-attention. Never silently alter a cross-attention path.
                    if encoder_hidden_states is not None or attn.add_k_proj is not None:
                        return self.base_processor(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

                    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                    matched_values = None
                    match_confidence = None
                    match_similarity = None
                    reciprocal = None
                    match_source_index = None
                    match_source_time = None
                    geometry_value_residual = geometry_active and geometry_transport_mode == "value_residual"
                    geometry_kv_memory = geometry_active and geometry_transport_mode == "kv_memory"
                    geometry_kv_transport = geometry_active and geometry_transport_mode == "kv_transport"
                    if geometry_value_residual:
                        (
                            matched_values,
                            match_confidence,
                            match_source_index,
                            match_source_time,
                        ) = _matched_geometry_values(value)
                    elif geometry_kv_transport:
                        key, value = _transport_geometry_kv(key, value)
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

                    if geometry_kv_memory:
                        hidden_states = _apply_geometry_kv_memory(
                            query,
                            key_before_rope,
                            value,
                            hidden_states,
                            rotary_emb,
                        )

                    if matched_values is not None and match_confidence is not None:
                        num_tokens_t, height_tokens, width_tokens = token_grid
                        spatial_tokens = height_tokens * width_tokens
                        attended_grid = hidden_states.reshape(
                            batch_size, num_tokens_t, spatial_tokens, attn.heads, -1
                        )
                        matched_value_grid = matched_values.unflatten(-1, (attn.heads, -1))
                        transport_alpha = geometry_transport_alpha if geometry_value_residual else attn_avg_alpha
                        blend = (
                            transport_alpha * match_confidence.reshape(
                                batch_size, num_tokens_t - 1, spatial_tokens, 1, 1
                            )
                        )
                        mixed_grid = attended_grid.clone()
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

            processor_layers = set()
            if attn_avg_alpha > 0.0 and attn_avg_layers and attn_avg_mode in {
                "query_match_prev", "key_match_prev", "kv_match_prev", "value_residual_prev",
                "c2f_value_residual_prev", "c2f_value_residual_anchor", "c2f_value_residual_memory",
            }:
                processor_layers.update(attn_avg_layers)
            if geometry_transport_mode != "shadow":
                processor_layers.update(geometry_transport_layers)
            hook_layers = set(attn_avg_layers or []) if attn_avg_alpha > 0.0 else set()

            for model in (self.transformer, self.transformer_2):
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
                        attention_layer.set_processor(_CorrespondenceKVProcessor(attention_layer.processor, layer_idx))
                    elif layer_idx in hook_layers:
                        handle = attention_layer.register_forward_hook(_make_attention_hook(layer_idx))
                        attn_avg_hook_handles.append(handle)

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
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
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

                if geometry_transport_layers and not geometry_transport_state["ready"] and i == geometry_anchor_step:
                    _build_geometry_transport_cache(noise_pred, latents, t)

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
        for attention_layer, processor in attn_avg_processor_backups:
            attention_layer.set_processor(processor)

        self._current_timestep = None

        if self.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

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
