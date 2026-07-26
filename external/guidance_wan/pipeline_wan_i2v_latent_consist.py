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
from typing import Any, Callable

import PIL
import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
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
        latent_consist_lr: float = 0.0,
        latent_consist_layer: int = 15,
        latent_consist_start_step: int = 5,
        latent_consist_end_step: int = 45,
        latent_consist_mode: str = "off",
        latent_consist_match_radius: int = 2,
        latent_consist_match_confidence: float = 0.55,
        latent_consist_static_coherence: float = 1.0,
        latent_consist_qk_require_mutual: bool = False,
        latent_consist_anchor_confidence: float = 0.45,
        latent_consist_tracklet_horizon: int = 3,
        latent_consist_tracklet_confidence: float = 0.40,
        latent_consist_descriptor_dim: int = 64,
        latent_consist_cond_only: bool = True,
        latent_consist_noise_rho: float = 0.0,
        latent_consist_max_relative_delta: float = 0.0,
        latent_consist_debug: bool = False,
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
            latent_consist_lr (`float`, defaults to `0.0`):
                Learning rate for latent-space frame consistency guidance. `0.0` disables this guidance.
            latent_consist_layer (`int`, defaults to `15`):
                Transformer block index whose self-attention output is used as the feature map.
            latent_consist_start_step (`int`, defaults to `5`):
                First denoising step index where latent consistency guidance is active.
            latent_consist_end_step (`int`, defaults to `45`):
                Last denoising step index where latent consistency guidance is active.
            latent_consist_mode (`str`, defaults to `"off"`):
                One of `off`, `same_position_l2`, `matched_cosine`, `match_prev`,
                `static_match_prev`, `match_prev_input`, `static_match_prev_input`,
                `static_anchor_match_prev_input`, or `noise_ar1`. The matching modes
                use frozen internal transformer features and do not decode RGB frames.
            latent_consist_match_radius (`int`, defaults to `2`):
                Local token-search radius used by correspondence-aware modes.
            latent_consist_match_confidence (`float`, defaults to `0.55`):
                Minimum cosine similarity required before feature propagation.
            latent_consist_static_coherence (`float`, defaults to `1.0`):
                Maximum local flow residual, in token units, for `static_match_prev`.
            latent_consist_qk_require_mutual (`bool`, defaults to `False`):
                Whether projected Q/K modes additionally require a reciprocal local match. This is
                disabled by default because Q and K use different learned projection spaces.
            latent_consist_anchor_confidence (`float`, defaults to `0.45`):
                Minimum descriptor cosine similarity to the conditioned first-frame anchor
                after composing trusted local correspondences. Only used by
                `static_anchor_match_prev_input`.
            latent_consist_descriptor_dim (`int`, defaults to `64`):
                Number of pooled attention channels used for correspondence matching.
            latent_consist_cond_only (`bool`, defaults to `True`):
                Apply direct feature propagation only to the conditional CFG branch.
            latent_consist_noise_rho (`float`, defaults to `0.0`):
                AR(1) temporal correlation applied once to the initial latent noise in
                `noise_ar1` mode.
            latent_consist_max_relative_delta (`float`, defaults to `0.0`):
                Optional per-guidance-step cap on the mean latent update magnitude for
                gradient-based modes. `0.0` disables capping.
            latent_consist_debug (`bool`, defaults to `False`):
                Print compact per-step correspondence diagnostics.

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

        allowed_latent_consist_modes = {
            "off",
            "same_position_l2",
            "matched_cosine",
            "match_prev",
            "static_match_prev",
            "match_prev_input",
            "static_match_prev_input",
            "static_track_prev_input",
            "static_anchor_match_prev_input",
            "static_tracklet_match_prev_input",
            "static_match_prev_kv",
            "static_qk_match_prev_kv",
            "qk_probe",
            "noise_ar1",
        }
        if latent_consist_mode not in allowed_latent_consist_modes:
            raise ValueError(f"latent_consist_mode must be one of {sorted(allowed_latent_consist_modes)}")
        if latent_consist_lr < 0.0:
            raise ValueError("latent_consist_lr must be non-negative")
        if latent_consist_match_radius < 0:
            raise ValueError("latent_consist_match_radius must be non-negative")
        if not -1.0 <= latent_consist_match_confidence < 1.0:
            raise ValueError("latent_consist_match_confidence must lie in [-1, 1)")
        if latent_consist_static_coherence <= 0.0:
            raise ValueError("latent_consist_static_coherence must be positive")
        if not -1.0 <= latent_consist_anchor_confidence < 1.0:
            raise ValueError("latent_consist_anchor_confidence must lie in [-1, 1)")
        if latent_consist_tracklet_horizon < 1:
            raise ValueError("latent_consist_tracklet_horizon must be positive")
        if not -1.0 <= latent_consist_tracklet_confidence < 1.0:
            raise ValueError("latent_consist_tracklet_confidence must lie in [-1, 1)")
        if latent_consist_descriptor_dim < 1:
            raise ValueError("latent_consist_descriptor_dim must be positive")
        if latent_consist_max_relative_delta < 0.0:
            raise ValueError("latent_consist_max_relative_delta must be non-negative")
        if latent_consist_mode not in {"off", "noise_ar1"} and (
            latent_consist_start_step < 0
            or latent_consist_end_step < latent_consist_start_step
            or latent_consist_end_step >= num_inference_steps
        ):
            raise ValueError("latent_consist_start_step/end_step must be a valid inclusive sampling-step interval")

        gradient_modes = {"same_position_l2", "matched_cosine"}
        output_injection_modes = {"match_prev", "static_match_prev"}
        input_injection_modes = {
            "match_prev_input",
            "static_match_prev_input",
            "static_track_prev_input",
            "static_anchor_match_prev_input",
            "static_tracklet_match_prev_input",
        }
        kv_injection_modes = {"static_match_prev_kv"}
        qk_injection_modes = {"static_qk_match_prev_kv"}
        qk_probe_modes = {"qk_probe"}
        qk_modes = qk_injection_modes | qk_probe_modes
        injection_modes = output_injection_modes | input_injection_modes | kv_injection_modes | qk_injection_modes
        is_gradient_mode = latent_consist_mode in gradient_modes and latent_consist_lr > 0.0
        is_injection_mode = latent_consist_mode in injection_modes and latent_consist_lr > 0.0
        is_qk_probe_mode = latent_consist_mode in qk_probe_modes
        if is_injection_mode and latent_consist_lr > 1.0:
            raise ValueError("latent_consist_lr is a blend strength for injection modes and must lie in [0, 1]")
        if latent_consist_mode == "noise_ar1" and not 0.0 <= latent_consist_noise_rho < 1.0:
            raise ValueError("latent_consist_noise_rho must lie in [0, 1)")

        latent_consist_losses = []
        latent_consist_gradient_stats = []
        latent_consist_hook_handles = []
        latent_consist_capture = {"enabled": False, "features": None, "inputs": None}
        latent_consist_state = {"step": -1, "branch": "none", "printed": set(), "diagnostics": []}
        latent_consist_kv_cache = {"key": None, "mixed": None}
        latent_consist_qk_cache = {
            "q_key": None,
            "query": None,
            "kv_key": None,
            "source_index": None,
            "blend": None,
        }

        # A cheap noise-only baseline: preserve N(0, I) marginals while correlating adjacent latent frames.
        if latent_consist_mode == "noise_ar1" and latent_consist_noise_rho > 0.0:
            innovation_scale = (1.0 - latent_consist_noise_rho**2) ** 0.5
            correlated_latents = [latents[:, :, 0]]
            for frame_idx in range(1, latents.shape[2]):
                correlated_latents.append(
                    latent_consist_noise_rho * correlated_latents[-1] + innovation_scale * latents[:, :, frame_idx]
                )
            latents = torch.stack(correlated_latents, dim=2)

        if is_gradient_mode or is_injection_mode or is_qk_probe_mode:
            p_t, p_h, p_w = patch_size
            token_grid = (latents.shape[2] // p_t, latents.shape[3] // p_h, latents.shape[4] // p_w)

            # The transformer remains frozen. Gradient modes differentiate only with respect to the sampled latents.
            if is_gradient_mode:
                self.transformer.requires_grad_(False)
                if getattr(self, "transformer_2", None) is not None:
                    self.transformer_2.requires_grad_(False)

            def _reduced_descriptors(hidden: torch.Tensor) -> torch.Tensor:
                channels = hidden.shape[-1]
                descriptor_dim = min(latent_consist_descriptor_dim, channels)
                group_size = channels // descriptor_dim
                usable_channels = descriptor_dim * group_size
                descriptors = hidden[..., :usable_channels].float().reshape(
                    *hidden.shape[:-1], descriptor_dim, group_size
                )
                descriptors = descriptors.mean(dim=-1)
                return F.normalize(descriptors, dim=-1, eps=1e-6)

            def _local_correspondence(
                hidden: torch.Tensor,
                input_hidden: torch.Tensor,
                require_static: bool,
            ):
                """Match each current-frame token to a local previous-frame token using detached descriptors."""
                batch, num_tokens_t, height_tokens, width_tokens, channels = hidden.shape
                if num_tokens_t < 2:
                    raise ValueError("Latent consistency needs at least two temporal tokens")

                descriptors = _reduced_descriptors(input_hidden.detach())
                previous_descriptors = descriptors[:, :-1]
                current_descriptors = descriptors[:, 1:]
                descriptor_dim = descriptors.shape[-1]
                kernel_size = 2 * latent_consist_match_radius + 1
                num_pairs = num_tokens_t - 1

                previous_4d = previous_descriptors.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_pairs, descriptor_dim, height_tokens, width_tokens
                )
                patches = F.unfold(previous_4d, kernel_size=kernel_size, padding=latent_consist_match_radius)
                patches = patches.reshape(
                    batch, num_pairs, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                ).permute(0, 1, 4, 5, 3, 2)
                similarity = (current_descriptors.unsqueeze(-2) * patches).sum(dim=-1)
                best_similarity, best_index = similarity.max(dim=-1)

                offset_y = torch.div(best_index, kernel_size, rounding_mode="floor") - latent_consist_match_radius
                offset_x = best_index.remainder(kernel_size) - latent_consist_match_radius
                base_y = torch.arange(height_tokens, device=hidden.device).view(1, 1, height_tokens, 1)
                base_x = torch.arange(width_tokens, device=hidden.device).view(1, 1, 1, width_tokens)
                source_y = (base_y + offset_y).clamp(0, height_tokens - 1)
                source_x = (base_x + offset_x).clamp(0, width_tokens - 1)
                source_index = (source_y * width_tokens + source_x).reshape(batch, num_pairs, -1)

                previous_output = hidden[:, :-1].reshape(batch, num_pairs, height_tokens * width_tokens, channels)
                matched_output = previous_output.gather(
                    2, source_index.unsqueeze(-1).expand(-1, -1, -1, channels)
                ).reshape(batch, num_pairs, height_tokens, width_tokens, channels)

                valid = best_similarity >= latent_consist_match_confidence
                mutual = torch.ones_like(valid, dtype=torch.bool)
                coherent = torch.ones_like(valid, dtype=torch.bool)

                if require_static:
                    current_4d = current_descriptors.permute(0, 1, 4, 2, 3).reshape(
                        batch * num_pairs, descriptor_dim, height_tokens, width_tokens
                    )
                    reverse_patches = F.unfold(
                        current_4d, kernel_size=kernel_size, padding=latent_consist_match_radius
                    )
                    reverse_patches = reverse_patches.reshape(
                        batch, num_pairs, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                    ).permute(0, 1, 4, 5, 3, 2)
                    reverse_similarity = (previous_descriptors.unsqueeze(-2) * reverse_patches).sum(dim=-1)
                    reverse_best_similarity, reverse_best_index = reverse_similarity.max(dim=-1)
                    reverse_offset_y = (
                        torch.div(reverse_best_index, kernel_size, rounding_mode="floor")
                        - latent_consist_match_radius
                    )
                    reverse_offset_x = reverse_best_index.remainder(kernel_size) - latent_consist_match_radius
                    reverse_y = (base_y + reverse_offset_y).clamp(0, height_tokens - 1)
                    reverse_x = (base_x + reverse_offset_x).clamp(0, width_tokens - 1)
                    reverse_index = (reverse_y * width_tokens + reverse_x).reshape(batch, num_pairs, -1)
                    query_index = (base_y * width_tokens + base_x).expand(
                        batch, num_pairs, height_tokens, width_tokens
                    ).reshape(batch, num_pairs, -1)
                    mutual = reverse_index.gather(2, source_index) == query_index
                    mutual = mutual.reshape(batch, num_pairs, height_tokens, width_tokens)
                    mutual = mutual & (reverse_best_similarity >= latent_consist_match_confidence)

                    flow = torch.stack(
                        [(source_y - base_y).float(), (source_x - base_x).float()], dim=2
                    ).reshape(batch * num_pairs, 2, height_tokens, width_tokens)
                    local_flow = F.avg_pool2d(flow, kernel_size=3, stride=1, padding=1)
                    flow_residual = torch.linalg.vector_norm(flow - local_flow, dim=1).reshape(
                        batch, num_pairs, height_tokens, width_tokens
                    )
                    coherent = flow_residual <= latent_consist_static_coherence
                    valid = valid & mutual & coherent

                confidence = (
                    (best_similarity - latent_consist_match_confidence)
                    / (1.0 - latent_consist_match_confidence + 1e-6)
                ).clamp(0.0, 1.0)
                valid_count = valid.float().sum().clamp_min(1.0)
                diagnostics = {
                    "match_cos_mean": best_similarity.detach().float().mean(),
                    "match_cos_p10": torch.quantile(best_similarity.detach().float(), 0.1),
                    "match_cos_p90": torch.quantile(best_similarity.detach().float(), 0.9),
                    "coverage": valid.detach().float().mean(),
                    "mutual_coverage": mutual.detach().float().mean(),
                    "coherent_coverage": coherent.detach().float().mean(),
                    "confidence_valid_mean": (confidence.detach().float() * valid.float()).sum() / valid_count,
                }
                return matched_output, valid, confidence, diagnostics, source_index

            def _gather_previous_features(feature: torch.Tensor, source_index: torch.Tensor) -> torch.Tensor:
                """Gather previous-frame features at local correspondence indices."""
                batch, num_tokens_t, height_tokens, width_tokens, channels = feature.shape
                previous = feature[:, :-1].reshape(
                    batch, num_tokens_t - 1, height_tokens * width_tokens, channels
                )
                return previous.gather(
                    2, source_index.unsqueeze(-1).expand(-1, -1, -1, channels)
                ).reshape(batch, num_tokens_t - 1, height_tokens, width_tokens, channels)

            def _local_qk_correspondence(query: torch.Tensor, key: torch.Tensor):
                """Find locally coherent static matches using Wan's projected Q/K descriptors.

                This follows the representation choice supported by DiffTrack: temporal matches are
                identified from query-key similarity, while the frozen model still decides how the
                resulting keys and values are consumed by full 3D attention.
                """
                if query.shape != key.shape or query.ndim != 5:
                    raise ValueError("Projected Q and K must have equal [B, T, H, W, C] shapes")

                batch, num_tokens_t, height_tokens, width_tokens, channels = query.shape
                if num_tokens_t < 2:
                    raise ValueError("Q/K consistency needs at least two temporal tokens")

                query_descriptors = _reduced_descriptors(query.detach())
                key_descriptors = _reduced_descriptors(key.detach())
                previous_key = key_descriptors[:, :-1]
                current_query = query_descriptors[:, 1:]
                descriptor_dim = query_descriptors.shape[-1]
                kernel_size = 2 * latent_consist_match_radius + 1
                num_pairs = num_tokens_t - 1

                previous_key_4d = previous_key.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_pairs, descriptor_dim, height_tokens, width_tokens
                )
                patches = F.unfold(previous_key_4d, kernel_size=kernel_size, padding=latent_consist_match_radius)
                patches = patches.reshape(
                    batch, num_pairs, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                ).permute(0, 1, 4, 5, 3, 2)
                similarity = (current_query.unsqueeze(-2) * patches).sum(dim=-1)
                best_similarity, best_index = similarity.max(dim=-1)

                offset_y = torch.div(best_index, kernel_size, rounding_mode="floor") - latent_consist_match_radius
                offset_x = best_index.remainder(kernel_size) - latent_consist_match_radius
                base_y = torch.arange(height_tokens, device=query.device).view(1, 1, height_tokens, 1)
                base_x = torch.arange(width_tokens, device=query.device).view(1, 1, 1, width_tokens)
                source_y = (base_y + offset_y).clamp(0, height_tokens - 1)
                source_x = (base_x + offset_x).clamp(0, width_tokens - 1)
                source_index = (source_y * width_tokens + source_x).reshape(batch, num_pairs, -1)

                threshold_valid = best_similarity >= latent_consist_match_confidence

                # Mutual Q/K matching avoids copying from a one-way accidental correspondence.
                previous_query = query_descriptors[:, :-1]
                current_key = key_descriptors[:, 1:]
                current_key_4d = current_key.permute(0, 1, 4, 2, 3).reshape(
                    batch * num_pairs, descriptor_dim, height_tokens, width_tokens
                )
                reverse_patches = F.unfold(
                    current_key_4d, kernel_size=kernel_size, padding=latent_consist_match_radius
                )
                reverse_patches = reverse_patches.reshape(
                    batch, num_pairs, descriptor_dim, kernel_size * kernel_size, height_tokens, width_tokens
                ).permute(0, 1, 4, 5, 3, 2)
                reverse_similarity = (previous_query.unsqueeze(-2) * reverse_patches).sum(dim=-1)
                reverse_best_similarity, reverse_best_index = reverse_similarity.max(dim=-1)
                reverse_offset_y = (
                    torch.div(reverse_best_index, kernel_size, rounding_mode="floor")
                    - latent_consist_match_radius
                )
                reverse_offset_x = reverse_best_index.remainder(kernel_size) - latent_consist_match_radius
                reverse_y = (base_y + reverse_offset_y).clamp(0, height_tokens - 1)
                reverse_x = (base_x + reverse_offset_x).clamp(0, width_tokens - 1)
                reverse_index = (reverse_y * width_tokens + reverse_x).reshape(batch, num_pairs, -1)
                query_index = (base_y * width_tokens + base_x).expand(
                    batch, num_pairs, height_tokens, width_tokens
                ).reshape(batch, num_pairs, -1)
                mutual = reverse_index.gather(2, source_index) == query_index
                mutual = mutual.reshape(batch, num_pairs, height_tokens, width_tokens)
                mutual = mutual & (reverse_best_similarity >= latent_consist_match_confidence)

                flow = torch.stack(
                    [(source_y - base_y).float(), (source_x - base_x).float()], dim=2
                ).reshape(batch * num_pairs, 2, height_tokens, width_tokens)
                local_flow = F.avg_pool2d(flow, kernel_size=3, stride=1, padding=1)
                flow_residual = torch.linalg.vector_norm(flow - local_flow, dim=1).reshape(
                    batch, num_pairs, height_tokens, width_tokens
                )
                coherent = flow_residual <= latent_consist_static_coherence
                valid = threshold_valid & coherent
                if latent_consist_qk_require_mutual:
                    valid = valid & mutual

                confidence = (
                    (best_similarity - latent_consist_match_confidence)
                    / (1.0 - latent_consist_match_confidence + 1e-6)
                ).clamp(0.0, 1.0)
                valid_count = valid.float().sum().clamp_min(1.0)
                displacement = torch.linalg.vector_norm(
                    torch.stack([(source_y - base_y).float(), (source_x - base_x).float()], dim=-1), dim=-1
                )
                at_search_edge = (offset_y.abs() == latent_consist_match_radius) | (
                    offset_x.abs() == latent_consist_match_radius
                )
                diagnostics = {
                    "match_cos_mean": best_similarity.detach().float().mean(),
                    "match_cos_p10": torch.quantile(best_similarity.detach().float(), 0.1),
                    "match_cos_p90": torch.quantile(best_similarity.detach().float(), 0.9),
                    "qk_threshold_coverage": threshold_valid.detach().float().mean(),
                    "coverage": valid.detach().float().mean(),
                    "mutual_coverage": mutual.detach().float().mean(),
                    "coherent_coverage": coherent.detach().float().mean(),
                    "confidence_valid_mean": (confidence.detach().float() * valid.float()).sum() / valid_count,
                    "qk_displacement_valid_mean": (displacement.detach().float() * valid.float()).sum() / valid_count,
                    "qk_at_search_edge_valid": (at_search_edge.float() * valid.float()).sum() / valid_count,
                }
                return source_index, valid, confidence, diagnostics

            def _anchor_gate(
                input_hidden: torch.Tensor,
                source_index: torch.Tensor,
                valid: torch.Tensor,
                confidence: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
                """Keep only one-step matches whose composed track remains first-frame consistent.

                The transported feature remains a one-step previous-frame feature. The anchor only
                gates unreliable paths, so an early mismatch cannot recursively overwrite later
                transformer features.
                """
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                spatial_tokens = height_tokens * width_tokens
                descriptors = _reduced_descriptors(input_hidden.detach()).reshape(
                    batch, num_tokens_t, spatial_tokens, -1
                )
                anchor_descriptors = descriptors[:, :1].expand(-1, num_tokens_t - 1, -1, -1)

                anchor_index = torch.arange(spatial_tokens, device=input_hidden.device).view(1, spatial_tokens)
                anchor_index = anchor_index.expand(batch, -1)
                anchor_valid = torch.ones((batch, spatial_tokens), device=input_hidden.device, dtype=torch.bool)
                anchor_confidence = torch.ones((batch, spatial_tokens), device=input_hidden.device, dtype=torch.float32)

                anchor_valid_frames = []
                anchor_confidence_frames = []
                anchor_similarity_frames = []
                for frame_idx in range(1, num_tokens_t):
                    indices = source_index[:, frame_idx - 1]
                    anchor_index = anchor_index.gather(1, indices)
                    prior_valid = anchor_valid.gather(1, indices)
                    prior_confidence = anchor_confidence.gather(1, indices)

                    anchor_descriptor = anchor_descriptors[:, frame_idx - 1].gather(
                        1, anchor_index.unsqueeze(-1).expand(-1, -1, descriptors.shape[-1])
                    )
                    anchor_similarity = (descriptors[:, frame_idx] * anchor_descriptor).sum(dim=-1)
                    edge_valid = valid[:, frame_idx - 1].reshape(batch, spatial_tokens)
                    edge_confidence = confidence[:, frame_idx - 1].reshape(batch, spatial_tokens).float()

                    anchor_valid = (
                        edge_valid
                        & prior_valid
                        & (anchor_similarity >= latent_consist_anchor_confidence)
                    )
                    anchor_confidence = torch.minimum(prior_confidence, edge_confidence)
                    anchor_confidence = anchor_confidence * anchor_valid.float()

                    anchor_valid_frames.append(anchor_valid)
                    anchor_confidence_frames.append(anchor_confidence)
                    anchor_similarity_frames.append(anchor_similarity)

                anchor_valid_all = torch.stack(anchor_valid_frames, dim=1).reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                )
                anchor_confidence_all = torch.stack(anchor_confidence_frames, dim=1).reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                )
                anchor_similarity_all = torch.stack(anchor_similarity_frames, dim=1)
                valid_count = anchor_valid_all.float().sum().clamp_min(1.0)
                diagnostics = {
                    "anchor_coverage": anchor_valid_all.float().mean(),
                    "anchor_cos_mean": anchor_similarity_all.float().mean(),
                    "anchor_confidence_valid_mean": (
                        anchor_confidence_all.float() * anchor_valid_all.float()
                    ).sum() / valid_count,
                }
                return anchor_valid_all, anchor_confidence_all, diagnostics

            def _tracklet_gate(
                input_hidden: torch.Tensor,
                source_index: torch.Tensor,
                valid: torch.Tensor,
                confidence: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
                """Validate one-step static matches over a short, non-recursive temporal tracklet.

                Each current token is composed backwards for at most ``tracklet_horizon``
                local matches.  The resulting correspondence is used only as a reliability
                gate for the original one-step feature transport below; no mixed feature is
                propagated through the tracklet itself.
                """
                batch, num_tokens_t, height_tokens, width_tokens, _ = input_hidden.shape
                spatial_tokens = height_tokens * width_tokens
                descriptors = _reduced_descriptors(input_hidden.detach()).reshape(
                    batch, num_tokens_t, spatial_tokens, -1
                )
                valid_flat = valid.reshape(batch, num_tokens_t - 1, spatial_tokens)
                confidence_flat = confidence.reshape(batch, num_tokens_t - 1, spatial_tokens).float()
                identity = torch.arange(spatial_tokens, device=input_hidden.device).view(1, spatial_tokens)

                tracklet_valid_frames = []
                tracklet_confidence_frames = []
                tracklet_similarity_frames = []
                tracklet_spans = []
                for frame_idx in range(1, num_tokens_t):
                    span = min(latent_consist_tracklet_horizon, frame_idx)
                    track_index = identity.expand(batch, -1)
                    track_valid = torch.ones((batch, spatial_tokens), device=input_hidden.device, dtype=torch.bool)
                    track_confidence = torch.ones(
                        (batch, spatial_tokens), device=input_hidden.device, dtype=torch.float32
                    )

                    # Compose frame_idx -> frame_idx - 1 -> ... -> frame_idx - span.
                    for edge_idx in range(frame_idx - 1, frame_idx - span - 1, -1):
                        edge_valid = valid_flat[:, edge_idx].gather(1, track_index)
                        edge_confidence = confidence_flat[:, edge_idx].gather(1, track_index)
                        track_valid = track_valid & edge_valid
                        track_confidence = torch.minimum(track_confidence, edge_confidence)
                        track_index = source_index[:, edge_idx].gather(1, track_index)

                    reference_descriptor = descriptors[:, frame_idx - span].gather(
                        1, track_index.unsqueeze(-1).expand(-1, -1, descriptors.shape[-1])
                    )
                    tracklet_similarity = (descriptors[:, frame_idx] * reference_descriptor).sum(dim=-1)
                    track_valid = track_valid & (tracklet_similarity >= latent_consist_tracklet_confidence)
                    track_confidence = track_confidence * track_valid.float()

                    tracklet_valid_frames.append(track_valid)
                    tracklet_confidence_frames.append(track_confidence)
                    tracklet_similarity_frames.append(tracklet_similarity)
                    tracklet_spans.append(float(span))

                tracklet_valid_all = torch.stack(tracklet_valid_frames, dim=1).reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                )
                tracklet_confidence_all = torch.stack(tracklet_confidence_frames, dim=1).reshape(
                    batch, num_tokens_t - 1, height_tokens, width_tokens
                )
                tracklet_similarity_all = torch.stack(tracklet_similarity_frames, dim=1)
                valid_count = tracklet_valid_all.float().sum().clamp_min(1.0)
                diagnostics = {
                    "tracklet_coverage": tracklet_valid_all.float().mean(),
                    "tracklet_cos_mean": tracklet_similarity_all.float().mean(),
                    "tracklet_confidence_valid_mean": (
                        tracklet_confidence_all.float() * tracklet_valid_all.float()
                    ).sum() / valid_count,
                    "tracklet_span_mean": torch.tensor(
                        tracklet_spans, device=input_hidden.device, dtype=torch.float32
                    ).mean(),
                }
                return tracklet_valid_all, tracklet_confidence_all, diagnostics

            def _record_diagnostics(diagnostics: dict[str, torch.Tensor]):
                key = (latent_consist_state["step"], latent_consist_state["branch"])
                if key in latent_consist_state["printed"]:
                    return
                latent_consist_state["printed"].add(key)
                summary = {"step": int(key[0]), "branch": key[1]}
                summary.update({name: float(value.item()) for name, value in diagnostics.items()})
                latent_consist_state["diagnostics"].append(summary)
                if latent_consist_debug:
                    print(
                        "[latent-consist] "
                        f"step={summary['step']} branch={summary['branch']} mode={latent_consist_mode} "
                        f"match_cos_mean={summary['match_cos_mean']:.3f} "
                        f"coverage={summary['coverage']:.3f} "
                        f"mutual={summary['mutual_coverage']:.3f} "
                        f"coherent={summary['coherent_coverage']:.3f} "
                        f"conf={summary['confidence_valid_mean']:.3f}"
                        + (
                            f" anchor_cov={summary['anchor_coverage']:.3f}"
                            f" anchor_cos={summary['anchor_cos_mean']:.3f}"
                            if "anchor_coverage" in summary
                            else ""
                        )
                        + (
                            f" tracklet_cov={summary['tracklet_coverage']:.3f}"
                            f" tracklet_cos={summary['tracklet_cos_mean']:.3f}"
                            f" span={summary['tracklet_span_mean']:.1f}"
                            if "tracklet_coverage" in summary
                            else ""
                        ),
                        flush=True,
                    )

            def _latent_consist_hook(module, inputs, output):
                if output is None:
                    return output
                if latent_consist_capture["enabled"]:
                    latent_consist_capture["features"] = output
                    latent_consist_capture["inputs"] = inputs[0]

                active_step = latent_consist_start_step <= latent_consist_state["step"] <= latent_consist_end_step
                active_branch = not latent_consist_cond_only or latent_consist_state["branch"] == "cond"
                if not (is_injection_mode and active_step and active_branch):
                    return output
                if output.ndim != 3 or not inputs or inputs[0].ndim != 3:
                    raise ValueError("Attention output and input must both have shape [B, N, C]")

                batch_size_tokens, seq_len, channels = output.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                expected_seq_len = num_tokens_t * height_tokens * width_tokens
                if seq_len != expected_seq_len:
                    raise ValueError(
                        f"Attention output seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                    )

                hidden = output.reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                input_hidden = inputs[0].reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                matched_output, valid, confidence, diagnostics, _ = _local_correspondence(
                    hidden,
                    input_hidden,
                    require_static=latent_consist_mode == "static_match_prev",
                )
                _record_diagnostics(diagnostics)
                blend = latent_consist_lr * confidence * valid.float()
                mixed = hidden.clone()
                mixed[:, 1:] = (
                    hidden[:, 1:].float() * (1.0 - blend.unsqueeze(-1))
                    + matched_output.float() * blend.unsqueeze(-1)
                ).to(output.dtype)
                return mixed.reshape(batch_size_tokens, seq_len, channels)

            def _latent_consist_input_hook(module, inputs):
                """Blend only matched static-world candidates before Wan's Q/K/V projections.

                Unlike output propagation, this lets the frozen attention block re-interpret the
                transported feature through its own attention weights and MLP stack.
                """
                active_step = latent_consist_start_step <= latent_consist_state["step"] <= latent_consist_end_step
                active_branch = not latent_consist_cond_only or latent_consist_state["branch"] == "cond"
                if not (latent_consist_mode in input_injection_modes and active_step and active_branch):
                    return None
                if not inputs or inputs[0].ndim != 3:
                    raise ValueError("Attention input must have shape [B, N, C]")

                input_hidden = inputs[0]
                batch_size_tokens, seq_len, channels = input_hidden.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                expected_seq_len = num_tokens_t * height_tokens * width_tokens
                if seq_len != expected_seq_len:
                    raise ValueError(
                        f"Attention input seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                    )

                hidden = input_hidden.reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                matched_input, valid, confidence, diagnostics, source_index = _local_correspondence(
                    hidden,
                    hidden,
                    require_static=latent_consist_mode in {
                        "static_match_prev_input",
                        "static_track_prev_input",
                        "static_anchor_match_prev_input",
                        "static_tracklet_match_prev_input",
                    },
                )
                if latent_consist_mode == "static_anchor_match_prev_input":
                    anchor_valid, anchor_confidence, anchor_diagnostics = _anchor_gate(
                        hidden, source_index, valid, confidence
                    )
                    valid = valid & anchor_valid
                    confidence = anchor_confidence
                    diagnostics.update(anchor_diagnostics)
                elif latent_consist_mode == "static_tracklet_match_prev_input":
                    tracklet_valid, tracklet_confidence, tracklet_diagnostics = _tracklet_gate(
                        hidden, source_index, valid, confidence
                    )
                    valid = valid & tracklet_valid
                    confidence = tracklet_confidence
                    diagnostics.update(tracklet_diagnostics)
                _record_diagnostics(diagnostics)
                blend = latent_consist_lr * confidence * valid.float()
                mixed = hidden.clone()
                if latent_consist_mode == "static_track_prev_input":
                    # Recursively transport trusted static tokens, giving each token a persistent
                    # feature track back to the conditioned first frame while leaving unmatched
                    # regions on the model's original trajectory.
                    for frame_idx in range(1, num_tokens_t):
                        previous = mixed[:, frame_idx - 1].reshape(
                            batch_size_tokens, height_tokens * width_tokens, channels
                        )
                        indices = source_index[:, frame_idx - 1]
                        tracked_input = previous.gather(
                            1, indices.unsqueeze(-1).expand(-1, -1, channels)
                        ).reshape(batch_size_tokens, height_tokens, width_tokens, channels)
                        frame_blend = blend[:, frame_idx - 1].unsqueeze(-1)
                        mixed[:, frame_idx] = (
                            hidden[:, frame_idx].float() * (1.0 - frame_blend)
                            + tracked_input.float() * frame_blend
                        ).to(input_hidden.dtype)
                else:
                    mixed[:, 1:] = (
                        hidden[:, 1:].float() * (1.0 - blend.unsqueeze(-1))
                        + matched_input.float() * blend.unsqueeze(-1)
                    ).to(input_hidden.dtype)
                return (mixed.reshape(batch_size_tokens, seq_len, channels), *inputs[1:])

            def _qk_mode_is_active() -> bool:
                active_step = latent_consist_start_step <= latent_consist_state["step"] <= latent_consist_end_step
                active_branch = not latent_consist_cond_only or latent_consist_state["branch"] == "cond"
                return latent_consist_mode in qk_modes and active_step and active_branch

            def _clear_qk_cache():
                latent_consist_qk_cache["q_key"] = None
                latent_consist_qk_cache["query"] = None
                latent_consist_qk_cache["kv_key"] = None
                latent_consist_qk_cache["source_index"] = None
                latent_consist_qk_cache["blend"] = None

            def _latent_consist_q_forward_hook(module, inputs, output):
                if not _qk_mode_is_active():
                    return output
                if output is None or not inputs or inputs[0].ndim != 3 or output.ndim != 3:
                    raise ValueError("Self-attention Q input/output must have shape [B, N, C]")
                latent_consist_qk_cache["q_key"] = (
                    int(latent_consist_state["step"]),
                    latent_consist_state["branch"],
                    int(inputs[0].data_ptr()),
                )
                latent_consist_qk_cache["query"] = output
                return output

            def _latent_consist_qk_k_forward_hook(module, inputs, output):
                if not _qk_mode_is_active():
                    return output
                if output is None or not inputs or inputs[0].ndim != 3 or output.ndim != 3:
                    raise ValueError("Self-attention K input/output must have shape [B, N, C]")

                input_hidden = inputs[0]
                cache_key = (
                    int(latent_consist_state["step"]),
                    latent_consist_state["branch"],
                    int(input_hidden.data_ptr()),
                )
                query = latent_consist_qk_cache["query"]
                if latent_consist_qk_cache["q_key"] != cache_key or query is None:
                    raise RuntimeError("Q/K consistency hook did not capture the matching query projection.")
                if query.shape != output.shape:
                    raise ValueError("Projected Q and K must have identical [B, N, C] shapes")

                batch_size_tokens, seq_len, channels = output.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                expected_seq_len = num_tokens_t * height_tokens * width_tokens
                if seq_len != expected_seq_len:
                    raise ValueError(
                        f"Projected K seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                    )

                query_grid = query.reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                key_grid = output.reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                source_index, valid, confidence, diagnostics = _local_qk_correspondence(query_grid, key_grid)
                _record_diagnostics(diagnostics)

                if latent_consist_mode in qk_probe_modes:
                    _clear_qk_cache()
                    return output

                blend = latent_consist_lr * confidence * valid.float()
                matched_key = _gather_previous_features(key_grid, source_index)
                mixed_key = key_grid.clone()
                mixed_key[:, 1:] = (
                    key_grid[:, 1:].float() * (1.0 - blend.unsqueeze(-1))
                    + matched_key.float() * blend.unsqueeze(-1)
                ).to(output.dtype)

                latent_consist_qk_cache["q_key"] = None
                latent_consist_qk_cache["query"] = None
                latent_consist_qk_cache["kv_key"] = cache_key
                latent_consist_qk_cache["source_index"] = source_index
                latent_consist_qk_cache["blend"] = blend
                return mixed_key.reshape(batch_size_tokens, seq_len, channels)

            def _latent_consist_qk_v_forward_hook(module, inputs, output):
                if not _qk_mode_is_active():
                    return output
                if output is None or not inputs or inputs[0].ndim != 3 or output.ndim != 3:
                    raise ValueError("Self-attention V input/output must have shape [B, N, C]")

                input_hidden = inputs[0]
                cache_key = (
                    int(latent_consist_state["step"]),
                    latent_consist_state["branch"],
                    int(input_hidden.data_ptr()),
                )
                source_index = latent_consist_qk_cache["source_index"]
                blend = latent_consist_qk_cache["blend"]
                if latent_consist_qk_cache["kv_key"] != cache_key or source_index is None or blend is None:
                    raise RuntimeError("Q/K/V consistency hooks lost their shared projected-attention state.")

                try:
                    batch_size_tokens, seq_len, channels = output.shape
                    num_tokens_t, height_tokens, width_tokens = token_grid
                    expected_seq_len = num_tokens_t * height_tokens * width_tokens
                    if seq_len != expected_seq_len:
                        raise ValueError(
                            f"Projected V seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                        )
                    value_grid = output.reshape(
                        batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels
                    )
                    matched_value = _gather_previous_features(value_grid, source_index)
                    mixed_value = value_grid.clone()
                    mixed_value[:, 1:] = (
                        value_grid[:, 1:].float() * (1.0 - blend.unsqueeze(-1))
                        + matched_value.float() * blend.unsqueeze(-1)
                    ).to(output.dtype)
                    return mixed_value.reshape(batch_size_tokens, seq_len, channels)
                finally:
                    _clear_qk_cache()

            def _latent_consist_k_pre_hook(module, inputs):
                """Inject trusted previous-frame features into K only; Q remains frame-native."""
                active_step = latent_consist_start_step <= latent_consist_state["step"] <= latent_consist_end_step
                active_branch = not latent_consist_cond_only or latent_consist_state["branch"] == "cond"
                if not (latent_consist_mode in kv_injection_modes and active_step and active_branch):
                    return None
                if not inputs or inputs[0].ndim != 3:
                    raise ValueError("Self-attention K input must have shape [B, N, C]")

                input_hidden = inputs[0]
                batch_size_tokens, seq_len, channels = input_hidden.shape
                num_tokens_t, height_tokens, width_tokens = token_grid
                expected_seq_len = num_tokens_t * height_tokens * width_tokens
                if seq_len != expected_seq_len:
                    raise ValueError(
                        f"Attention input seq_len {seq_len} does not match token grid {token_grid} = {expected_seq_len}."
                    )

                hidden = input_hidden.reshape(batch_size_tokens, num_tokens_t, height_tokens, width_tokens, channels)
                matched_input, valid, confidence, diagnostics, _ = _local_correspondence(
                    hidden, hidden, require_static=True
                )
                _record_diagnostics(diagnostics)
                blend = latent_consist_lr * confidence * valid.float()
                mixed = hidden.clone()
                mixed[:, 1:] = (
                    hidden[:, 1:].float() * (1.0 - blend.unsqueeze(-1))
                    + matched_input.float() * blend.unsqueeze(-1)
                ).to(input_hidden.dtype)
                mixed = mixed.reshape(batch_size_tokens, seq_len, channels)

                # Wan calls Q before K and V. Cache this exact K input so V receives the
                # same transport while Q remains the unmodified current-frame representation.
                latent_consist_kv_cache["key"] = (
                    int(latent_consist_state["step"]),
                    latent_consist_state["branch"],
                    int(input_hidden.data_ptr()),
                )
                latent_consist_kv_cache["mixed"] = mixed
                return (mixed, *inputs[1:])

            def _latent_consist_v_pre_hook(module, inputs):
                active_step = latent_consist_start_step <= latent_consist_state["step"] <= latent_consist_end_step
                active_branch = not latent_consist_cond_only or latent_consist_state["branch"] == "cond"
                if not (latent_consist_mode in kv_injection_modes and active_step and active_branch):
                    return None
                if not inputs or inputs[0].ndim != 3:
                    raise ValueError("Self-attention V input must have shape [B, N, C]")

                input_hidden = inputs[0]
                cache_key = (
                    int(latent_consist_state["step"]),
                    latent_consist_state["branch"],
                    int(input_hidden.data_ptr()),
                )
                if latent_consist_kv_cache["key"] != cache_key or latent_consist_kv_cache["mixed"] is None:
                    raise RuntimeError("K/V consistency hooks lost their shared self-attention input.")
                mixed = latent_consist_kv_cache["mixed"]
                latent_consist_kv_cache["key"] = None
                latent_consist_kv_cache["mixed"] = None
                return (mixed, *inputs[1:])

            for model in (self.transformer, self.transformer_2):
                if model is None:
                    continue
                if latent_consist_layer < 0 or latent_consist_layer >= len(model.blocks):
                    raise IndexError(
                        f"latent_consist_layer {latent_consist_layer} is out of range for {model.__class__.__name__}"
                    )
                attention = model.blocks[latent_consist_layer].attn1
                if latent_consist_mode in qk_modes:
                    if getattr(attention, "fused_projections", False):
                        raise RuntimeError("Q/K consistency requires unfused Wan self-attention projections.")
                    latent_consist_hook_handles.append(
                        attention.to_q.register_forward_hook(_latent_consist_q_forward_hook)
                    )
                    latent_consist_hook_handles.append(
                        attention.to_k.register_forward_hook(_latent_consist_qk_k_forward_hook)
                    )
                    if latent_consist_mode in qk_injection_modes:
                        handle = attention.to_v.register_forward_hook(_latent_consist_qk_v_forward_hook)
                    else:
                        handle = None
                elif latent_consist_mode in kv_injection_modes:
                    if getattr(attention, "fused_projections", False):
                        raise RuntimeError("K/V-only consistency requires unfused Wan self-attention projections.")
                    latent_consist_hook_handles.append(attention.to_k.register_forward_pre_hook(_latent_consist_k_pre_hook))
                    handle = attention.to_v.register_forward_pre_hook(_latent_consist_v_pre_hook)
                elif latent_consist_mode in input_injection_modes:
                    handle = model.blocks[latent_consist_layer].attn1.register_forward_pre_hook(
                        _latent_consist_input_hook
                    )
                else:
                    handle = model.blocks[latent_consist_layer].attn1.register_forward_hook(_latent_consist_hook)
                if handle is not None:
                    latent_consist_hook_handles.append(handle)

        def _prepare_latent_model_input(current_latents: torch.Tensor, current_timestep: torch.Tensor):
            if self.config.expand_timesteps:
                model_input = (1 - first_frame_mask) * condition + first_frame_mask * current_latents
                model_input = model_input.to(transformer_dtype)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * current_timestep).flatten()
                model_timestep = temp_ts.unsqueeze(0).expand(current_latents.shape[0], -1)
            else:
                model_input = torch.cat([current_latents, condition], dim=1).to(transformer_dtype)
                model_timestep = current_timestep.expand(current_latents.shape[0])
            return model_input, model_timestep

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

                do_latent_consist_guidance = (
                    is_gradient_mode and latent_consist_start_step <= i <= latent_consist_end_step
                )

                if do_latent_consist_guidance:
                    latents = latents.detach().requires_grad_(True)
                    latent_consist_capture["features"] = None
                    latent_consist_capture["inputs"] = None

                    with torch.enable_grad():
                        latent_model_input, timestep = _prepare_latent_model_input(latents, t)

                        latent_consist_state["step"] = i
                        latent_consist_state["branch"] = "cond"
                        latent_consist_capture["enabled"] = True
                        try:
                            with current_model.cache_context("cond"):
                                noise_pred_guidance = current_model(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_hidden_states_image=image_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]
                        finally:
                            latent_consist_capture["enabled"] = False

                        features = latent_consist_capture["features"]
                        feature_inputs = latent_consist_capture["inputs"]
                        if features is None or feature_inputs is None:
                            raise RuntimeError("Latent consistency hook did not capture attention features.")
                        if features.ndim != 3 or feature_inputs.ndim != 3:
                            raise ValueError(f"Attention output must be [B, N, C], got shape {tuple(features.shape)}")

                        batch_size_tokens, seq_len, channels = features.shape
                        num_frames_tokens, height_tokens, width_tokens = token_grid
                        expected_seq_len = num_frames_tokens * height_tokens * width_tokens
                        if seq_len != expected_seq_len:
                            raise ValueError(
                                f"Attention output seq_len {seq_len} does not match token grid "
                                f"{token_grid} = {expected_seq_len}."
                            )

                        features_5d = features.reshape(
                            batch_size_tokens, num_frames_tokens, height_tokens, width_tokens, channels
                        )
                        if num_frames_tokens > 1:
                            if latent_consist_mode == "same_position_l2":
                                # A deliberately simple control: it is normalized, but still confuses camera motion
                                # with inconsistency because it compares identical image coordinates.
                                loss = (
                                    features_5d[:, 1:].float() - features_5d[:, :-1].detach().float()
                                ).square().mean()
                            else:
                                input_5d = feature_inputs.reshape(
                                    batch_size_tokens, num_frames_tokens, height_tokens, width_tokens, channels
                                )
                                matched_output, valid, _, diagnostics, _ = _local_correspondence(
                                    features_5d,
                                    input_5d,
                                    require_static=False,
                                )
                                _record_diagnostics(diagnostics)
                                feature_cosine = F.cosine_similarity(
                                    features_5d[:, 1:].float(), matched_output.detach().float(), dim=-1, eps=1e-6
                                )
                                weights = valid.float()
                                loss = ((1.0 - feature_cosine) * weights).sum() / weights.sum().clamp_min(1.0)
                        else:
                            loss = features_5d.sum() * 0.0
                        if not loss.requires_grad:
                            raise RuntimeError(
                                "Latent consistency loss has no gradient path. If gradient checkpointing is enabled, "
                                "the hook may be capturing no-grad checkpoint activations."
                            )

                        grad = torch.autograd.grad(loss, latents)[0]
                        loss_value = float(loss.detach().item())
                        latent_consist_losses.append((int(i), loss_value))
                        update = latent_consist_lr * grad
                        latent_mean_abs = latents.detach().float().abs().mean()
                        if latent_consist_max_relative_delta > 0.0:
                            max_delta = latent_consist_max_relative_delta * (latent_mean_abs + 1e-8)
                            update_mean_abs = update.detach().float().abs().mean()
                            if update_mean_abs > max_delta:
                                update = update * (max_delta / (update_mean_abs + 1e-8))
                        update_mean_abs = update.detach().float().abs().mean()
                        grad_stats = {
                            "step": int(i),
                            "loss": loss_value,
                            "grad_norm": float(grad.detach().float().norm().item()),
                            "grad_mean_abs": float(grad.detach().float().abs().mean().item()),
                            "latent_mean_abs": float(latent_mean_abs.item()),
                            "latent_delta": float(update_mean_abs.item()),
                            "relative_delta": float((update_mean_abs / (latent_mean_abs + 1e-8)).item()),
                        }
                        latent_consist_gradient_stats.append(grad_stats)
                        print(
                            f"latent_consist_loss({i}): {loss_value:.6f} "
                            f"grad_norm={grad_stats['grad_norm']:.6f} "
                            f"relative_delta={100.0 * grad_stats['relative_delta']:.5f}%",
                            flush=True,
                        )

                        latents = (latents - update).detach()
                        del grad, update, loss, features, feature_inputs, features_5d, noise_pred_guidance

                    # The guidance update changes x_t. Recompute the conditional model prediction at the updated
                    # latent before scheduler.step; reusing the pre-update prediction would be a stale solver field.
                    with torch.no_grad():
                        latent_model_input, timestep = _prepare_latent_model_input(latents, t)
                        latent_consist_state["step"] = i
                        latent_consist_state["branch"] = "cond"
                        with current_model.cache_context("cond"):
                            noise_pred = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                else:
                    latent_model_input, timestep = _prepare_latent_model_input(latents, t)

                    latent_consist_state["step"] = i
                    latent_consist_state["branch"] = "cond"
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
                    latent_consist_state["branch"] = "uncond"
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
                latent_consist_state["branch"] = "none"
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

        for handle in latent_consist_hook_handles:
            handle.remove()
        self._latent_consist_losses = latent_consist_losses
        self._latent_consist_gradient_stats = latent_consist_gradient_stats
        self._latent_consist_diagnostics = latent_consist_state["diagnostics"]

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
