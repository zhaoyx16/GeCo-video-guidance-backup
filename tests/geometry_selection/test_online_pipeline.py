from __future__ import annotations

import importlib.util
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _module(**attributes) -> ModuleType:
    result = ModuleType("stub")
    for name, value in attributes.items():
        setattr(result, name, value)
    return result


def _load_pipeline_class(monkeypatch: pytest.MonkeyPatch):
    """Load the local pipeline with light stubs, avoiding heavyweight imports."""

    class DiffusionPipeline:
        pass

    class WanLoraLoaderMixin:
        pass

    class WanPipelineOutput:
        def __init__(self, frames):
            self.frames = frames

    class Dummy:
        pass

    class Logger:
        @staticmethod
        def get_logger(_name):
            return SimpleNamespace()

    def identity_decorator(_value):
        return lambda function: function

    stubs = {
        "regex": re,
        "transformers": _module(
            AutoTokenizer=Dummy,
            CLIPImageProcessor=Dummy,
            CLIPVisionModel=Dummy,
            UMT5EncoderModel=Dummy,
        ),
        "diffusers": _module(),
        "diffusers.callbacks": _module(
            MultiPipelineCallbacks=Dummy, PipelineCallback=Dummy
        ),
        "diffusers.image_processor": _module(PipelineImageInput=object),
        "diffusers.loaders": _module(WanLoraLoaderMixin=WanLoraLoaderMixin),
        "diffusers.models": _module(AutoencoderKLWan=Dummy, WanTransformer3DModel=Dummy),
        "diffusers.models.autoencoders": _module(),
        "diffusers.models.autoencoders.autoencoder_kl_wan": _module(
            unpatchify=lambda value, *_args, **_kwargs: value
        ),
        "diffusers.schedulers": _module(FlowMatchEulerDiscreteScheduler=Dummy),
        "diffusers.utils": _module(
            is_ftfy_available=lambda: False,
            is_torch_xla_available=lambda: False,
            logging=Logger,
            replace_example_docstring=identity_decorator,
        ),
        "diffusers.utils.torch_utils": _module(randn_tensor=torch.randn),
        "diffusers.video_processor": _module(VideoProcessor=Dummy),
        "diffusers.pipelines": _module(),
        "diffusers.pipelines.pipeline_utils": _module(DiffusionPipeline=DiffusionPipeline),
        "diffusers.pipelines.wan": _module(),
        "diffusers.pipelines.wan.pipeline_output": _module(WanPipelineOutput=WanPipelineOutput),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    root = Path(__file__).resolve().parents[2]
    source = root / "external/guidance_wan/pipeline_wan_i2v_full_guided.py"
    spec = importlib.util.spec_from_file_location("tested_online_wan_pipeline", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanImageToVideoPipeline


class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0, 0.0])
        self.order = 1
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, _steps, device=None) -> None:
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)

    def step(self, model_output, timestep, sample, return_dict=False):
        self.calls.append((model_output.clone(), timestep.clone(), sample.clone()))
        assert return_dict is False
        return (sample + model_output,)


class _TinyTransformer:
    dtype = torch.float32
    config = SimpleNamespace(image_dim=None, patch_size=(1, 1, 1))
    is_cache_enabled = False

    def __init__(self) -> None:
        self.hidden_states: list[torch.Tensor] = []

    def requires_grad_(self, _flag):
        return self

    @contextmanager
    def cache_context(self, _kind):
        yield

    def __call__(self, *, hidden_states, **_kwargs):
        self.hidden_states.append(hidden_states.detach().clone())
        return (hidden_states[:, :1] + 10.0,)


class _TinyVAE:
    dtype = torch.float32
    config = SimpleNamespace(z_dim=1)

    def requires_grad_(self, _flag):
        return self


class _TinyVideoProcessor:
    def preprocess(self, _image, *, height, width):
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


class _TinySelector:
    def __init__(self) -> None:
        self.contexts = []
        self.scheduler_outputs = []

    def is_active(self, step_index):
        return step_index == 0

    def __call__(self, context):
        self.contexts.append(context)
        return SimpleNamespace(selected_latents=context.incumbent_latents + 5.0)

    def record_scheduler_output(self, step_index, latents):
        self.scheduler_outputs.append((step_index, latents.detach().clone()))


def test_online_handoff_recomputes_selected_branch_and_steps_scheduler_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    WanPipeline = _load_pipeline_class(monkeypatch)

    class TinyWanPipeline(WanPipeline):
        @property
        def _execution_device(self):
            return torch.device("cpu")

        @property
        def config(self):
            return SimpleNamespace(expand_timesteps=False, boundary_ratio=None)

        def check_inputs(self, *_args, **_kwargs):
            return None

        def encode_prompt(self, **_kwargs):
            return torch.zeros((1, 1, 1)), None

        def prepare_latents(self, *_args, **_kwargs):
            return torch.ones((1, 1, 1, 1, 1)), torch.zeros((1, 1, 1, 1, 1))

        @contextmanager
        def progress_bar(self, total):
            assert total == 1
            yield SimpleNamespace(update=lambda: None)

        def maybe_free_model_hooks(self):
            return None

    pipe = object.__new__(TinyWanPipeline)
    pipe.scheduler = _TinyScheduler()
    pipe.transformer = _TinyTransformer()
    pipe.transformer_2 = None
    pipe.vae = _TinyVAE()
    pipe.video_processor = _TinyVideoProcessor()
    pipe.vae_scale_factor_temporal = 4
    pipe.vae_scale_factor_spatial = 8
    selector = _TinySelector()

    result = pipe(
        image=torch.zeros((3, 1, 1)),
        prompt="test",
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
        online_selector=selector,
    )

    # First forward evaluates the incumbent (latent 1).  The selection returns
    # latent 6: the re-prediction and sole scheduler call must use that state.
    assert [item[:, :1].item() for item in pipe.transformer.hidden_states] == [1.0, 6.0]
    assert len(pipe.scheduler.calls) == 1
    model_output, _timestep, scheduler_input = pipe.scheduler.calls[0]
    assert scheduler_input.item() == 6.0
    assert model_output.item() == 16.0
    assert selector.scheduler_outputs == [(0, torch.tensor([[[[[22.0]]]]]))]
    assert result.frames.item() == 22.0
