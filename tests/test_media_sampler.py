import torch

from media.generative import ImageGenerator, VideoGenerator
from media.sampler import DiffusionSchedule, ddim_sample, ddpm_sample


def test_ddim_sampler_preserves_image_shape_and_finite_values():
    model = ImageGenerator(channels=4, hidden=16)
    schedule = DiffusionSchedule(steps=8)
    out = ddim_sample(model, (2, 4, 8, 8), schedule=schedule, inference_steps=4)
    assert out.shape == (2, 4, 8, 8)
    assert torch.isfinite(out).all()


def test_ddpm_sampler_preserves_video_shape():
    model = VideoGenerator(channels=4, hidden=16)
    schedule = DiffusionSchedule(steps=6)
    out = ddpm_sample(model, (1, 4, 2, 8, 8), schedule=schedule)
    assert out.shape == (1, 4, 2, 8, 8)
    assert torch.isfinite(out).all()


def test_ddim_eta_zero_is_reproducible_for_fixed_seed():
    model = ImageGenerator(channels=4, hidden=16)
    schedule = DiffusionSchedule(steps=8)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    first = ddim_sample(model, (1, 4, 8, 8), schedule=schedule, inference_steps=4, generator=g1)
    second = ddim_sample(model, (1, 4, 8, 8), schedule=schedule, inference_steps=4, generator=g2)
    assert torch.equal(first, second)
