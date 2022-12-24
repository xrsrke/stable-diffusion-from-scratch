import torch
from torch.testing import assert_close
import pytest

from foundation.scheduler.lms import LMSDiscreteScheduler


@pytest.fixture
def scheduler():
    return LMSDiscreteScheduler()


def test_create_scheduler(scheduler):

    assert_close(scheduler.init_noise_sigma, torch.tensor(14.6146), rtol=1e-5, atol=1e-5)
    assert len(scheduler.timesteps) == 1000


def test_set_timesteps_for_scheduler(scheduler):
    n_inference_steps = 30
    scheduler.set_timesteps(n_inference_steps)

    assert len(scheduler.sigmas) == n_inference_steps + 1

def test_scale_model_input_for_scheduler(scheduler):
    timestep = scheduler.timesteps[-1]
    latents = torch.randn(3, 4)

    output = scheduler.scale_model_input(latents, timestep)

    assert output.shape == (3, 4)

def test_step_for_scheduler(scheduler):
    pass
