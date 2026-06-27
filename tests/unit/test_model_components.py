from __future__ import annotations

import torch

from chdqn.reference import get_reference_config
from chdqn.utils import finite_difference_lipschitz


def test_encoder_matches_reference(trainer, reference_run):
    trainer.model.eval()
    encoded = trainer.model.encoder(reference_run.observation)
    trainer.model.train()
    assert torch.allclose(encoded, reference_run.encoded, atol=1e-3)


def test_encoder_scaling_is_linear(trainer, reference_run):
    trainer.model.eval()
    scaled = trainer.model.encoder(reference_run.observation * 2.0)
    trainer.model.train()
    assert torch.allclose(scaled, reference_run.encoded * 2.0, atol=1e-3)


def test_filter_matches_reference_single_step(trainer, reference_run):
    prev_h = torch.zeros(1, 2)
    trainer.model.eval()
    out = trainer.model.forward_step(reference_run.observation.unsqueeze(0), prev_h)
    trainer.model.train()
    assert torch.allclose(out.h.squeeze(0), reference_run.latent, atol=2e-3)


def test_filter_is_contractive_locally(trainer):
    prev_1 = torch.tensor([[0.1, -0.1]])
    prev_2 = torch.tensor([[0.11, -0.11]])
    z = torch.tensor([[0.085, 0.055]])
    out_1 = trainer.model.filter(prev_1, z)
    out_2 = trainer.model.filter(prev_2, z)
    assert torch.norm(out_1 - out_2).item() < torch.norm(prev_1 - prev_2).item()


def test_markov_property_depends_only_on_previous_hidden_and_current_obs(trainer):
    obs = torch.tensor([[0.2, 0.05]])
    prev_h = torch.tensor([[0.03, -0.04]])
    trainer.model.eval()
    h_a = trainer.model.forward_step(obs, prev_h).h
    h_b = trainer.model.forward_step(obs, prev_h).h
    trainer.model.train()
    assert torch.allclose(h_a, h_b, atol=1e-6)


def test_smoother_is_non_trivial(trainer):
    h_prev = torch.tensor([[0.05, -0.02]])
    h_next = torch.tensor([[0.08, -0.05]])
    h_smooth = trainer.model.smoother(h_prev, h_next)
    assert not torch.allclose(h_smooth, h_next, atol=1e-6)


def test_policy_is_smooth(trainer):
    h = torch.tensor([[0.1, -0.05]])
    lipschitz = finite_difference_lipschitz(lambda x: trainer.model.policy(x, temperature=0.5), h)
    assert lipschitz < 10.0


def test_reference_epoch_curve_is_monotone():
    config = get_reference_config()
    q_values = torch.tensor(config.epoch_q0)
    assert torch.all(q_values[1:] >= q_values[:-1])
