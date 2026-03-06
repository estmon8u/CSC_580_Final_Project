import torch

from tiny_dreamer_highway.models import RecurrentStateSpaceModel


def test_rssm_initial_state_shapes() -> None:
    rssm = RecurrentStateSpaceModel(action_dim=2, embedding_dim=256)

    state = rssm.initial_state(batch_size=4)

    assert state.deterministic is not None
    assert state.stochastic is not None
    assert state.deterministic.shape == (4, 128)
    assert state.stochastic.shape == (4, 32)
    assert state.features.shape == (4, 160)


def test_rssm_imagine_step_preserves_latent_shapes() -> None:
    rssm = RecurrentStateSpaceModel(action_dim=2, embedding_dim=256)
    prev_state = rssm.initial_state(batch_size=3)
    action = torch.randn(3, 2)

    next_state = rssm.imagine_step(prev_state, action)

    assert next_state.embedding is None
    assert next_state.deterministic is not None
    assert next_state.stochastic is not None
    assert next_state.deterministic.shape == (3, 128)
    assert next_state.stochastic.shape == (3, 32)
    assert next_state.features.shape == (3, 160)


def test_rssm_observe_step_incorporates_encoder_embedding() -> None:
    rssm = RecurrentStateSpaceModel(action_dim=2, embedding_dim=256)
    prev_state = rssm.initial_state(batch_size=5)
    action = torch.randn(5, 2)
    embedding = torch.randn(5, 256)

    posterior_state = rssm.observe_step(prev_state, action, embedding)

    assert posterior_state.embedding is not None
    assert posterior_state.embedding.shape == (5, 256)
    assert posterior_state.deterministic is not None
    assert posterior_state.stochastic is not None
    assert posterior_state.deterministic.shape == (5, 128)
    assert posterior_state.stochastic.shape == (5, 32)


def test_rssm_rolls_forward_multiple_steps() -> None:
    rssm = RecurrentStateSpaceModel(action_dim=2, embedding_dim=64, deterministic_dim=64, stochastic_dim=16)
    state = rssm.initial_state(batch_size=2)

    for _ in range(4):
        action = torch.randn(2, 2)
        embedding = torch.randn(2, 64)
        state = rssm.observe_step(state, action, embedding)

    assert state.deterministic is not None
    assert state.stochastic is not None
    assert state.deterministic.shape == (2, 64)
    assert state.stochastic.shape == (2, 16)
    assert state.features.shape == (2, 64)