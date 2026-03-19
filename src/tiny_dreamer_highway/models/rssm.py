"""Recurrent State-Space Model (RSSM) — the latent dynamics core of DreamerV1.

The RSSM maintains a two-part latent state at every time step:

* **Deterministic state** (``h_t``): carries long-term memory through a
  GRU cell.  Shape: ``(B, deterministic_dim)``.
* **Stochastic state** (``s_t``): captures per-step uncertainty via a
  learned diagonal Gaussian.  Shape: ``(B, stochastic_dim)``.

Two distributions are defined over ``s_t``:

1. **Prior** — conditioned only on the previous latent state and the
   action (no observation).  Used during *imagination* rollouts where
   the agent predicts future latent states without environment access.
2. **Posterior** — additionally conditioned on the CNN embedding of the
   current observation.  Used during *training* to produce a more
   accurate latent state that the prior is trained to match (via KL
   divergence).

The concatenation ``[h_t ; s_t]`` is called the *latent feature vector*
and is consumed by the decoder, reward predictor, continue predictor,
actor, and critic.

Reference: Hafner et al., "Dream to Control: Learning Behaviors by
Latent Imagination" (ICLR 2020), Section 2 — Latent Dynamics.

Name: Esteban Montelongo
Course: CSC 580 AI 2
Assignment: Final Project — Dream the Road
AI tools consulted: GitHub Copilot
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tiny_dreamer_highway.models.encoder import LatentState


class RecurrentStateSpaceModel(nn.Module):
    """GRU-backed recurrent state-space model for DreamerV1.

    The forward dynamics are:

    1. Concatenate the previous stochastic state ``s_{t-1}`` with the
       action ``a_{t-1}`` and project through a small MLP (``input_layer``).
    2. Feed the projection into the GRU to obtain the new deterministic
       state ``h_t``.
    3. Compute a Gaussian distribution over ``s_t``:
       - **Prior**: ``p(s_t | h_t)`` — uses ``prior_model``.
       - **Posterior**: ``q(s_t | h_t, e_t)`` — uses ``posterior_model``,
         where ``e_t`` is the CNN embedding of the observation.
    4. Sample ``s_t`` via the reparameterization trick so gradients
       flow through sampling.

    Args:
        action_dim:       Dimensionality of the (one-hot or continuous) action.
        embedding_dim:    Dimensionality of the CNN encoder output ``e_t``.
        deterministic_dim: Width of the GRU hidden state ``h_t``.
        stochastic_dim:   Width of the stochastic state ``s_t``.
        hidden_dim:       Width of the hidden layers in the prior/posterior MLPs.
        min_std:          Floor on the predicted standard deviation to avoid
                          posterior collapse.
        num_layers:       Number of hidden layers in the prior/posterior MLPs.
    """

    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        deterministic_dim: int = 200,
        stochastic_dim: int = 30,
        hidden_dim: int = 200,
        min_std: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if deterministic_dim <= 0:
            raise ValueError("deterministic_dim must be positive")
        if stochastic_dim <= 0:
            raise ValueError("stochastic_dim must be positive")

        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        # ── Input projection: [s_{t-1} ; a_{t-1}] → hidden_dim ────────
        self.input_layer = nn.Sequential(
            nn.Linear(action_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
        )

        # ── Deterministic backbone ──────────────────────────────────────
        self.gru = nn.GRUCell(hidden_dim, deterministic_dim)

        # ── Prior p(s_t | h_t): predicts stochastic state without obs ──
        # Output is 2·stochastic_dim: first half = mean, second half =
        # pre-softplus std parameter.
        self.prior_model = self._build_fc_network(
            deterministic_dim, hidden_dim, 2 * stochastic_dim, num_layers,
        )

        # ── Posterior q(s_t | h_t, e_t): refines with observation ───────
        self.posterior_model = self._build_fc_network(
            deterministic_dim + embedding_dim, hidden_dim, 2 * stochastic_dim, num_layers,
        )

        # Lightweight buffer used solely to track the module's current dtype
        self.register_buffer('_dtype_buf', torch.zeros(1), persistent=False)

    @staticmethod
    def _build_fc_network(
        in_dim: int, hidden_dim: int, out_dim: int, num_layers: int,
    ) -> nn.Sequential:
        """Build a fully connected network with ``num_layers`` hidden layers."""
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ELU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    @property
    def _dtype(self) -> torch.dtype:
        return self._dtype_buf.dtype

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        """Return an all-zeros latent state to bootstrap the recurrence.

        Both ``h_0`` and ``s_0`` are zero-initialized, following the
        convention from the DreamerV1 reference implementation.

        Returns:
            LatentState with shapes ``(B, deterministic_dim)`` and
            ``(B, stochastic_dim)``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        _dt = self._dtype
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device, dtype=_dt)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device, dtype=_dt)
        return LatentState(deterministic=deterministic, stochastic=stochastic)

    def _distribution_parameters(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        """Split raw network output into Gaussian mean and std.

        The network outputs ``2 * stochastic_dim`` values.  The first half
        is the mean (unconstrained).  The second half is mapped through
        softplus and floored at ``min_std`` to guarantee a valid, non-
        degenerate standard deviation.
        """
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        # softplus keeps std positive; min_std prevents posterior collapse
        std = torch.nn.functional.softplus(std_param) + self.min_std
        return mean, std

    def _sample_stochastic(self, mean: Tensor, std: Tensor) -> Tensor:
        """Reparameterized Gaussian sample: ``mean + std * N(0, I)``.

        The reparameterization trick allows gradients to flow through the
        sampling operation and back into the distribution parameters.
        """
        noise = torch.randn_like(mean)
        return mean + std * noise

    def _next_deterministic(self, prev_state: LatentState, action: Tensor) -> Tensor:
        """Advance the deterministic GRU backbone by one step.

        Computes ``h_t = GRU(MLP([s_{t-1}; a_{t-1}]), h_{t-1})``.
        This is shared between the prior (``imagine_step``) and posterior
        (``observe_step``) pathways — the difference is what happens
        *after* ``h_t`` is obtained.

        Returns:
            New deterministic state ``h_t`` with shape ``(B, deterministic_dim)``.
        """
        if prev_state.stochastic is None or prev_state.deterministic is None:
            raise ValueError("prev_state must contain stochastic and deterministic tensors")

        # Cast to model dtype (relevant under AMP mixed precision)
        _dt = self._dtype
        stochastic = prev_state.stochastic.to(dtype=_dt)
        deterministic = prev_state.deterministic.to(dtype=_dt)
        action = action.to(dtype=_dt)

        # Concatenate previous stochastic state with the action, project,
        # then update the GRU hidden state
        gru_input = self.input_layer(torch.cat([stochastic, action], dim=-1))
        return self.gru(gru_input, deterministic)

    def imagine_step(self, prev_state: LatentState, action: Tensor) -> LatentState:
        """One-step *prior* transition — no observation required.

        Used during imagination rollouts (behavior learning) where the
        agent unrolls its learned dynamics model to predict future states
        and evaluate candidate action sequences.

        Flow:  ``(h_{t-1}, s_{t-1}, a_{t-1})``  →  ``h_t``  →  ``p(s_t|h_t)``  →  sample ``s_t``

        Returns:
            LatentState containing ``h_t``, sampled ``s_t``, and the
            prior distribution parameters (``dist_mean``, ``dist_std``).
        """
        deterministic = self._next_deterministic(prev_state, action)

        # Prior: predict stochastic state from deterministic alone
        prior_stats = self.prior_model(deterministic)
        prior_mean, prior_std = self._distribution_parameters(prior_stats)
        stochastic = self._sample_stochastic(prior_mean, prior_std)

        return LatentState(
            deterministic=deterministic,
            stochastic=stochastic,
            dist_mean=prior_mean,
            dist_std=prior_std,
        )

    def imagine_rollout(self, start_state: LatentState, actions: Tensor) -> list[LatentState]:
        """Unroll the prior for ``T`` steps to produce an imagined trajectory.

        This is the core of DreamerV1 behavior learning: the actor proposes
        actions, the RSSM predicts future states using only the prior, and
        the critic evaluates the resulting imagined trajectory.

        Args:
            start_state: Initial latent state ``(h_0, s_0)``.
            actions:     Planned action tensor of shape ``(B, T, action_dim)``.

        Returns:
            List of ``T`` LatentStates, one per time step.
        """
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, T, action_dim)")
        if start_state.deterministic is None or start_state.stochastic is None:
            raise ValueError("start_state must contain deterministic and stochastic tensors")

        state = start_state
        rollout: list[LatentState] = []
        for step in range(actions.shape[1]):
            state = self.imagine_step(state, actions[:, step])
            rollout.append(state)
        return rollout

    def observe_step(self, prev_state: LatentState, action: Tensor, embedding: Tensor) -> LatentState:
        """One-step *posterior* transition — uses the real observation.

        Used during world-model training to compute a more accurate latent
        state that incorporates the CNN embedding of the actual observation.
        The KL loss encourages the prior to match this posterior so that
        imagination rollouts (which only use the prior) stay accurate.

        Flow:  ``(h_{t-1}, s_{t-1}, a_{t-1})``  →  ``h_t``  →  ``q(s_t | h_t, e_t)``  →  sample ``s_t``

        Args:
            prev_state: Latent state from the previous time step.
            action:     Action taken at the previous step, shape ``(B, action_dim)``.
            embedding:  CNN encoder output for the current observation,
                        shape ``(B, embedding_dim)``.

        Returns:
            LatentState containing ``h_t``, sampled ``s_t``, the posterior
            distribution parameters, and the observation embedding.
        """
        deterministic = self._next_deterministic(prev_state, action)

        # Posterior: condition on both h_t and the observation embedding e_t
        embedding = embedding.to(dtype=self._dtype)
        posterior_stats = self.posterior_model(torch.cat([deterministic, embedding], dim=-1))
        posterior_mean, posterior_std = self._distribution_parameters(posterior_stats)
        stochastic = self._sample_stochastic(posterior_mean, posterior_std)

        return LatentState(
            embedding=embedding,
            deterministic=deterministic,
            stochastic=stochastic,
            dist_mean=posterior_mean,
            dist_std=posterior_std,
        )