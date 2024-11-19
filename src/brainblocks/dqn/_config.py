"""Configuration of parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from typing import Any, Self

import torch
from tianshou.data import Collector, VectorReplayBuffer


class TianshouDQNCartPoleModel(torch.nn.Module):
    """Model for the DQN agent and CartPole environment."""

    def __init__(self: Self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, n_actions),
        )

    def forward(
        self: Self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[Any, torch.Tensor]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


PARAMETERS = {
    'CartPole': {
        'tianshou': {
            'policy_model': lambda agent: TianshouDQNCartPoleModel(
                n_observations=agent.n_observations_,
                n_actions=agent.n_actions_,
            ),
            'policy_optim': lambda agent: torch.optim.Adam(params=agent.policy_model_.parameters(), lr=1e-3),
            'policy_discount_factor': 0.9,
            'policy_estimation_step': 3,
            'policy_target_update_freq': 320,
            'trainer_collector_train': lambda agent: Collector(
                policy=agent.policy_,
                env=agent.envs_,
                buffer=VectorReplayBuffer(20000, 10),
                exploration_noise=True,
            ),
            'trainer_collector_eval': lambda agent: Collector(
                policy=agent.policy_,
                env=agent.eval_envs_,
                exploration_noise=True,
            ),
            'trainer_max_epoch': 10,
            'trainer_step_per_epoch': 10000,
            'trainer_step_per_collect': 10,
            'trainer_update_per_step': 0.1,
            'trainer_episode_per_eval': 100,
            'trainer_batch_size': 64,
            'trainer_train_fn': lambda agent: (lambda epoch, env_step: agent.policy_.set_eps(0.1)),
            'trainer_eval_fn': lambda agent: (lambda epoch, env_step: agent.policy_.set_eps(0.05)),
            'trainer_stop_fn': lambda agent: (
                lambda mean_rewards: mean_rewards >= agent.base_env_.spec.reward_threshold
            ),
        },
    },
}
