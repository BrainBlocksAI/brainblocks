"""Implementation of Q-learning algorithm and its variants."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import warnings
from collections.abc import Callable
from typing import Any, ClassVar, Self, cast

import torch
from sklearn.utils import check_scalar
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.lr_scheduler import MultipleLRSchedulers
from tianshou.utils.net.common import Net
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseAgent, EnvType
from ._config import PARAMETERS

warnings.filterwarnings('ignore')


class TianshouModel(torch.nn.Module):
    """Default model for the DQN agent."""

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


class DQNAgent(BaseAgent):
    """Implementation of Deep Q Network and its variants."""

    _backends: ClassVar[list[str]] = ['tianshou']
    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'policy_model': (torch.nn.Module, Net),
        'policy_optim': (torch.optim.Optimizer,),
        'policy_lr_scheduler': (LRScheduler, MultipleLRSchedulers),
        'policy_reward_normalization': (bool,),
        'policy_is_double': (bool,),
        'policy_clip_loss_grad': (bool,),
        'trainer_collector_train': (Collector,),
        'trainer_collector_eval': (Collector,),
        'trainer_buffer': (ReplayBuffer,),
        'trainer_train_fn': (Callable,),
        'trainer_eval_fn': (Callable,),
        'trainer_stop_fn': (Callable,),
        'trainer_save_best_fn': (Callable,),
        'trainer_save_checkpoint_fn': (Callable,),
        'trainer_resume_from_log': (bool,),
        'trainer_reward_metric': (Callable,),
        'trainer_eval_in_train': (bool,),
        'backend': (str,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {
        'policy_discount_factor': {'target_type': float, 'min_val': 0.0, 'max_val': 1.0},
        'policy_estimation_step': {'target_type': int, 'min_val': 0},
        'policy_target_update_freq': {'target_type': int, 'min_val': 0},
        'trainer_max_epoch': {'target_type': int, 'min_val': 0},
        'trainer_batch_size': {'target_type': int, 'min_val': 0},
        'trainer_step_per_epoch': {'target_type': int, 'min_val': 0},
        'trainer_repeat_per_collect': {'target_type': int, 'min_val': 0},
        'trainer_episode_per_eval': {'target_type': int, 'min_val': 0},
        'trainer_update_per_step': {'target_type': float, 'min_val': 0},
        'trainer_step_per_collect': {'target_type': int, 'min_val': 0},
        'trainer_episode_per_collect': {'target_type': int, 'min_val': 0},
    }
    _default_parameters: ClassVar[dict[str, dict]] = {
        'tianshou': {
            'policy_model': lambda agent: TianshouModel(agent.n_observations_, agent.n_actions_),
            'policy_optim': lambda agent: torch.optim.Adam(agent.policy_model_.parameters()),
            'policy_discount_factor': 0.99,
            'policy_estimation_step': 1,
            'policy_target_update_freq': 10,
            'policy_reward_normalization': False,
            'policy_is_double': False,
            'policy_clip_loss_grad': False,
            'policy_lr_scheduler': None,
            'trainer_collector_train': lambda agent: Collector(policy=agent.policy_, env=agent.envs_),
            'trainer_collector_eval': lambda agent: Collector(policy=agent.policy_, env=agent.envs_),
            'trainer_max_epoch': 100,
            'trainer_batch_size': None,
            'trainer_buffer': None,
            'trainer_step_per_epoch': 100,
            'trainer_repeat_per_collect': None,
            'trainer_episode_per_eval': 10,
            'trainer_update_per_step': 1.0,
            'trainer_step_per_collect': 10,
            'trainer_episode_per_collect': None,
            'trainer_train_fn': None,
            'trainer_eval_fn': None,
            'trainer_stop_fn': lambda agent: (lambda mean_rewards: mean_rewards >= agent.env_.spec.reward_threshold),
            'trainer_save_best_fn': None,
            'trainer_save_checkpoint_fn': None,
            'trainer_resume_from_log': False,
            'trainer_reward_metric': None,
            'trainer_eval_in_train': True,
            'backend': 'tianshou',
        },
    }
    _optimal_parameters: ClassVar[dict[str, dict]] = PARAMETERS

    def __init__(
        self: Self,
        policy_model: torch.nn.Module | Net | None = None,
        policy_optim: torch.optim.Optimizer | None = None,
        policy_discount_factor: float | None = None,
        policy_estimation_step: int | None = None,
        policy_target_update_freq: int | None = None,
        policy_reward_normalization: bool | None = None,
        policy_is_double: bool | None = None,
        policy_clip_loss_grad: bool | None = None,
        policy_lr_scheduler: LRScheduler | MultipleLRSchedulers | None = None,
        trainer_collector_train: Collector | None = None,
        trainer_collector_eval: Collector | None = None,
        trainer_max_epoch: int | None = None,
        trainer_batch_size: int | None = None,
        trainer_buffer: ReplayBuffer | None = None,
        trainer_step_per_epoch: int | None = None,
        trainer_repeat_per_collect: int | None = None,
        trainer_episode_per_eval: int | None = None,
        trainer_update_per_step: float | None = None,
        trainer_step_per_collect: int | None = None,
        trainer_episode_per_collect: int | None = None,
        trainer_train_fn: Callable | None = None,
        trainer_eval_fn: Callable | None = None,
        trainer_stop_fn: Callable | None = None,
        trainer_save_best_fn: Callable | None = None,
        trainer_save_checkpoint_fn: Callable | None = None,
        trainer_resume_from_log: bool | None = None,
        trainer_reward_metric: Callable | None = None,
        trainer_eval_in_train: bool | None = None,
        backend: str | None = None,
    ) -> None:
        super().__init__(backend=backend)
        self.policy_model = policy_model
        self.policy_optim = policy_optim
        self.policy_discount_factor = policy_discount_factor
        self.policy_estimation_step = policy_estimation_step
        self.policy_target_update_freq = policy_target_update_freq
        self.policy_reward_normalization = policy_reward_normalization
        self.policy_is_double = policy_is_double
        self.policy_clip_loss_grad = policy_clip_loss_grad
        self.policy_lr_scheduler = policy_lr_scheduler
        self.trainer_collector_train = trainer_collector_train
        self.trainer_collector_eval = trainer_collector_eval
        self.trainer_max_epoch = trainer_max_epoch
        self.trainer_batch_size = trainer_batch_size
        self.trainer_buffer = trainer_buffer
        self.trainer_step_per_epoch = trainer_step_per_epoch
        self.trainer_repeat_per_collect = trainer_repeat_per_collect
        self.trainer_episode_per_eval = trainer_episode_per_eval
        self.trainer_update_per_step = trainer_update_per_step
        self.trainer_step_per_collect = trainer_step_per_collect
        self.trainer_episode_per_collect = trainer_episode_per_collect
        self.trainer_train_fn = trainer_train_fn
        self.trainer_eval_fn = trainer_eval_fn
        self.trainer_stop_fn = trainer_stop_fn
        self.trainer_save_best_fn = trainer_save_best_fn
        self.trainer_save_checkpoint_fn = trainer_save_checkpoint_fn
        self.trainer_resume_from_log = trainer_resume_from_log
        self.trainer_reward_metric = trainer_reward_metric
        self.trainer_eval_in_train = trainer_eval_in_train

    def _learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
    ) -> Self:

        super()._learn(env, eval_env, logging_terminal, logging_tensorboard)

        # Check policy
        self.policy_model_: torch.nn.Module | Net = self.policy_model_(self)
        self.policy_optim_: torch.optim.Optimizer = self.policy_optim_(self)
        self.policy_ = DQNPolicy(
            model=self.policy_model_,
            optim=self.policy_optim_,
            discount_factor=self.policy_discount_factor_,
            estimation_step=self.policy_estimation_step_,
            target_update_freq=self.policy_target_update_freq_,
            reward_normalization=self.policy_reward_normalization_,
            is_double=self.policy_is_double_,
            clip_loss_grad=self.policy_clip_loss_grad_,
            lr_scheduler=self.policy_lr_scheduler_,
            observation_space=self.observation_space_,
            action_space=self.action_space_,
        )

        # Check trainer
        self.trainer_collector_train_: Collector = self.trainer_collector_train_(self)
        self.trainer_collector_eval_: Collector = self.trainer_collector_eval_(self)
        if self.trainer_train_fn_ is not None:
            self.trainer_train_fn_: Callable = self.trainer_train_fn_(self)
        if self.trainer_eval_fn_ is not None:
            self.trainer_eval_fn_: Callable = self.trainer_eval_fn_(self)
        if self.trainer_stop_fn_ is not None:
            self.trainer_stop_fn_: Callable = self.trainer_stop_fn_(self)
        if self.trainer_save_best_fn_ is not None:
            self.trainer_save_best_fn_: Callable = self.trainer_save_best_fn_(self)
        if self.trainer_save_checkpoint_fn_ is not None:
            self.trainer_save_checkpoint_fn_: Callable = self.trainer_save_checkpoint_fn_(self)
        if self.trainer_reward_metric_ is not None:
            self.trainer_reward_metric_: Callable = self.trainer_reward_metric_(self)
        self.trainer_ = OffpolicyTrainer(
            policy=self.policy_,
            train_collector=self.trainer_collector_train_,
            test_collector=self.trainer_collector_eval_,
            max_epoch=self.trainer_max_epoch_,
            batch_size=self.trainer_batch_size_,
            buffer=self.trainer_buffer_,
            step_per_epoch=self.trainer_step_per_epoch_,
            repeat_per_collect=self.trainer_repeat_per_collect_,
            episode_per_test=self.trainer_episode_per_eval_,
            update_per_step=self.trainer_update_per_step_,
            step_per_collect=self.trainer_step_per_collect_,
            episode_per_collect=self.trainer_episode_per_collect_,
            train_fn=self.trainer_train_fn_,
            test_fn=self.trainer_eval_fn_,
            stop_fn=self.trainer_stop_fn_,
            save_best_fn=self.trainer_save_best_fn_,
            save_checkpoint_fn=self.trainer_save_checkpoint_fn_,
            resume_from_log=self.trainer_resume_from_log_,
            reward_metric=self.trainer_reward_metric_,
            verbose=False,
            show_progress=False,
            test_in_train=self.trainer_eval_in_train_,
        )

        # Run trainer
        self.learn_results_ = self.trainer_.run()

        return self

    def interact(
        self: Self,
        env: EnvType,
        logging_terminal: bool | dict | None = None,
        logging_tensorboard: bool | dict | None = None,
        n_episodes: int | None = None,
        n_steps: int | None = None,
        **kwargs: dict,
    ) -> dict[str, Any]:
        """Interact with the environment."""

        envs, n_episodes, n_steps = self._interact(env, logging_terminal, logging_tensorboard, n_episodes, n_steps)

        # Keyword arguments
        render_time = kwargs.get('render_time')
        if render_time is None:
            render_time = 1 / env.get_env_attr('metadata')[0].get('render_fps', 10.0)
        check_scalar(render_time, 'render_time', float)
        reset_kwargs = {k: v for k, v in kwargs.items() if k != 'render_time'}

        # Interact
        cast(DQNPolicy, self.policy_).eval()
        collector = Collector(self.policy_, envs)
        return collector.collect(
            n_step=n_steps,
            n_episode=n_episodes,
            random=False,
            render=render_time,
            reset_before_collect=True,
            gym_reset_kwargs=reset_kwargs,
        )
