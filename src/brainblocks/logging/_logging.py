"""Implementation of logging function and classes."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import warnings
from collections.abc import Callable
from typing import Any, Self, SupportsFloat

import gymnasium as gym
import numpy as np
import rich.layout

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import pygame
import rich
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_video

TRIGGER_EPISODE_NUM = 50


class LoggingTerminal(gym.Wrapper, gym.utils.RecordConstructorArgs):

    layout = rich.layout.Layout()
    layout.split_row(
        rich.layout.Layout(name='Learning'),
        rich.layout.Layout(name='Evaluation'),
    )
    layout['Learning'].visible = False
    layout['Evaluation'].visible = False

    def __init__(
        self: Self,
        env: gym.Env,
        terminal_episode_trigger: Callable[[int], bool] | None = None,
        window_size: int | None = None,
        agent_name: str | None = None,
        env_type: str | None = None,
    ) -> None:
        super().__init__(env)
        self.terminal_episode_trigger = (
            (lambda num: num % TRIGGER_EPISODE_NUM == 0)
            if terminal_episode_trigger is None
            else terminal_episode_trigger
        )
        self.window_size = TRIGGER_EPISODE_NUM // 2 if window_size is None else window_size
        self.agent_name = 'Unknown' if agent_name is None else agent_name.removesuffix('Agent')
        self.env_type = env_type if env_type is not None else 'Learning'
        env_name = self.env.spec.name if hasattr(self.env.spec, 'name') else 'Unknown'
        title = f'{self.agent_name} agent, {env_name} environment'
        if self.env_type is not None:
            title = f'\n{title}\n{self.env_type}'
        caption = f'\nMetrics are moving averages of the latest {self.window_size} episodes'
        self.table = rich.table.Table(title=title, min_width=max(len(title), len(caption)), caption=caption)
        self.table.add_column('Episode')
        self.table.add_column('Cumulative Reward')
        self.table.add_column('Length')
        self.last_episode_num = None

    def reset(
        self: Self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(
        self: Self,
        action: gym.core.ActType,
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        episode_num = self.episode_count
        if self.terminal_episode_trigger(episode_num) and self.last_episode_num != episode_num:
            self.last_episode_num = episode_num
            self.table.add_row(
                str(episode_num),
                str(np.mean(list(self.return_queue)[-self.window_size :])),
                str(np.mean(list(self.length_queue)[-self.window_size :])),
            )
            self.layout[self.env_type].update(self.table)
            self.layout[self.env_type].visible = True
            rich.print(self.layout)
        return super().step(action)


class LoggingTensorboard(gym.wrappers.RecordVideo):

    def __init__(
        self: Self,
        env: gym.Env,
        writer: SummaryWriter | None = None,
        video_folder: str | None = None,
        video_episode_trigger: Callable[[int], bool] | None = None,
        video_step_trigger: Callable[[int], bool] | None = None,
        video_length: int | None = None,
        video_name_prefix: str | None = None,
    ) -> None:
        path = 'runs' if writer is None else str(writer.get_logdir())
        self.writer = SummaryWriter(log_dir=path) if writer is None else writer
        self.video_folder = path if video_folder is None else video_folder
        self.video_episode_trigger = (
            (lambda num: num % TRIGGER_EPISODE_NUM == 0)
            if (video_episode_trigger is None and video_step_trigger is None)
            else video_episode_trigger
        )
        self.video_step_trigger = video_step_trigger
        self.video_length = 0 if video_length is None else video_length
        self.video_name_prefix = 'video' if video_name_prefix is None else video_name_prefix
        super().__init__(
            env=env,
            video_folder=self.video_folder,
            episode_trigger=self.video_episode_trigger,
            step_trigger=self.video_step_trigger,
            video_length=self.video_length,
            name_prefix=self.video_name_prefix,
            disable_logger=True,
        )

    def close_video_recorder(self: Self) -> None:
        if self.video_recorder is not None:
            video_tensor = read_video(self.video_recorder.path, output_format='tchw', pts_unit='sec')[0]
            if video_tensor.nelement() > 0:
                video_tensor = video_tensor.reshape(-1, *video_tensor.shape)
                self.writer.add_video(f'Video of episode {self.episode_id}', video_tensor)
        super().close_video_recorder()
        pygame.init()

    def step(
        self: Self,
        action: gym.core.ActType,
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        if self.return_queue:
            self.writer.add_scalar('Cumulative Reward', self.return_queue[-1][0], self.episode_id)
        return observation, reward, terminated, truncated, info

    def close(self: Self) -> None:
        super().close()
        self.writer.flush()
        self.writer.close()
