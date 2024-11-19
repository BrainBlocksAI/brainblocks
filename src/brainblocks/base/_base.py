"""Implementation of base class of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from inspect import signature
from pathlib import Path
from typing import Any, ClassVar, Self, TypeAlias

import gymnasium as gym
import numpy as np
import rich
import tianshou
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_scalar
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

from ..logging import LoggingTensorboard, LoggingTerminal

EnvType: TypeAlias = (
    gym.Env | gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv | tianshou.env.BaseVectorEnv
)


def check_base_env(env: EnvType, param_name: str) -> gym.Env:
    """Check the base environment.

    Args:
        env:
            A simple or vectorized Gymnasium or Tianshou environment.

        param_name:
            The identifier of the environment.

    Returns:
        base_env:
            The base Gymnasium environment.
    """

    # Check environment type
    if not isinstance(env, EnvType):
        error_msg = (
            f'Parameter `{param_name}` should be an object from either `gym.Env` or `gym.vector.VectorEnv` or '
            f'`tianshou.env.BaseVectorEnv` class. Got {type(env)} instead.'
        )
        raise TypeError(error_msg)

    # Check specification
    if isinstance(env, gym.Env | gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv):
        spec = env.spec
        render_modes = env.metadata['render_modes']
    else:
        spec = env.spec[0]
        render_modes = env.metadata[0]['render_modes']
    if spec is None:
        error_msg = (
            f'Environment `{param_name}` has a `spec` attribute equal to `None`. Please '
            'use a `gymnasium` environment with a specification.'
        )
        raise ValueError(error_msg)

    # Create base environment
    base_env = gym.make(spec, render_mode='rgb_array') if 'rgb_array' in render_modes else gym.make(spec)

    return base_env


def get_writer(env_name: str, env_type: str, agent_name: str, logging_path: str) -> SummaryWriter:
    """Get the summary writer.

    Args:
        env_name:
            The environment's name.

        env_type:
            The type of environment.

        agent_name:
            The agent's name.

        logging_path:
            The path of the logging files.

    Returns:
        writer:
            The summary writer.
    """
    path = Path(logging_path) / 'runs'
    path.mkdir(exist_ok=True)
    env_paths = [env_path for env_path in path.iterdir() if env_path.name == env_name]
    env_path = env_paths[0] if env_paths else path / env_name
    env_train_path = env_path / env_type.title() / agent_name
    n_exps = len(list(env_train_path.iterdir())) if env_train_path.exists() else 0
    writer = SummaryWriter(env_train_path / f'Experiment {n_exps + 1}')
    return writer


def wrap_logging_env(
    env: gym.Env,
    logging_terminal: bool | dict | None,
    logging_tensorboard: bool | dict | None,
    agent_name: str,
    env_type: str,
) -> gym.Env:
    """Wrap the base environment to support logging.

    Args:
        env:
            The gymnasium environment to wrap.

        logging_terminal:
            Whether or not to log to terminal or the logging parameters.

        logging_tensorboard:
            Whether or not to log to Tensorboard or the logging parameters.

        agent_name:
            The agent's name.

        env_type:
            The type of environment.

    Returns:
        wrapped_env:
            The wrapped Gymnasium environment.
    """

    # Wrap environment
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

    # Check logging terminal
    if logging_terminal is None:
        logging_terminal = True
    if isinstance(logging_terminal, bool):
        if logging_terminal:
            wrapped_env = LoggingTerminal(env=wrapped_env, agent_name=agent_name, env_type=env_type)
    elif isinstance(logging_terminal, dict):
        wrapped_env = LoggingTerminal(env=wrapped_env, agent_name=agent_name, env_type=env_type, **logging_terminal)
    else:
        error_msg = 'Parameter `logging_terminal` should be either a boolean value or a dictionary.'
        raise TypeError(error_msg)

    # Check logging tensorboard
    if logging_tensorboard is None:
        logging_tensorboard = False
    if isinstance(logging_tensorboard, bool):
        if logging_tensorboard:
            writer = get_writer(
                env_name=env.spec.name,
                env_type=env_type,
                agent_name=agent_name,
                logging_path='.',
            )
            wrapped_env = LoggingTensorboard(env=wrapped_env, writer=writer)
        elif isinstance(logging_tensorboard, dict):
            writer = get_writer(
                env_name=env.spec.name,
                env_type=env_type,
                agent_name=agent_name,
                logging_path=logging_tensorboard.get('logging_path', '.'),
            )
            wrapped_env = LoggingTensorboard(
                env=wrapped_env,
                writer=writer,
                **logging_tensorboard,
            )
    else:
        error_msg = 'Parameter `logging_tensorboard` should be either a boolean value or a dictionary.'
        raise TypeError(error_msg)

    return wrapped_env


def extract_env_info(env: gym.Env) -> tuple[gym.spaces.Space, int, gym.spaces.Space, int]:
    """Extract information for the environment.

    Args:
        env:
            The gymnasium environment to extract info.

    Returns:
        info:
            The environment's spaces and their dimensionality.
    """
    observation_space = env.observation_space
    action_space = env.action_space
    n_observations = np.prod(observation_space.shape).astype(int) if observation_space.shape else observation_space.n
    n_actions = np.prod(action_space.shape).astype(int) if action_space.shape else action_space.n
    info = observation_space, n_observations, action_space, n_actions
    return info


def check_vectorized_envs(
    env: EnvType,
    wrapped_env: gym.Env,
    backend: str,
) -> EnvType:
    """Check the vectorized environments.

    Args:
        env:
            The gymnasium environment to wrap.

        wrapped_env:
            The wrapped Gymnasium environment.

        backend:
            The selected backend.

    Returns:
        vectorized_envs:
            The vectorized environments.
    """
    if isinstance(env, gym.Env):
        n_envs = 1
    elif isinstance(env, gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv):
        n_envs = env.num_envs
    elif isinstance(env, tianshou.env.BaseVectorEnv):
        n_envs = env.env_num
    if backend != 'tianshou':
        vectorized_envs = gym.make_vec(
            wrapped_env.spec,
            num_envs=n_envs,
            vectorization_mode='async' if n_envs > 1 else 'sync',
        )
    else:
        vectorized_envs = (
            tianshou.env.SubprocVectorEnv([lambda: wrapped_env for _ in range(n_envs)])
            if n_envs > 1
            else tianshou.env.DummyVectorEnv([lambda: wrapped_env])
        )
    return vectorized_envs


class BaseAgent(BaseEstimator):
    """Base class for agents."""

    _backends: ClassVar[list[str]] = ['native']
    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'backend': (str,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {}
    _default_parameters: ClassVar[dict[str, dict]] = {
        'native': {
            'backend': 'native',
        },
    }
    _optimal_parameters: ClassVar[dict[str, dict]] = {}

    MAX_EPISODES = 100

    def __init__(self: Self, backend: str | None = None) -> None:
        self.backend = backend

    def _check_backend(self: Self) -> Self:
        if not self._backends:
            error_msg = 'No available backends are set for this agent.'
            raise ValueError(error_msg)
        if self.backend is not None and self.backend not in self._backends:
            error_msg = (
                f'Parameter `{self.backend}` should be one of {", ".join(self._backend)} or `None`. '
                f'Got {self.backend} instead.'
            )
            raise ValueError(error_msg)
        return self

    def _check_params_attrs(self: Self) -> Self:

        # Check default values of initialization
        error_msg = 'All parameters default value in agent\'s initialization method should be `None`.'
        assert all(param.default is None for param in signature(self.__init__).parameters.values()), error_msg  # type: ignore[misc]

        # Non numeric
        for param_name, param_types in self._non_numeric_params.items():
            attr_val = getattr(self, param_name)
            if attr_val is not None and not isinstance(attr_val, param_types):
                error_instance_msg = ' or '.join([str(attr) for attr in param_types]) + ' or `None`'
                error_msg = (
                    f'Parameter `{param_name}` should be an instance of '
                    f'{error_instance_msg}. Got {type(attr_val)} instead.'
                )
                raise TypeError(error_msg)
            if hasattr(self, f'_check_{param_name}'):
                getattr(self, f'_check_{param_name}')()

        # Numeric
        for param_name, param_info in self._numeric_params.items():
            attr_val = getattr(self, param_name)
            if attr_val is not None:
                check_scalar(
                    attr_val,
                    name=param_name,
                    target_type=param_info['target_type'],
                    min_val=param_info.get('min_val'),
                    max_val=param_info.get('max_val'),
                )

        # Get backend
        backend = self._backends[0] if self.backend is None else self.backend

        # Get default parameters
        default_params = self._default_parameters.get(backend, {})
        if default_params:
            error_msg = 'Provided default parameters are not complete.'
            assert sorted(default_params) == sorted(signature(self.__init__).parameters), error_msg  # type: ignore[misc]

        # Get environment specific parameters
        params_env = self._optimal_parameters.get(self.base_env_.spec.name, {}).get(backend, {})

        # Create attributes
        for param_name, param_val_default in default_params.items():
            param_val = getattr(self, param_name)
            if param_val is None:
                param_val = params_env.get(param_name, param_val_default)
            setattr(self, f'{param_name}_', param_val)

        return self

    def _launch_tensorboard(self: Self, logging_tensorboard: bool | dict | None) -> None:
        if (isinstance(logging_tensorboard, bool) and logging_tensorboard) or isinstance(logging_tensorboard, dict):
            tensorboard = program.TensorBoard()
            base_path = logging_tensorboard.get('logging_path', '.') if isinstance(logging_tensorboard, dict) else '.'
            path = Path(base_path) / 'runs'
            tensorboard.configure(argv=[None, '--logdir', str(path)])
            tensorboard_url = tensorboard.launch()
            rich.print(tensorboard_url)

    def _learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
    ) -> Self:

        # Base environments
        self.base_env_ = check_base_env(env, 'env')
        eval_env = eval_env if eval_env is not None else env
        self.base_eval_env_ = check_base_env(eval_env, 'eval_env')

        # Check attributes
        self._check_params_attrs()

        # Check wrapped environments
        self.wrapped_env_ = wrap_logging_env(
            self.base_env_,
            logging_terminal,
            logging_tensorboard,
            self.__class__.__name__,
            'Learning',
        )
        self.wrapped_eval_env_ = wrap_logging_env(
            self.base_eval_env_,
            logging_terminal,
            logging_tensorboard,
            self.__class__.__name__,
            'Evaluation',
        )
        self._launch_tensorboard(logging_tensorboard)

        # Check vectorized environments
        self.envs_ = check_vectorized_envs(env, self.wrapped_env_, self.backend_)
        self.eval_envs_ = check_vectorized_envs(eval_env, self.wrapped_eval_env_, self.backend_)

        # Environment features
        self.observation_space_, self.n_observations_, self.action_space_, self.n_actions_ = extract_env_info(
            self.wrapped_env_,
        )

        # Results
        self.learn_results_ = None

        # Policy
        self.policy_ = None

        return self

    def _interact(
        self: Self,
        env: EnvType,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
        n_episodes: int | None,
        n_steps: int | None,
        **kwargs: dict,
    ) -> tuple[EnvType, int | None, int | None]:

        # Check agent is fitted
        if (
            not hasattr(self, 'learn_results_')
            or self.learn_results_ is None
            or not hasattr(self, 'policy_')
            or self.policy_ is None
        ):
            error_msg = (
                f'The `{self.__class__.__name__}` instance has not learned from interacting with the environment. '
                'Call `learn` with appropriate arguments before using this agent.'
            )
            raise NotFittedError(error_msg)

        # Check environment
        base_env = check_base_env(env, 'env')
        wrapped_env = wrap_logging_env(
            base_env,
            logging_terminal,
            logging_tensorboard,
            self.__class__.__name__,
            'Interaction',
        )
        envs = check_vectorized_envs(env, wrapped_env, self.backend_)

        # Check parameters
        max_episodes = 100
        if n_steps is None and n_episodes is None:
            n_episodes = base_env.spec.max_episode_steps if base_env.spec is not None else max_episodes
        elif n_episodes is None:
            check_scalar(n_steps, 'n_steps', int, min_val=1)
        elif n_steps is None:
            check_scalar(n_episodes, 'n_episodes', int, min_val=1)

        return envs, n_episodes, n_steps

    def learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None = None,
        logging_terminal: bool | dict | None = None,
        logging_tensorboard: bool | dict | None = None,
    ) -> Self:
        """Learn from online or offline interaction with the environment."""
        return self._learn(env, eval_env, logging_terminal, logging_tensorboard)

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
        envs, n_episodes, n_steps = self._interact(
            env,
            logging_terminal,
            logging_tensorboard,
            n_episodes,
            n_steps,
            **kwargs,
        )
        return {}
