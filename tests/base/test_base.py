"""Test the base agent class."""

from typing import Any, ClassVar, Self

import gymnasium as gym
import pytest

from brainblocks.base import BaseAgent, EnvType

ENV = gym.wrappers.FlattenObservation(gym.make('Blackjack-v1'))
DEFAULT_PARAMETERS = {
    'backend': 'native',
    'param_str': 'test',
    'param_int': 1,
    'param_bool': False,
    'param_float': 1.0,
}


class Agent(BaseAgent):
    """Test agent."""

    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'param_str': (str,),
        'param_bool': (bool,),
        'backend': (str,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {
        'param_int': {'target_type': int, 'min_val': 1},
        'param_float': {'target_type': float, 'min_val': 1.0, 'max_val': 6.0},
    }
    _default_parameters: ClassVar[dict[str, dict]] = {'native': DEFAULT_PARAMETERS}

    def __init__(
        self,
        param_str: str | None = None,
        param_int: int | None = None,
        param_bool: bool | None = None,
        param_float: float | None = None,
        backend: str | None = None,
    ) -> None:
        """Initialize the agent."""
        super().__init__(backend=backend)
        self.param_str = param_str
        self.param_int = param_int
        self.param_bool = param_bool
        self.param_float = param_float

    def _learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
    ) -> Self:
        super()._learn(env, eval_env, logging_terminal, logging_tensorboard)
        self.policy_: dict[str, str] = {}
        self.learn_results_: dict[str, float] = {}
        return self


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'param_int': 9, 'param_float': 5.0},
        {'param_str': 'test', 'param_int': 3, 'param_bool': True, 'param_float': 4.0},
    ],
)
def test_init(params):
    """Test the initialization of agent with parameters."""
    agent = Agent(**params)
    assert agent.param_str is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool is None if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float is None if params.get('param_float') is None else params['param_float']


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'param_int': 9, 'param_float': 5.0},
        {'param_str': 'test', 'param_int': 3, 'param_bool': True, 'param_float': 4.0},
    ],
)
def test_learn(params):
    """Test the learn method."""
    agent = Agent(**params)
    agent.learn(ENV)
    assert agent.param_str is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool is None if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float is None if params.get('param_float') is None else params['param_float']
    assert agent.param_str_ == DEFAULT_PARAMETERS['param_str'] if agent.param_str is None else agent.param_str
    assert agent.param_int_ == DEFAULT_PARAMETERS['param_int'] if agent.param_int is None else agent.param_int
    assert agent.param_bool_ == DEFAULT_PARAMETERS['param_bool'] if agent.param_bool is None else agent.param_bool
    assert agent.param_float_ == DEFAULT_PARAMETERS['param_float'] if agent.param_float is None else agent.param_float
    assert isinstance(agent.observation_space_, gym.spaces.Box)
    assert isinstance(agent.action_space_, gym.spaces.Discrete)
    assert agent.n_actions_ == agent.action_space_.n
    assert agent.learn_results_ == {}
    assert isinstance(agent.policy_, dict)
    assert isinstance(agent.base_env_, gym.Env)
    assert isinstance(agent.base_eval_env_, gym.Env)
    assert isinstance(agent.envs_, EnvType)
    assert isinstance(agent.eval_envs_, EnvType)
    assert agent.envs_.num_envs == 1
    assert agent.eval_envs_.num_envs == 1


def test_interact():
    """Test the interact method."""
    agent = Agent()
    agent.learn(ENV)
    assert agent.interact(ENV) == {}
