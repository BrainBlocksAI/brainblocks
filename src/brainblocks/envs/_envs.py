"""Implementation of environments load functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from typing import Literal, get_args

import gymnasium as gym
from sklearn.utils import check_scalar
from tianshou.env import BaseVectorEnv, DummyVectorEnv, RayVectorEnv, ShmemVectorEnv, SubprocVectorEnv

VECTORIZERS = Literal['dummy', 'ray', 'shmem', 'subproc']


def _check_loading(
    name: str,
    version: str | None,
    n_envs: int = 1,
    n_eval_envs: int | None = None,
    n_test_envs: int | None = None,
    vectorizer: VECTORIZERS | None = None,
) -> tuple[str, type[BaseVectorEnv]]:
    names = sorted({env.split('-')[0] for env in gym.registry})
    if name not in names:
        error_msg = 'Call the function `gymnasium.pprint_registry()` to print a list of available names and versions.'
        raise ValueError(error_msg)
    versions = [env_name.split('-')[1] for env_name in gym.envs.registry if env_name.split('-')[0] == name]
    if version is None:
        version = versions[-1]
    elif version not in versions:
        error_msg = f'Available versions for environment {name} are {", ".join(versions)}. Got `{version}` instead.'
        raise ValueError(error_msg)
    vectorizers = get_args(VECTORIZERS)
    if vectorizer is None:
        vectorizer = 'dummy'
    if vectorizer not in vectorizers:
        error_msg = f'Parameter `vectorizer` should be one of {", ".join(vectorizers)}. Got {vectorizer} instead.'
        raise ValueError(error_msg)
    vectorizers_mapping = {
        'dummy': DummyVectorEnv,
        'ray': RayVectorEnv,
        'shmem': ShmemVectorEnv,
        'subproc': SubprocVectorEnv,
    }
    vectorizer_class = vectorizers_mapping[vectorizer]
    check_scalar(n_envs, 'n_envs', int, min_val=1)
    if n_eval_envs is not None:
        check_scalar(n_eval_envs, 'n_eval_envs', int, min_val=1)
    if n_test_envs is not None:
        check_scalar(n_test_envs, 'n_test_envs', int, min_val=1)
    return version, vectorizer_class


def load_env_data(
    name: str,
    version: str | None = None,
    n_envs: int = 1,
    n_eval_envs: int | None = None,
    n_test_envs: int | None = None,
    vectorizer: VECTORIZERS | None = None,
    backend: Literal['gymnasium', 'tianshou'] | None = None,
) -> tuple[gym.Env, gym.Env | None, gym.Env | None]:
    version, vectorizer_class = _check_loading(name, version, n_envs, n_eval_envs, n_test_envs, vectorizer)
    base_env = gym.make(f'{name}-{version}')
    env = vectorizer_class([lambda: base_env for _ in range(n_envs)])
    eval_env = vectorizer_class([lambda: base_env for _ in range(n_eval_envs)]) if n_eval_envs is not None else None
    test_env = vectorizer_class([lambda: base_env for _ in range(n_test_envs)]) if n_test_envs is not None else None
    return env, eval_env, test_env
