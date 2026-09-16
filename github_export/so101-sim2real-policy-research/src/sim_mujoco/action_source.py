from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ActionSourceConfig:
    type: str = "random"
    constant: list[float] | None = None
    scripted_callable: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ActionSourceConfig":
        raw = raw or {}
        return cls(
            type=str(raw.get("type", "random")),
            constant=raw.get("constant"),
            scripted_callable=raw.get("scripted_callable"),
        )


class ActionSource:
    def __init__(self, env: Any):
        self.env = env

    def reset(self, episode_index: int, observation: Any, info: dict[str, Any] | None) -> None:
        _ = episode_index, observation, info

    def next_action(
        self,
        observation: Any,
        info: dict[str, Any] | None,
        step_index: int,
        episode_index: int,
    ) -> np.ndarray:
        raise NotImplementedError


class RandomActionSource(ActionSource):
    def next_action(
        self,
        observation: Any,
        info: dict[str, Any] | None,
        step_index: int,
        episode_index: int,
    ) -> np.ndarray:
        _ = observation, info, step_index, episode_index
        return _to_action_array(self.env.action_space.sample())


class ConstantActionSource(ActionSource):
    def __init__(self, env: Any, constant: list[float]):
        super().__init__(env)
        arr = np.asarray(constant, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"`constant` action must be 1D, got shape={arr.shape}.")
        self._constant = arr

    def next_action(
        self,
        observation: Any,
        info: dict[str, Any] | None,
        step_index: int,
        episode_index: int,
    ) -> np.ndarray:
        _ = observation, info, step_index, episode_index
        return self._constant.copy()


class ScriptedCallableActionSource(ActionSource):
    def __init__(self, env: Any, callable_path: str):
        super().__init__(env)
        self._callback = _load_callable(callable_path)

    def next_action(
        self,
        observation: Any,
        info: dict[str, Any] | None,
        step_index: int,
        episode_index: int,
    ) -> np.ndarray:
        action = self._callback(
            observation=observation,
            info=info,
            step_index=step_index,
            episode_index=episode_index,
            env=self.env,
        )
        return _to_action_array(action)


def create_action_source(config: ActionSourceConfig, env: Any) -> ActionSource:
    source_type = config.type.lower()
    if source_type == "random":
        return RandomActionSource(env)
    if source_type == "constant":
        if config.constant is None:
            raise ValueError("`action_source.constant` must be set when `type=constant`.")
        return ConstantActionSource(env, config.constant)
    if source_type == "scripted":
        if not config.scripted_callable:
            raise ValueError("`action_source.scripted_callable` must be set when `type=scripted`.")
        return ScriptedCallableActionSource(env, config.scripted_callable)
    raise ValueError(f"Unsupported action source type: {config.type}")


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(
            f"Invalid callable path '{path}'. Expected format 'module.submodule:function_name'."
        )
    module_name, fn_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Could not load callable '{fn_name}' from module '{module_name}'.")
    return fn


def _to_action_array(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Action must be a 1D array, got shape={arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Action contains NaN or Inf values.")
    return arr

