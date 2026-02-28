from abc import abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from livn.types import Decoding, Encoding

if TYPE_CHECKING:
    from livn.types import Env


class GymnasiumEnv(gym.Env):
    """Gymnasium-compliant wrapper that turns a livn.Env into a gym interface"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: "Env",
        encoding: Encoding,
        decoding: Decoding,
    ):
        self.env = env
        self.encoding = encoding
        self.decoding = decoding

        self.action_space = encoding.input_space

        gym_step = (
            decoding.get_stage(GymStep) if hasattr(decoding, "get_stage") else None
        )
        base_space = (
            gym_step.gym_env.observation_space if gym_step is not None else None
        )

        extra_dim = (
            sum(s.obs_dim for s in decoding.stages if hasattr(s, "obs_dim"))
            if hasattr(decoding, "stages")
            else 0
        )
        if base_space is not None and extra_dim > 0:
            self.observation_space = gym.spaces.Box(
                low=np.concatenate(
                    [
                        base_space.low.flatten(),
                        np.zeros(extra_dim, dtype=np.float32),
                    ]
                ),
                high=np.concatenate(
                    [
                        base_space.high.flatten(),
                        np.full(extra_dim, np.inf, dtype=np.float32),
                    ]
                ),
                dtype=np.float32,
            )
        elif base_space is not None:
            self.observation_space = base_space
        else:
            self.observation_space = None

        self._is_async = (
            env is not None
            and hasattr(env, "submit_call")
            and hasattr(env, "receive_response")
        )

    def reset(self, **kwargs):
        # reset neural simulation state unless explicitly opted out
        if not kwargs.pop("without_clear", False):
            if self.env is not None and hasattr(self.env, "clear"):
                self.env.clear()
        if not hasattr(self.decoding, "reset"):
            return None

        # signal the worker to reset via state flag
        if self._is_async and hasattr(self.decoding, "state"):
            self.decoding.state["_needs_reset"] = True

        result = self.decoding.reset(**kwargs)
        if result is None:
            return None
        obs, info = result

        if hasattr(self.decoding, "state"):
            self.decoding.state["raw_gym_obs"] = np.asarray(obs, dtype=np.float32)

        if hasattr(self.decoding, "stages"):
            aug_stages = [
                s for s in self.decoding.stages if hasattr(s, "zero_features")
            ]
            if aug_stages:
                extra = np.concatenate([s.zero_features() for s in aug_stages])
                obs = np.concatenate([np.asarray(obs, dtype=np.float32), extra])
        return obs, info

    def step(self, action):
        pipe = self.decoding
        if hasattr(pipe, "state"):
            pipe.state["io_action"] = np.asarray(action, dtype=np.float32)
        # encoding can read raw_gym_obs from env.decoding.state["raw_gym_obs"]
        return self.env(self.decoding, action, self.encoding)

    def submit_step(self, action) -> dict:
        if self._is_async:
            task_id = self.env.submit_call(
                self.decoding,
                action,
                self.encoding,
            )
            return {"_async": True, "task_id": task_id}
        else:
            return {"_async": False, "action": action}

    def poll_step(self, handle: dict):
        if not handle.get("_async", False):
            return self.env(self.decoding, handle["action"], self.encoding)

        raw = self.env.probe_response(handle["task_id"])
        if raw is None:
            return None  # still in flight

        return raw

    def get_step(self, handle: dict) -> tuple:
        if handle.get("_async", False):
            result = self.env.receive_response()
        else:
            result = self.env(self.decoding, handle["action"], self.encoding)

        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    def render(self):
        pass

    def close(self):
        gym_step = (
            self.decoding.get_stage(GymStep)
            if hasattr(self.decoding, "get_stage")
            else None
        )
        if gym_step is not None:
            gym_step.close()


class GymStep:
    def __init__(self, gym_env):
        self.gym_env = gym_env

    def __hash__(self):
        spec = getattr(self.gym_env, "spec", None)
        return hash(spec.id if spec else id(self))

    def __eq__(self, other):
        if not isinstance(other, GymStep):
            return NotImplemented
        s1 = getattr(self.gym_env, "spec", None)
        s2 = getattr(other.gym_env, "spec", None)
        if s1 is None or s2 is None:
            return self is other
        return s1.id == s2.id

    def __getstate__(self):
        # pickle only the spec, not the full gym env state
        spec = getattr(self.gym_env, "spec", None)
        return {"spec_id": spec.id if spec else None}

    def __setstate__(self, d):
        spec_id = d.get("spec_id")
        self.gym_env = gym.make(spec_id) if spec_id else None

    def reset(self, **kwargs):
        obs, info = self.gym_env.reset(**kwargs)
        return np.asarray(obs, dtype=np.float32), info

    def _clamp_action(self, action):
        space = self.gym_env.action_space
        if hasattr(space, "n"):
            return int(np.round(float(np.squeeze(action))))
        return np.clip(
            np.asarray(action, dtype=space.dtype),
            space.low,
            space.high,
        )

    def __call__(self, env, action):
        pipe = getattr(env, "decoding", None)

        action = self._clamp_action(action)

        if hasattr(pipe, "context"):
            pipe.context["gym_action"] = action

        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        if hasattr(pipe, "state"):
            pipe.state["raw_gym_obs"] = obs

        if terminated or truncated:
            env.clear()
        return obs, reward, terminated, truncated, info

    def close(self):
        if callable(getattr(self.gym_env, "close", None)):
            self.gym_env.close()


class ObsAugmentation:
    """Base class for post-GymStep observation augmentation decoding stages"""

    obs_dim: int = 0  # override in subclasses

    def __init__(self, obs_dim: int | None = None):
        if obs_dim is not None:
            self.obs_dim = obs_dim

    def __hash__(self):
        return hash((type(self).__name__, tuple(sorted(self.__dict__.items()))))

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    @abstractmethod
    def _features(self, env, context: dict, state: dict) -> np.ndarray:  # (obs_dim,)
        """Returns feature vector to append."""

    def zero_features(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)

    def __call__(self, env, obs, reward, terminated, truncated, info):
        pipe = getattr(env, "decoding", None)
        ctx = getattr(pipe, "context", {})
        st = getattr(pipe, "state", {})
        features = self._features(env, ctx, st)
        obs_aug = np.concatenate([np.asarray(obs, dtype=np.float32), features])
        return obs_aug, reward, terminated, truncated, info
