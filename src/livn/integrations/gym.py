from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, WrapperActType

from livn.env import Pass
from livn.types import Decoding, Encoding

if TYPE_CHECKING:
    from livn.types import Env


class LivnGym(gym.ActionWrapper[ObsType, WrapperActType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        livn_env: "Env",
        encoding: Encoding,
        decoding: Decoding,
        **run_kwargs,
    ):
        gym.Wrapper.__init__(self, env)

        self.envpass = Pass(livn_env, encoding, decoding, run_kwargs)
        self.action_space = encoding.feature_space

    def action(self, action: WrapperActType) -> ActType:
        return self.envpass(action)


class PulseEncoding(Encoding):
    def __init__(self, env, duration=1000):
        self.env = env
        self.duration = duration

    def __call__(self, features):
        features = {
            "channel": int(np.round(features[0])),
            "time": features[1],
            "amplitude": features[2],
        }

        t_stim = int(features["time"])

        inputs = np.zeros([self.duration, self.env.io.num_channels], dtype=np.float32)
        inputs[t_stim : t_stim + 20, features["channel"]] = features["amplitude"]

        return inputs, self.duration

    def _get_feature_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([0, 0, 250]),
            high=np.array([self.env.io.num_channels - 1, self.duration - 100, 750]),
            dtype=np.float32,
        )
