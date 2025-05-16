import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from livn.backend import backend
from livn.env import Env
from livn.integrations.gym import LivnGym
from livn.system import predefined
from livn.types import Array, Decoding, Encoding, Float, Int

assert backend() != "brian2"  # no multiprocessing support


def make_env():
    benchmark_name = "Pendulum-v1"
    benchmark = gym.make(benchmark_name)

    env = Env(predefined("S1")).init()
    encoder = FeatureEncoder(env, benchmark)
    decoder = FirstSpikeDecoder(env)

    wrapped = LivnGym(
        benchmark,
        env,
        encoding=encoder,
        decoding=decoder,
        verbose=False,
    )

    env.init()
    env.apply_model_defaults(noise=False)
    env.record_spikes()

    return wrapped


def main():
    num_cpu = max(1, os.cpu_count() // 2)
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    model = SAC(
        "MlpPolicy",
        env,
        train_freq=1,
        gradient_steps=2,
        verbose=1,
    )

    for iteration in range(3):
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=3, deterministic=True
        )

        print(iteration, mean_reward, std_reward)

        model.learn(1_000, log_interval=1)


class FeatureEncoder(Encoding):
    def __init__(self, env, benchmark, duration=1000):
        self.env = env
        self.benchmark = benchmark
        self.duration = duration

    def __call__(self, features):
        t_stim = int((features[0] + 2.0) / 4.0 * self.duration)
        inputs = np.zeros([self.duration, self.env.io.num_channels], dtype=np.float32)
        inputs[t_stim : t_stim + 20, 1:4] = 750

        return inputs, self.duration

    def _get_feature_space(self) -> gym.Space:
        # use original action space
        return self.benchmark.action_space


class FirstSpikeDecoder(Decoding):
    def __init__(self, env):
        self.env = env
        self.action_min = -2.0
        self.action_max = 2.0
        self.action_range = self.action_max - self.action_min

    def __call__(
        self,
        duration: float,
        it: Int[Array, "n_spiking_neuron_ids"],
        tt: Float[Array, "n_spiking_neuron_times"],
        iv: Int[Array, "n_voltage_neuron_ids"],
        vv: Float[Array, "neuron_ids voltages"],
    ):
        if len(it) == 0 or len(tt) == 0:
            return np.array([0.0])

        cit, ct = self.env.channel_recording(it, tt)

        per_channel_firing_rate = {
            key: np.nan_to_num(
                np.mean(np.unique(val, return_counts=True)[1] / (duration / 100))
            )
            for key, val in cit.items()
        }

        # spatial
        max_channel = max(per_channel_firing_rate, key=per_channel_firing_rate.get)
        first_spike = ct[max_channel][0]

        bin_size = self.action_range / (len(cit) - 1)
        a = self.action_min + (first_spike / duration) * self.action_range

        return np.array([a])

    @property
    def output_space(self) -> gym.Space:
        # the benchmark's action space
        return self.env.env.action_space


if __name__ == "__main__":
    main()
