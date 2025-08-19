#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "multiprocess",
#   "xxhash",
#   "stable-baselines3>=2.6.0",
#   "datasets==3.6.0",
#   "machinable",
# ]
# ///
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from datasets import load_dataset
from machinable.utils import save_file
from stable_baselines3 import SAC
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from livn.backend import backend
from livn.env import Env
from livn.integrations.gym import LivnGym, PulseEncoding
from livn.system import predefined
from livn.types import Array, Decoding, Encoding, Float, Int


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


def create_replay_buffer(system_name="S1"):
    """Create a replay buffer from the LIVN dataset."""
    benchmark_name = "Pendulum-v1"
    benchmark = gym.make(benchmark_name)

    env = Env(predefined(system_name))
    encoder = PulseEncoding(env)
    decoder = FirstSpikeDecoder(env)

    # Load dataset
    dataset = load_dataset("livn-org/livn", name=system_name, split="train")

    replay_buffer = ReplayBuffer(
        len(dataset),
        benchmark.observation_space,
        encoder.feature_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    obs, _ = benchmark.reset()
    for sample in dataset:
        action = np.array([a for a in sample["features"]])

        low, high = encoder.feature_space.low, encoder.feature_space.high
        scaled_action = 2.0 * ((action - low) / (high - low)) - 1.0

        # Decode the action from neural response
        latent_action = decoder(
            duration=sample["t_end"],
            it=np.array(sample["trial_it"][0]),
            tt=np.array(sample["trial_t"][0]),
            iv=None,
            vv=None,
        )

        next_obs, reward, terminated, truncated, _ = benchmark.step(latent_action)
        done = terminated or truncated

        replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=scaled_action,
            reward=reward,
            done=done,
            infos=[{"latent_action": latent_action}],
        )

        obs = next_obs
        if done:
            obs, _ = benchmark.reset()

    return replay_buffer


def main(replay=False):
    if replay:
        # Replay buffer mode - single environment training
        system_name = "S1"
        benchmark_name = "Pendulum-v1"
        benchmark = gym.make(benchmark_name)

        env = Env(predefined(system_name))
        encoder = PulseEncoding(env)
        decoder = FirstSpikeDecoder(env)

        env.init()
        env.apply_model_defaults(noise=False)
        env.record_spikes()

        wrapped = LivnGym(
            benchmark,
            env,
            encoding=encoder,
            decoding=decoder,
            verbose=False,
        )

        # Create replay buffer
        replay_buffer = create_replay_buffer(system_name)

        # Create model with replay buffer
        model = SAC(
            "MlpPolicy",
            wrapped,
            buffer_size=len(replay_buffer.observations),
            learning_starts=0,
            verbose=1,
        )
        model._logger = utils.configure_logger(1)
        model.replay_buffer = replay_buffer

        # Training with replay buffer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "rl", system_name
        )
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"{benchmark_name}_{timestamp}.jsonl")

        print(f"Training with replay buffer. Saving results to: {results_file}")

        total_timesteps = 1_000
        batch_size = 2048
        eval_freq = 5000

        training_rewards = []
        all_results = []

        for step in range(total_timesteps):
            # Batch gradient update
            model.train(gradient_steps=1, batch_size=batch_size)

            current_step = step + batch_size
            if current_step % eval_freq < batch_size or current_step >= total_timesteps:
                mean_reward, std_reward = evaluate_policy(
                    model, wrapped, n_eval_episodes=10, deterministic=False
                )
                training_rewards.append(mean_reward)

                result = {
                    "step": current_step,
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "timestamp": datetime.now().isoformat(),
                    "type": "training_eval",
                }
                all_results.append(result)
                save_file(results_file, result, mode="a+")

                print(
                    f"Step {step}/{total_timesteps}, Training reward: {mean_reward:.2f}"
                )

        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model,
            wrapped,
            n_eval_episodes=3,
            deterministic=True,
        )

        result = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "timestamp": datetime.now().isoformat(),
            "type": "final_evaluation",
        }
        all_results.append(result)
        save_file(results_file, result, mode="a+")

        print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"All results saved to: {results_file}")

        save_file([results_dir, f"{benchmark_name}_{timestamp}_all.json"], all_results)

    else:
        # multiprocessing mode
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

        a = self.action_min + (first_spike / duration) * self.action_range

        return np.array([a])

    @property
    def output_space(self) -> gym.Space:
        # the benchmark's action space
        return gym.spaces.Box(
            low=self.action_min, high=self.action_max, shape=(1,), dtype=np.float32
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RL training with or without replay buffer"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use replay buffer from dataset instead of online training",
    )

    args = parser.parse_args()

    if args.replay and backend() == "brian2":
        raise RuntimeError("Replay buffer mode requires NEURON backend, not brian2")
    elif not args.replay and backend() == "brian2":
        raise RuntimeError("Multiprocessing mode requires backend != 'brian2'")

    main(replay=args.replay)
