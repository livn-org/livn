import json
import os
from datetime import datetime
from typing import Any

import gymnasium
import numpy as np
from datasets import load_dataset
from machinable.utils import save_file
from stable_baselines3 import SAC
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

from livn.backend import backend
from livn.env import Env
from livn.integrations.gym import LivnGym, PulseEncoding
from livn.system import predefined
from livn.types import Array, Decoding, Float, Int


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
    def output_space(self) -> gymnasium.Space:
        return benchmark.action_space


assert backend() == "neuron"


system_name = "S1"

env = Env(predefined(system_name))

encoder = PulseEncoding(env)
decoder = FirstSpikeDecoder(env)

benchmark_name = "Pendulum-v1"
benchmark = gymnasium.make(benchmark_name)

# generate replay buffer from dataset
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

    # encoding + cell_stim + run coming from dataset

    # decode
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

model = SAC(
    "MlpPolicy",
    wrapped,
    buffer_size=len(replay_buffer.observations),
    learning_starts=0,
    verbose=1,
)
model._logger = utils.configure_logger(1)
model.replay_buffer = replay_buffer


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "rl", system_name
)
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"{benchmark_name}_{timestamp}.jsonl")

print(f"Saving results to: {results_file}")

total_timesteps = 1_000
batch_size = 2048
eval_freq = 5000

training_rewards = []
all_results = []

for step in range(total_timesteps):
    # batch gradient update
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

        print(f"Step {step}/{total_timesteps}, Training reward: {mean_reward:.2f}")


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
