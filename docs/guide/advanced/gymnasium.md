# Gymnasium Integration

The `GymnasiumEnv` wrapper turns a livn `Env` into a standard [Gymnasium](https://gymnasium.farama.org/) interface, allowing the neural culture to serve as a controller for any Gymnasium task. The culture receives observations encoded as electrical stimuli and responds with spiking activity that is decoded back into actions for the task environment.

## Overview

The integration sits between two worlds. On one side is the Gymnasium task (e.g. CartPole, Acrobot) that defines the RL problem; on the other side is the livn neural culture producing "actions":

```
Agent action → Encoding → livn culture → Decoding → Gym env → obs, reward
        ↑                                                          |
        └──────────────────────────────────────────────────────────┘
```

Three components wire this together:

| Component | Role |
|---|---|
| `GymnasiumEnv` | Gym-compliant wrapper; owns the env, encoding, and decoding |
| `GymStep` | Decoding pipeline stage that forwards decoded actions to the Gymnasium task |
| `ObsAugmentation` | Base class for stages that append extra features to observations |

## `GymnasiumEnv`

`GymnasiumEnv` implements `gym.Env` so it can be used with any RL framework that expects a Gymnasium interface (e.g. Stable-Baselines3, CleanRL, or custom JAX training loops).

```python
from livn.env.gymnasium import GymnasiumEnv

gym_env = GymnasiumEnv(
    env=livn_env,          # a livn Env (including DistributedEnv)
    encoding=my_encoding,  # Encoding: agent action -> IO stimulus
    decoding=my_decoding,  # Decoding (typically a Pipe containing GymStep)
)
```

### Spaces

- `action_space` is taken from `encoding.input_space`, i.e. the space of IO stimulation vectors the RL agent can produce
- `observation_space` is derived from the Gymnasium task's observation space (via `GymStep.gym_env.observation_space`) and automatically extended when the decoding pipeline contains `ObsAugmentation` stages

### `reset`

Delegates to the decoding pipeline's `reset()` method (which in turn resets the Gymnasium task via `GymStep`). If `ObsAugmentation` stages are present, their `zero_features()` are appended to the initial observation.

```python
obs, info = gym_env.reset()
```

### `step`

Stores the agent's raw action in `pipe.state["io_action"]`, then runs the full simulation loop:

```python
obs, reward, terminated, truncated, info = gym_env.step(action)
```

Under the hood this calls `env(decoding, action, encoding)` which:

1. Encodes the action into a stimuli
2. Runs the neural simulation
3. Decodes the response into a task action
4. Steps the Gymnasium task and returns the result

## `GymStep`

`GymStep` is a decoding pipeline stage that bridges the neural simulation output to a Gymnasium environment. Place it inside a [`Pipe`](/guide/advanced/decoding-pipelines) after your activity-decoding stage:

```python
from livn.decoding import Pipe
from livn.env.gymnasium import GymStep

decoding = Pipe(
    duration=100,
    stages=[
        my_spike_decoder,   # decodes spikes into gym action
        GymStep(gym.make("CartPole-v1")),  # forwards action to CartPole
    ],
)
```

Specifically, `GymStep` does the following on each call:

1. Clamps the decoded action to the task's action space. For discrete spaces (`gym.spaces.Discrete`), the action is rounded to the nearest integer; for continuous spaces (`gym.spaces.Box`), it's clipped to `[low, high]`.
2. Publishes the clamped action to `context["gym_action"]` for downstream stages
3. Steps the Gymnasium environment
4. Stores the resulting observation in `state["raw_gym_obs"]`
5. Calls `env.clear()` on episode termination to reset neural state



## `ObsAugmentation`

`ObsAugmentation` is a base class for decoding stages that append extra features to the Gymnasium observation. This allows the RL agent to observe additional information beyond the raw task state, for example the culture's spike counts or the stimulus that was applied.

```python
from livn.env.gymnasium import ObsAugmentation

class MySpikeFeatures(ObsAugmentation):
    obs_dim: int = 16  # number of features to append

    def _features(self, env, context: dict, state: dict) -> np.ndarray:
        counts = context.get("spike_counts", {})
        v = np.zeros(self.obs_dim, dtype=np.float32)
        for ch, cnt in counts.items():
            if 0 <= int(ch) < self.obs_dim:
                v[int(ch)] = float(cnt)
        return v
```

- Set `obs_dim` to the length of the feature vector your stage appends
- Implement `_features(env, context, state)` to return the feature array
- `zero_features()` returns zeros of length `obs_dim`; used during `reset()` before any simulation has run
- `GymnasiumEnv` automatically extends `observation_space` to account for all augmentation stages

Place augmentation stages after `GymStep` in the pipeline so they can append to the Gymnasium observation:

```python
decoding = Pipe(
    duration=100,
    stages=[
        my_spike_decoder,
        GymStep(gym.make("Acrobot-v1")),
        MySpikeFeatures(obs_dim=16),  # appended to obs from prior stage
    ],
)
```

## Shared state keys

The Gymnasium integration uses the [decoding pipeline](/guide/advanced/decoding-pipelines) state mechanisms to pass data between stages. The following keys are used by convention:

| Key | Scope | Writer | Description |
|---|---|---|---|
| `raw_gym_obs` | `state` | `GymStep` | Raw Gymnasium observation from the last step |
| `gym_action` | `context` | `GymStep` | Clamped action applied to the Gymnasium env |
| `io_action` | `state` | `GymnasiumEnv.step` | Raw IO from the RL agent |


## Async stepping

When backed by a [`DistributedEnv`](/guide/advanced/distributed), `GymnasiumEnv` supports non-blocking step submission for overlapping multiple episodes across MPI workers:

```python
# Submit without blocking
handle = gym_env.submit_step(action)

# Poll (returns None while in-flight)
result = gym_env.poll_step(handle)

# Or block until ready
obs, reward, terminated, truncated, info = gym_env.get_step(handle)
```

This is detected automatically: if the underlying `env` has `submit_call` and `receive_response` methods, `submit_step` dispatches asynchronously. Otherwise it falls back to a synchronous call.

## Full example

The following example wires a CartPole task to a livn neural culture using a spike-count decoding pipeline with observation augmentation:

```python
import gymnasium as gym
import numpy as np
from livn import make
from livn.decoding import Pipe
from livn.env.gymnasium import GymnasiumEnv, GymStep, ObsAugmentation


# 1. A custom spike decoder (converts spike counts to a force value)
from livn.types import Decoding
from livn.utils import P

class ForceDecoder(Decoding):
    duration: int = 100
    scale: float = 5.0

    def setup(self, env):
        env.record_spikes()

    def __call__(self, env, it, tt, iv, vv, im, mp):
        cit, ct = env.channel_recording(it, tt)
        cit, ct = P.gather(cit, ct, comm=env.comm)
        if P.is_root(comm=env.comm):
            cit, ct = P.merge(cit, ct)
            left = sum(len(t) for ch, t in cit.items() if ch in {1, 3})
            right = sum(len(t) for ch, t in cit.items() if ch in {0, 2})
            pipe = getattr(env, "decoding", None)
            if hasattr(pipe, "context"):
                pipe.context["spike_counts"] = {
                    int(ch): len(t) for ch, t in cit.items()
                }
            return float((right - left) * self.scale)


# 2. An observation augmentation (appends spike counts)
class SpikeObs(ObsAugmentation):
    obs_dim: int = 8

    def _features(self, env, context, state):
        counts = context.get("spike_counts", {})
        v = np.zeros(self.obs_dim, dtype=np.float32)
        for ch, cnt in counts.items():
            if 0 <= int(ch) < self.obs_dim:
                v[int(ch)] = float(cnt)
        return v


# 3. Build the decoding pipeline
cart_env = gym.make("CartPole-v1")

decoding = Pipe(
    duration=100,
    stages=[
        ForceDecoder(duration=100),
        GymStep(cart_env),
        SpikeObs(obs_dim=8),
    ],
)

# 4. Create the livn environment and wrap it
livn_env = make("EI1")
livn_env.record_spikes()

gymnasium_env = GymnasiumEnv(
    env=livn_env,
    encoding=my_encoding,
    decoding=decoding,
)

# 5. Standard Gymnasium loop
obs, info = gymnasium_env.reset()
for _ in range(500):
    action = gymnasium_env.action_space.sample()
    obs, reward, terminated, truncated, info = gymnasium_env.step(action)
    if terminated or truncated:
        obs, info = gymnasium_env.reset()

gymnasium_env.close()
```

