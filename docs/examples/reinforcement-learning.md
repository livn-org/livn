# Reinforcement Learning

livn integrates with the [Gymnasium](https://gymnasium.farama.org/) standard interface:

```python
import gymnasium as gym

from livn.env import Env
from livn.system import predefined
from livn.integrations.gym import LivnGym

# standard RL environment
benchmark = gym.make("Pendulum-v1")

# livn system
env = Env(predefined("EI2")).init()

rl_env = LivnGym(
    benchmark,
    env,
    encoding=encoder,  # define callable to encode observations
    decoding=decoder,  # define callable to decode actions
)

# use rl_env like any other gymnasium environment
```
