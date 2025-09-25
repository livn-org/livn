# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
# ]
# ///
from livn.env import Env
from livn.system import predefined

env = Env(predefined("S1")).init()

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()

it, t, iv, v, *_ = env.run(100)

print("Initial voltages: ", v[:, 0])
