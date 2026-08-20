# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn[diffrax] @ git+https://github.com/livn-org/livn.git",
# ]
# ///

import os

os.environ["LIVN_BACKEND"] = "diffrax"

import numpy as np

from livn.env import Env
from livn.models.glif import GLIF
from livn.stimulus import Stimulus
from livn.system import predefined

from optimization.fit import fit
from optimization.losses import voltage_mse

DURATION, DT = 40.0, 0.5

env = Env(
    predefined("EI"), model=GLIF.leaky_integrate_and_fire(mechanism="hard")
).init()
env.record_voltage(dt=DT)

n = env.num_cells
steps = int(round(DURATION / DT)) + 1
stimulus = Stimulus.from_current(np.full((steps, n), 0.05), dt=DT)  # nA, subthreshold

truth = np.linspace(-75.0, -68.0, n)
target = env.cells.set_params({"E_L": truth}).run(DURATION, stimulus, dt=DT).voltage


def loss(run, recorded):
    return voltage_mse(run.voltage, recorded)


theta, history = fit(
    env,
    target,
    loss,
    {"E_L": np.full(n, -71.5)},
    duration=DURATION,
    stimulus=stimulus,
    dt=DT,
    steps=60,
    learning_rate=0.5,
    callback=lambda step, params, value: (
        print(f" step {step:3d}  loss {value:.5f}") if step % 10 == 0 else None
    ),
)

recovered = np.asarray(theta["E_L"])
print(f"\nloss {history['loss'][0]:.4f} -> {history['loss'][-1]:.6f}")
print("true      ", np.round(truth, 2))
print("recovered ", np.round(recovered, 2))
print(f"max error  {np.max(np.abs(recovered - truth)):.3f} mV")
