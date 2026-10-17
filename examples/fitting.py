# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn[diffrax]",
# ]
# ///

import functools
import os

os.environ["LIVN_BACKEND"] = "diffrax"

import numpy as np
import optax

from livn.env import Env
from livn.models.glif import GLIF
from livn.optimization.fit import fit
from livn.optimization.losses import voltage_mse
from livn.optimization.train import Every
from livn.stimulus import Stimulus
from livn.system import Monolayer

DURATION, DT = 40.0, 0.5

env = Env(
    Monolayer(total_cells=40),
    model=GLIF.leaky_integrate_and_fire(mechanism="hard"),
).init()
env = env.record_voltage(dt=DT)

n = env.num_cells
steps = round(DURATION / DT) + 1
stimulus = Stimulus.from_current(np.full((steps, n), 0.05), dt=DT)  # nA, subthreshold

truth = np.linspace(-75.0, -68.0, n)
target = env.cells.set_params({"E_L": truth}).run(DURATION, stimulus, dt=DT).voltage

start = env.cells.set_params({"E_L": np.full(n, -71.5)})
result = fit(
    functools.partial(start, round(DURATION), stimulus, dt=DT),
    lambda run, _: voltage_mse(run.voltage, target),
    params={"E_L": lambda m: m.func.module.E_L},
    optimizer=optax.adam(0.5),
    steps=60,
    callbacks=[
        Every(
            10,
            lambda step, state, metrics: print(
                f" step {step:3d}  loss {metrics['loss']:.5f}"
            ),
        )
    ],
)

recovered = np.asarray(result.theta["E_L"])
print(f"\nloss {result.history['loss'][0]:.4f} -> {result.history['loss'][-1]:.6f}")
print("true      ", np.round(truth, 2))
print("recovered ", np.round(recovered, 2))
print(f"max error  {np.max(np.abs(recovered - truth)):.3f} mV")
