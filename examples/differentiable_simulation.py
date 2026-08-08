# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn[diffrax] @ git+https://github.com/livn-org/livn.git",
# ]
# ///

import os

os.environ["LIVN_BACKEND"] = "diffrax"

import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

from livn.env import Env
from livn.system import predefined


@eqx.filter_jit
def systempass(inputs, env, t_end, targets, key):
    # pass through IO and system
    stimulus = env.cell_stimulus(inputs)
    mask, _, gids, v, *_ = env.run(t_end, stimulus, unroll="mask")

    return -jnp.mean(v)  # dummy loss: maximize action potentials


@eqx.filter_jit
def make_step(
    env,
    inputs,
    t_end,
    grad_loss,
    optim,
    targets,
    opt_state,
    key,
):
    loss, grads = grad_loss(inputs, env, t_end, targets, key)
    updates, opt_state = optim.update(grads, opt_state)
    new_inputs = eqx.apply_updates(inputs, updates)
    return loss, opt_state, new_inputs


env = Env(predefined("S1")).init()

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()


step_key = env.run_key

t_end = 30
inputs = jnp.zeros([t_end, 16])

optim = optax.adam(1)
opt_state = optim.init(eqx.filter(inputs, eqx.is_inexact_array))
grad_loss = eqx.filter_value_and_grad(systempass)
targets = []


print("Initial input:", inputs.mean())

for iteration in range(5):
    start = time.time()
    step = jnp.asarray(iteration)
    step_key = jr.fold_in(step_key, step)

    loss, opt_state, inputs = make_step(
        env,
        inputs,
        t_end,
        grad_loss,
        optim,
        targets,
        opt_state,
        step_key,
    )

    end = time.time()

    print(
        f"[{end - start} s]: {iteration}, loss: {loss:2f}, updated input: {inputs.mean():2f}"
    )

import jax  # noqa: E402
import numpy as np  # noqa: E402

from livn.models.glif import GLIF  # noqa: E402

cells = Env(4, model=GLIF(level=3)).init()

duration, step = 60.0, 0.1
current = jnp.full((int(duration / step) + 1, 4), 0.3)  # nA

target = 12.0


def first_spike_loss(params):
    _, times, *_ = cells.cells.set_params(params).module.run(
        input_current=current,
        t0=0.0,
        t1=duration,
        dt=step,
        unroll="padded",
    )
    times = times.reshape(4, -1)
    first = jnp.min(jnp.where(jnp.isfinite(times), times, duration), axis=1)
    return jnp.mean((first - target) ** 2)


theta = {"V_threshold_base": cells.cells.get_params()["V_threshold_base"]}
optimizer = optax.adam(5.0)
state = optimizer.init(theta)

for iteration in range(10):
    value, gradients = eqx.filter_value_and_grad(first_spike_loss)(theta)
    updates, state = optimizer.update(gradients, state)
    theta = eqx.apply_updates(theta, updates)
    print(
        f"{iteration}: loss {float(value):.3f}, "
        f"d(loss)/d(threshold) {np.asarray(gradients['V_threshold_base'])[0]:+.4f}, "
        f"threshold {float(theta['V_threshold_base'][0]):.2f} mV"
    )


assert float(jax.grad(first_spike_loss)(theta)["V_threshold_base"][0]) != 0.0
