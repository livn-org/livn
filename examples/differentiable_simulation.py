# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
# ]
# ///

import os

os.environ["LIVN_BACKEND"] = "diffrax"

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from livn.env import Env
from livn.system import predefined


@eqx.filter_jit
def systempass(inputs, env, t_end, targets, key):
    # pass through IO and system
    stimulus = env.cell_stimulus(inputs)
    mask, _, gids, v = env.run(t_end, stimulus, unroll="mask")

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
