"""Gradients through the photoelectric artifact."""

import numpy as np
import pytest

pytest.importorskip("optax")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from livn.conditions import Conditions
from livn.io import (
    MEA,
    Bandpass,
    Chain,
    ComposedIO,
    ElectrodeRC,
    ExtracellularPotential,
    LightArtifact,
    PatternedLight,
    Sum,
)
from livn.io.base import _USES_JAX

pytestmark = pytest.mark.skipif(not _USES_JAX, reason="needs a jax backend")

DT = 0.1
SPOTS = [[0, 0.0, 0.0, 40.0], [1, 300.0, 0.0, 40.0]]
ELECTRODES = [[0, 30.0, 0.0, 0.0], [1, 290.0, 20.0, 0.0], [2, 900.0, 0.0, 0.0]]


def _finite_and_nonzero(x):
    x = np.asarray(x)
    assert np.all(np.isfinite(x)) and np.any(x != 0.0)


def _commands(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    commands = np.zeros((n, len(SPOTS)), dtype=np.float32)
    for spot in range(len(SPOTS)):
        t = 0
        while t < n:
            t += int(rng.integers(20, 200))
            width = int(rng.integers(2, 150))
            commands[t : t + width, spot] = 1.0
            t += width
    return commands


def test_the_artifact_is_differentiable_through_a_composition():
    io = ComposedIO(
        inputs=PatternedLight(SPOTS, irradiance=2.0),
        outputs=MEA(ELECTRODES, output_radius=2000.0),
        recording=Chain(
            Sum(
                extracellular=ExtracellularPotential(),
                artifact=LightArtifact(
                    jnp.ones((3, 2)), taus=(0.5, 5.0), saturation=4.0
                ),
            ),
            rc=ElectrodeRC(20.0),
            band=Bandpass(300.0, 3000.0),
        ),
    )
    cells = np.asarray([[0, 50.0, 30.0, 5.0], [1, 500.0, -20.0, 5.0]])
    currents = jnp.asarray(np.random.default_rng(1).normal(size=(2, 500)) * 1e-3)
    conditions = Conditions(
        membrane_currents_uA=currents,
        neuron_coordinates=cells,
        dt=DT,
        inputs=_commands(500),
    )

    def energy(io):
        return jnp.sum(io.observe(conditions) ** 2)

    grad = jax.jit(jax.grad(energy))(io)
    _finite_and_nonzero(grad.recording["artifact"].weights)
    _finite_and_nonzero(grad.recording["artifact"].log_saturation)
    _finite_and_nonzero(grad.recording["rc"].log_r_seal)
    # the electrodes sit on the spots' edges, so the light on them moves with the blur
    _finite_and_nonzero(grad.medium.optical.width)


def test_the_empty_dish_is_differentiable():
    io = ComposedIO(
        inputs=PatternedLight(SPOTS, irradiance=2.0),
        outputs=MEA(ELECTRODES),
        recording=Chain(
            artifact=LightArtifact(jnp.ones((3, 2)), taus=(0.5, 5.0)),
            rc=ElectrodeRC(20.0),
        ),
    )
    conditions = Conditions(dt=DT, inputs=_commands(500))

    def energy(io):
        return jnp.sum(io.observe(conditions) ** 2)

    grad = jax.jit(jax.grad(energy))(io)
    _finite_and_nonzero(grad.recording["artifact"].weights)
    _finite_and_nonzero(grad.medium.optical.width)


def test_a_known_kernel_is_recovered_from_a_synthetic_artifact():
    taus = (0.2, 2.0, 20.0)
    true_weights = jnp.asarray(
        [[40.0, -30.0, 5.0], [-20.0, 25.0, 2.0], [0.0, 0.0, 0.0]], dtype=jnp.float32
    )

    def dish(weights):
        return ComposedIO(
            inputs=PatternedLight(SPOTS, irradiance=2.0),
            outputs=MEA(ELECTRODES),
            recording=Chain(artifact=LightArtifact(weights, taus=taus)),
        )

    conditions = Conditions(dt=DT, inputs=_commands())
    target = dish(true_weights).observe(conditions)

    io = dish(jnp.zeros((3, len(taus)), jnp.float32))

    def loss(weights):
        fitted = eqx.tree_at(lambda io: io.recording["artifact"].weights, io, weights)
        return jnp.mean((fitted.observe(conditions) - target) ** 2)

    weights = io.recording["artifact"].weights
    optimizer = optax.adam(0.5)
    state = optimizer.init(weights)

    @jax.jit
    def step(weights, state):
        value, grad = jax.value_and_grad(loss)(weights)
        updates, state = optimizer.update(grad, state, weights)
        return optax.apply_updates(weights, updates), state, value

    first = None
    for _ in range(1500):
        weights, state, value = step(weights, state)
        first = value if first is None else first

    assert float(value) < 1e-4 * float(first)
    np.testing.assert_allclose(np.asarray(weights), np.asarray(true_weights), atol=0.5)
