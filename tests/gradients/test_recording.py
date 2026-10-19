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
    Chain,
    ComposedIO,
    ElectrodeRC,
    ExtracellularPotential,
    LightArtifact,
    Noise,
    PatternedLight,
    Sum,
)
from livn.io.base import _USES_JAX
from livn.optimization.data import Window
from livn.optimization.fit import fit

pytestmark = pytest.mark.skipif(not _USES_JAX, reason="needs a jax backend")

DT = 0.1
SPOTS = [[0, 0.0, 0.0, 40.0], [1, 300.0, 0.0, 40.0]]
ELECTRODES = [[0, 30.0, 0.0, 0.0], [1, 290.0, 20.0, 0.0], [2, 900.0, 0.0, 0.0]]
CELLS = np.asarray([[0, 50.0, 30.0, 5.0], [1, 500.0, -20.0, 5.0]])
B, T = 4, 300


def _io(weights=None):
    return ComposedIO(
        inputs=PatternedLight(SPOTS, irradiance=2.0),
        outputs=MEA(ELECTRODES, output_radius=2000.0),
        recording=Chain(
            Sum(
                extracellular=ExtracellularPotential(),
                artifact=LightArtifact(
                    jnp.full((3, 2), 5.0) if weights is None else weights,
                    taus=(0.5, 5.0),
                ),
            ),
            rc=ElectrodeRC(20.0),
            noise=Noise(1.0),
        ),
    )


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    currents = (rng.normal(size=(B, len(CELLS), T)) * 1e-3).astype(np.float32)
    commands = (rng.uniform(size=(B, T, len(SPOTS))) > 0.7).astype(np.float32)
    return currents, commands


def test_unset_fields_are_no_leaves_and_static_ones_no_leaves_either():
    conditions = Conditions(inputs=jnp.ones((3, 2)), dt=0.1, segment=2)

    leaves, treedef = jax.tree_util.tree_flatten(conditions)
    assert len(leaves) == 2
    again = jax.tree_util.tree_unflatten(treedef, leaves)
    assert again.dt == 0.1 and again.membrane_currents_uA is None


def test_a_batch_of_partially_set_conditions_vmaps_and_jits():
    io = _io()
    currents, commands = _batch()
    batch = Conditions(
        membrane_currents_uA=currents, inputs=commands, dt=DT, segment=jnp.arange(B)
    )
    key = jax.random.PRNGKey(0)

    @jax.jit
    def record(io, batch, keys):
        def one(conditions, k):
            return io.observe(conditions.replace(neuron_coordinates=CELLS), key=k)

        return jax.vmap(one)(batch, keys)

    keys = jax.random.split(key, B)
    got = record(io, batch, keys)

    assert got.shape == (B, 3, T)
    for b in range(B):
        one = io.observe(
            Conditions(
                membrane_currents_uA=currents[b],
                inputs=commands[b],
                dt=DT,
                neuron_coordinates=CELLS,
            ),
            key=keys[b],
        )
        np.testing.assert_allclose(got[b], one, rtol=2e-5, atol=1e-5)


def test_gradients_reach_every_block_of_the_composition_and_the_medium():
    io = _io()
    currents, commands = _batch()
    conditions = Conditions(
        membrane_currents_uA=currents[0],
        inputs=commands[0],
        dt=DT,
        neuron_coordinates=CELLS,
    )

    grad = jax.jit(jax.grad(lambda io: jnp.sum(io.observe(conditions) ** 2)))(io)

    for leaf in (
        grad.recording["artifact"].weights,
        grad.recording["rc"].log_r_seal,
        grad.medium.optical.width,
        grad.medium.electrical.sigma_s_per_m,
    ):
        assert np.all(np.isfinite(leaf)) and np.any(np.asarray(leaf) != 0.0)


class _Rig(eqx.Module):
    io: ComposedIO

    def __call__(self, window):
        def one(w):
            return self.io.observe(w.replace(neuron_coordinates=CELLS))

        return jax.vmap(one)(window)


def test_fit_recovers_the_artifact_weights_from_windows():
    true = jnp.asarray([[30.0, -10.0], [-15.0, 20.0], [0.0, 0.0]], jnp.float32)
    currents, commands = _batch(1)
    without_noise = _io(true).recording.replace("noise", Noise(0.0))
    target_io = _io(true)
    target_io.recording = without_noise
    window = Window(membrane_currents_uA=currents, inputs=commands, dt=DT)
    target = _Rig(target_io)(window)
    window = window.replace(context=target)

    start = _io(jnp.zeros((3, 2), jnp.float32))
    start.recording = start.recording.replace("noise", Noise(0.0))

    def loss(pred, window):
        return jnp.mean((pred - window.context) ** 2)

    result = fit(
        _Rig(start),
        loss,
        data=window,
        params=lambda rig: rig.io.recording["artifact"].weights,
        optimizer=optax.adam(1.0),
        steps=1500,
    )

    assert float(result.history["loss"][-1]) < 1e-3 * float(result.history["loss"][0])
    np.testing.assert_allclose(
        np.asarray(result.model.io.recording["artifact"].weights),
        np.asarray(true),
        atol=0.5,
    )


def test_gradients_reach_the_artifact_through_two_light_sources():
    io = ComposedIO(
        inputs=[
            PatternedLight([SPOTS[0]], irradiance=2.0),
            PatternedLight([SPOTS[1]], irradiance=1.0),
        ],
        outputs=MEA(ELECTRODES, output_radius=2000.0),
        recording=Chain(
            artifact=LightArtifact(jnp.ones((3, 2)), taus=(0.5, 5.0)),
            rc=ElectrodeRC(20.0),
        ),
    )
    _, commands = _batch()
    conditions = Conditions(inputs=commands[0], dt=DT)

    grad = jax.jit(jax.grad(lambda io: jnp.sum(io.observe(conditions) ** 2)))(io)

    for leaf in (grad.recording["artifact"].weights, grad.medium.optical.width):
        assert np.all(np.isfinite(leaf)) and np.any(np.asarray(leaf) != 0.0)
