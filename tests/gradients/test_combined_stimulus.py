import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from livn.io.base import _USES_JAX
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.stimulus import Stimulus
from testing import livn_test_env

pytestmark = pytest.mark.skipif(not _USES_JAX, reason="needs a jax backend")

DT, DURATION = 0.1, 10.0


def test_a_combined_run_differentiates_in_both_inputs():
    env = livn_test_env(
        model=ReducedCalciumSomaDendrite(input_mode=("current", "irradiance"))
    )
    env.selection(2)
    env = env.init().record_voltage()
    gids = np.asarray(env.simulated_gids()).astype(int)

    pulse = np.zeros((round(DURATION / DT), len(gids)), dtype=np.float32)
    pulse[20:80] = 1.0

    def depolarization(current, light):
        stimulus = Stimulus.from_current(
            current * pulse, dt=DT, gids=gids
        ) + Stimulus.from_irradiance(light * pulse, dt=DT, gids=gids)
        run = env.run(DURATION, stimulus=stimulus, dt=DT, dt_solver=0.025)
        return jnp.mean(run.voltage)

    by_current, by_light = jax.grad(depolarization, argnums=(0, 1))(
        jnp.float32(1e-4), jnp.float32(1.0)
    )

    for name, grad in (("current", by_current), ("light", by_light)):
        assert np.isfinite(grad), f"the gradient in the {name} is {grad}"
        # both depolarize
        assert grad > 0.0, f"more {name} lowered the membrane ({grad})"


def test_a_composite_is_a_pytree_that_jits_and_differentiates_in_its_parts():
    from livn.stimulus import ComposedStimulus

    gids = np.arange(3)
    a = Stimulus.from_current(jnp.ones((10, 3)), dt=DT, gids=gids)
    b = Stimulus.from_current(jnp.ones((10, 2)), dt=DT, gids=gids[1:])

    leaves, treedef = jax.tree_util.tree_flatten(a + b)
    assert len(leaves) == 2
    assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), ComposedStimulus)

    @jax.jit
    def total(scale, composite):
        scaled = jax.tree_util.tree_map(lambda x: scale * x, composite)
        return jnp.sum(scaled.groups()["current"].array ** 2)

    value, grad = jax.value_and_grad(total)(jnp.float32(2.0), a + b)
    # cells 1 and 2 get 1 + 1, cell 0 gets 1: 10 * (4 + 16 + 16) at scale 2
    np.testing.assert_allclose(value, 10 * (4 + 16 + 16), rtol=1e-6)
    np.testing.assert_allclose(grad, 2 * value / 2.0, rtol=1e-6)
