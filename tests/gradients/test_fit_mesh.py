import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent(
    """
    import equinox as eqx, jax, jax.numpy as jnp, numpy as np, optax

    from livn.optimization.fit import fit
    from livn.optimization.train import data_mesh

    assert jax.device_count() == 4, jax.devices()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5], np.float32)).astype(np.float32)
    batch = (X, y)

    class Linear(eqx.Module):
        w: jax.Array

        def __call__(self, batch):
            return batch[0] @ self.w

    def loss(pred, batch):
        return jnp.mean((pred - batch[1]) ** 2)

    def run(mesh, optimizer):
        return fit(
            Linear(jnp.zeros(3)),
            loss,
            data=batch,
            optimizer=optimizer,
            steps=6,
            mesh=mesh,
        )

    for optimizer in (optax.sgd(0.1), optax.adam(0.1)):
        single = run(None, optimizer)
        for mesh in (data_mesh(), jax.make_mesh((4,), ("batch",))):  # Explicit gets fixed up
            sharded = run(mesh, optimizer)
            np.testing.assert_allclose(
                np.asarray(sharded.theta.w), np.asarray(single.theta.w), atol=1e-6
            )
            np.testing.assert_allclose(
                sharded.history["loss"], single.history["loss"], rtol=1e-5, atol=1e-6
            )

    # a stream of per-step slices is placed on the mesh too
    seen = []

    def stream(step):
        return (X * (1 + 0.1 * step), y)

    class Probe:
        def on_fit_start(self, state): pass
        def on_fit_end(self, state): pass
        def on_step_end(self, step, state, metrics):
            seen.append(len(state.raw.w.sharding.device_set))

    sharded = fit(
        Linear(jnp.zeros(3)), loss, data=stream,
        optimizer=optax.sgd(0.05), steps=3, mesh=data_mesh(), callbacks=[Probe()],
    )
    single = fit(
        Linear(jnp.zeros(3)), loss, data=stream,
        optimizer=optax.sgd(0.05), steps=3,
    )
    np.testing.assert_allclose(
        np.asarray(sharded.theta.w), np.asarray(single.theta.w), atol=1e-6
    )
    assert seen == [4, 4, 4], "model and optimizer state are replicated over the mesh"
    print("OK")
    """
)


def test_a_sharded_fit_equals_the_single_device_fit(tmp_path):
    script = tmp_path / "mesh_check.py"
    script.write_text(SCRIPT)
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    }
    done = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert done.returncode == 0 and "OK" in done.stdout, done.stdout + done.stderr
