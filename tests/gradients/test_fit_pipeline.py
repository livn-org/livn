import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("grain")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from livn.optimization.data import Standardize, windows
from livn.optimization.fit import fit
from livn.optimization.train import EqxCheckpointer, data_mesh
from livn.run import Run, Series

FS, C, T = 10_000.0, 3, 8_000


def _pipeline(run_recording):
    values = np.random.default_rng(0).normal(size=(C, T)).astype(np.float32)
    run = Run(
        potential=Series(ids=np.arange(C), values=values, dt=1e3 / FS),
        duration=T / FS * 1e3,
    )
    return windows(
        run_recording(run),
        context=2.0,
        prediction=0.5,
        normalize=Standardize({"median": 0.0, "scale": 1.0}),
        batch_size=4,
        per_segment=8,
        threads=2,
        seed=3,
    )


class _Model(eqx.Module):
    w: jax.Array

    def __call__(self, batch):
        # predict the first prediction sample's channel 0 from the context's mean over time
        return batch.context.mean(axis=1) @ self.w


def _model():
    return _Model(jnp.zeros((C,), jnp.float32))


def _loss(pred, batch):
    return jnp.mean((pred - batch.prediction[:, 0, 0]) ** 2)


def _run(run_recording, steps, **kw):
    return fit(
        _model(),
        _loss,
        data=_pipeline(run_recording),
        optimizer=optax.adam(0.05),
        steps=steps,
        **kw,
    )


def _w(result):
    return np.asarray(result.theta.w)


def test_fit_trains_on_a_pipeline_and_resumes_from_the_checkpointed_step(
    tmp_path, run_recording
):
    whole = _run(run_recording, 8)
    assert float(whole.history["loss"][-1]) != float(whole.history["loss"][0])

    _run(run_recording, 5, checkpoint=EqxCheckpointer(tmp_path, every=2))
    resumed = _run(run_recording, 8, checkpoint=EqxCheckpointer(tmp_path, every=2))
    assert resumed.state.step == 8
    np.testing.assert_allclose(_w(resumed), _w(whole), atol=1e-6)


def test_fit_with_a_mesh_places_the_process_slices_and_matches_unsharded(
    tmp_path, run_recording
):
    plain = _run(run_recording, 6)
    meshed = _run(run_recording, 6, mesh=data_mesh())
    np.testing.assert_allclose(_w(meshed), _w(plain), atol=1e-6)

    _run(
        run_recording,
        4,
        mesh=data_mesh(),
        checkpoint=EqxCheckpointer(tmp_path, every=2),
    )
    resumed = _run(
        run_recording,
        6,
        mesh=data_mesh(),
        checkpoint=EqxCheckpointer(tmp_path, every=2),
    )
    np.testing.assert_allclose(_w(resumed), _w(plain), atol=1e-6)
