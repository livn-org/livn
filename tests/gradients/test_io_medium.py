import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from livn.io import (
    FIR,
    MEA,
    Bandpass,
    ComposedIO,
    DispersivePointSource,
    ElectrodeRC,
    KubelkaMunk,
    LightArray,
    Medium,
    Noise,
    PointSource,
)
from livn.io.mea import _USES_JAX

pytestmark = pytest.mark.skipif(not _USES_JAX, reason="needs a jax backend")

DT = 0.1


def _scene(seed=0, n_cells=12, n_steps=64):
    rng = np.random.default_rng(seed)
    electrodes = np.array([[0, 0.0, 0.0, 5.0], [1, 120.0, 0.0, 5.0]])
    cells = np.column_stack(
        [
            np.arange(n_cells),
            rng.uniform(-150, 250, (n_cells, 2)),
            np.full(n_cells, 15.0),
        ]
    )
    currents = jnp.asarray(rng.normal(size=(n_cells, n_steps)), dtype=jnp.float32)
    return electrodes, cells, currents


def _finite_and_nonzero(x):
    x = np.asarray(x)
    assert np.all(np.isfinite(x)) and np.any(x != 0.0)


def test_the_potential_differentiates_in_sigma():
    electrodes, cells, currents = _scene()
    mea = MEA(electrodes, output_radius=500.0)
    d = mea.distances(cells)

    def loss(device):
        return device.potential_recording(d, currents).sum()

    grad = jax.grad(loss)(mea)
    g = grad.medium.electrical.sigma_s_per_m
    _finite_and_nonzero(g)
    # recording is 1/sigma: dV/dsigma = -V / sigma
    expected = -float(loss(mea)) / mea.medium.electrical.sigma_s_per_m
    assert float(g) == pytest.approx(expected, rel=1e-4)

    assert float(grad.medium.electrical.stimulation_resistivity_ohm_m) == 0.0


def test_a_fitted_sigma_reaches_stimulation_only_when_the_override_is_none():
    electrodes, cells, _ = _scene()

    def induction(device):
        return device.cell_induction(device.distances(cells))[:, -1].sum()

    shipped = jax.grad(induction)(MEA(electrodes)).medium.electrical
    assert float(shipped.sigma_s_per_m) == 0.0

    tied = MEA(
        electrodes,
        medium=Medium(electrical=PointSource(stimulation_resistivity_ohm_m=None)),
    )
    grad = jax.grad(induction)(tied).medium.electrical
    _finite_and_nonzero(grad.sigma_s_per_m)
    assert float(grad.height_um) != 0.0


def test_the_optical_medium_differentiates_in_scattering():
    fibers = np.array([[0, 0.0, 0.0, 0.0]])
    cells = np.array([[0, 30.0, 0.0, 120.0], [1, -60.0, 20.0, 300.0]])
    array = LightArray(fibers)
    grad = jax.grad(lambda a: a.cell_induction(cells).sum())(array)
    g = grad.medium.optical.scattering_per_mm
    _finite_and_nonzero(g)
    assert float(g) < 0.0  # more scattering, less light


def test_the_dispersive_width_differentiates():
    electrodes, cells, currents = _scene()
    mea = MEA(
        electrodes,
        output_radius=500.0,
        medium=Medium(electrical=DispersivePointSource()),
    )
    d = mea.distances(cells)
    target = jnp.zeros((2, currents.shape[1]))

    def loss(device):
        return ((device.potential_recording(d, currents, dt=DT) - target) ** 2).mean()

    grad = jax.grad(loss)(mea).medium.electrical
    _finite_and_nonzero(grad.tissue_alpha_ms_per_um)
    _finite_and_nonzero(grad.sigma_s_per_m)


def test_gradients_flow_through_each_transfer_leaf():
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 200)), dtype=jnp.float32)

    rc = jax.grad(lambda t: (t(x, dt=DT) ** 2).sum())(ElectrodeRC([20.0, 40.0]))
    _finite_and_nonzero(rc.log_r_seal)
    assert rc.log_r_seal.shape == (2,)

    fir = FIR.identity(4)
    g = jax.grad(lambda t: t(x).sum())(fir).taps
    # d sum(y) / d taps[k] is the sum of the signal delayed by k
    expected = [float(x[:, : x.shape[1] - k].sum()) for k in range(4)]
    np.testing.assert_allclose(np.asarray(g), expected, rtol=1e-4, atol=1e-3)

    per = jax.grad(lambda t: (t(x) ** 2).sum())(FIR.identity(3, n_channels=2)).taps
    assert per.shape == (2, 3)
    _finite_and_nonzero(per)

    band = Bandpass(300.0, 3000.0, 2)
    gx = jax.grad(lambda v: (band(v, dt=0.02) ** 2).sum())(x)
    _finite_and_nonzero(gx)


def test_the_scan_stages_match_the_numpy_reference():
    from scipy.signal import lfilter, sosfilt

    x = np.random.default_rng(2).normal(size=(2, 500)).astype(np.float32)
    beta = np.exp(-DT / (50.0 * 100.0 * 1e-3))
    ref = np.stack([lfilter([1 - beta], [1, -beta], row) for row in x])
    np.testing.assert_allclose(
        np.asarray(ElectrodeRC(50.0, 100.0)(x, dt=DT)), ref, rtol=1e-4, atol=1e-5
    )

    band = Bandpass(300.0, 3000.0, 3)
    np.testing.assert_allclose(
        np.asarray(band(x, dt=0.02)),
        sosfilt(band.sos(0.02), x, axis=-1),
        rtol=1e-3,
        atol=1e-4,
    )

    taps = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(FIR(taps)(x)), lfilter(taps, [1.0], x, axis=-1), rtol=1e-4, atol=1e-5
    )


def _composition():
    electrodes, cells, currents = _scene()
    composed = ComposedIO(
        LightArray([[0, 0.0, 0.0, 0.0]]),
        [
            MEA(electrodes, output_radius=500.0),
            ElectrodeRC([20.0, 30.0]),
            FIR.identity(3),
            Noise(2.0),
        ],
        medium=Medium(electrical=PointSource(stimulation_resistivity_ohm_m=None)),
    )
    return composed, cells, currents


def test_the_medium_leaves_appear_once_in_a_composition():
    composed, _, _ = _composition()
    leaves = jax.tree_util.tree_leaves(composed)
    expected = jax.tree_util.tree_leaves(composed.medium) + jax.tree_util.tree_leaves(
        composed.transfers
    )
    assert len(leaves) == len(expected)
    # PointSource (sigma, height, min distance, resistivity->None gives none) + KM (2) + RC + FIR + Noise
    assert len(jax.tree_util.tree_leaves(composed.medium)) == 3 + 2

    rebuilt = jax.tree_util.tree_map(lambda v: v * 2.0, composed)
    assert rebuilt.medium.electrical.sigma_s_per_m == 0.6
    assert rebuilt.inputs.medium is rebuilt.outputs.medium is rebuilt.medium
    assert rebuilt.transfers[0] is rebuilt.outputs._stack[0]


def test_a_gradient_through_a_composition_is_not_counted_twice():
    composed, cells, currents = _composition()
    d = composed.distances(cells)

    def through_composition(io):
        recorded = io.potential_recording(d, currents, dt=DT).sum()
        lit = io.inputs.cell_induction(cells).sum()
        return recorded + lit

    def through_parts(medium):
        mea = MEA(
            composed.outputs.electrode_coordinates, output_radius=500.0, medium=medium
        )
        array = LightArray([[0, 0.0, 0.0, 0.0]], medium=medium)
        rc, fir, _noise = composed.transfers
        recorded = fir(rc(mea.potential_recording(d, currents), dt=DT)).sum()
        return recorded + array.cell_induction(cells).sum()

    whole = jax.grad(through_composition)(composed)
    parts = jax.grad(through_parts)(composed.medium)

    for got, want in zip(
        jax.tree_util.tree_leaves(whole.medium),
        jax.tree_util.tree_leaves(parts),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4)
    _finite_and_nonzero(whole.medium.electrical.sigma_s_per_m)
    _finite_and_nonzero(whole.medium.optical.scattering_per_mm)
    _finite_and_nonzero(whole.transfers[0].log_r_seal)
    _finite_and_nonzero(whole.transfers[1].taps)


def test_a_composition_traces_under_jit_and_keys_noise():
    composed, cells, currents = _composition()
    d = composed.distances(cells)

    @jax.jit
    def run(io, key):
        return io.potential_recording(d, currents, dt=DT, key=key)

    quiet = run(composed, None)
    again = run(composed, None)
    np.testing.assert_array_equal(np.asarray(quiet), np.asarray(again))
    eager = composed.potential_recording(d, currents, dt=DT)
    np.testing.assert_allclose(
        np.asarray(quiet), np.asarray(eager), rtol=1e-4, atol=1e-3
    )

    noisy = run(composed, jax.random.PRNGKey(0))
    assert float(jnp.std(noisy - quiet)) == pytest.approx(2.0, rel=0.3)


def test_the_default_gains_are_bit_identical_under_jax():
    pi, floor, h = 3.141592653589793, 5.0, (15.0**2 + 50.0**2) ** 0.5
    recording = 1000.0 / (4.0 * pi * 0.0003 * floor)
    stimulation = 1000.0 * 3.5 / (4.0 * pi * h)
    d = jnp.asarray(np.linspace(0.0, 500.0, 101))

    source = PointSource()
    assert np.array_equal(
        np.asarray(source.potential_gain(d)),
        np.asarray(recording * (floor / (d + floor))),
    )
    shape = h / np.sqrt(np.asarray(d) ** 2 + h**2)
    assert np.array_equal(
        np.asarray(source.field_gain(d, 15.0)), np.asarray(stimulation * shape)
    )

    medium = KubelkaMunk()
    offsets = jnp.asarray(np.random.default_rng(0).uniform(-300, 300, (2, 5, 3)))
    got = medium.irradiance_gain(offsets, 0.37, 100.0)
    z0 = 0.1 * 1.36 / 0.37
    diff = offsets / 1000.0
    r = jnp.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
    z = jnp.abs(diff[..., 2])
    want = (
        (0.37**2 / (1.36**2 * jnp.maximum((z + z0) ** 2, 1e-12)))
        * jnp.exp(-11.2 * jnp.sqrt(r**2 + z**2))
        / (4 * np.pi)
    )
    assert np.array_equal(np.asarray(got), np.asarray(want))
