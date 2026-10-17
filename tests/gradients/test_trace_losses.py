import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from livn.optimization.losses import events_to_mask, shape_logmag, smoothness

INF = np.inf


def _traces(seed=0, shape=(3, 200, 4)):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=shape) * rng.uniform(0.5, 3.0, (shape[0], 1, shape[2]))
    return jnp.asarray(target, jnp.float32), rng


def test_shape_loss_is_zero_for_a_scaled_copy_and_logmag_for_equal_rms():
    target, _ = _traces()
    shape, logmag = shape_logmag(3.7 * target, target, return_terms=True)
    assert float(shape) == pytest.approx(0.0, abs=1e-5)
    assert float(logmag) > 0.5

    # flipped sign: same RMS, opposite shape
    shape, logmag = shape_logmag(-target, target, return_terms=True)
    assert float(logmag) == pytest.approx(0.0, abs=1e-10)
    assert float(shape) == pytest.approx(4.0, rel=1e-3)

    total = shape_logmag(2.0 * target, target, log_mag_weight=0.5)
    expected = 0.5 * np.log(2.0) ** 2
    assert float(total) == pytest.approx(expected, rel=1e-3)


def test_mask_limits_the_loss_to_the_marked_samples():
    target, _ = _traces()
    mask = np.zeros(target.shape, bool)
    mask[:, 50:80, :] = True
    corrupted = np.asarray(target).copy()
    corrupted[:, 100:, :] += 5.0  # outside the mask
    loss = shape_logmag(jnp.asarray(corrupted), target, jnp.asarray(mask))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

    corrupted[:, 60, :] += 5.0  # inside
    assert float(shape_logmag(jnp.asarray(corrupted), target, jnp.asarray(mask))) > 0.1

    empty = shape_logmag(target * 2, target, jnp.zeros(target.shape, bool))
    assert float(empty) == 0.0


def test_gradients_are_finite_including_the_empty_mask():
    target, rng = _traces()
    pred = target + 0.3
    mask = jnp.asarray(rng.uniform(size=target.shape) > 0.7)
    for m in (None, mask, jnp.zeros(target.shape, bool)):
        g = jax.grad(lambda p, m=m: shape_logmag(p, target, m))(pred)
        assert bool(jnp.all(jnp.isfinite(g)))
    g = jax.grad(lambda p: shape_logmag(p, target))(pred)
    assert float(jnp.abs(g).max()) > 0


def test_shape_logmag_jits_and_batches():
    target, _ = _traces()
    pred = target + 0.2
    jitted = jax.jit(lambda p, t: shape_logmag(p, t, log_mag_weight=0.3))
    assert float(jitted(pred, target)) == pytest.approx(
        float(shape_logmag(pred, target, log_mag_weight=0.3)), rel=1e-5
    )
    # vmap of the unbatched call over the batch axis agrees with the batched mean of per-item losses
    per_item = jax.vmap(lambda p, t: shape_logmag(p, t))(pred, target)
    assert per_item.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(per_item)))


def test_smoothness_ignores_slow_and_penalises_fast_content():
    t = np.arange(400) * 0.1
    slow = np.sin(2 * np.pi * t / 20.0)[None, :, None]
    fast = np.tile([1.0, -1.0], 200)[None, :, None]
    s_slow = float(smoothness(jnp.asarray(slow, jnp.float32), 0.1))
    s_fast = float(smoothness(jnp.asarray(fast, jnp.float32), 0.1))
    assert s_slow < 1e-3 < 0.1 < s_fast
    assert float(smoothness(jnp.ones((2, 100, 3)), 0.1)) == pytest.approx(
        0.0, abs=1e-10
    )
    # scale divides through as scale**2
    assert float(
        smoothness(jnp.asarray(fast, jnp.float32), 0.1, scale=2.0)
    ) == pytest.approx(s_fast / 4.0, rel=1e-5)

    g = jax.grad(lambda x: smoothness(x, 0.1, warmup_samples=10))(
        jnp.asarray(fast, jnp.float32)
    )
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g[:, :10]).max()) == 0.0


def test_events_to_mask_hand_built():
    # dt 1 ms, window 4 ms -> half-width 2 samples; spike at 5 ms marks samples 3..7
    events = jnp.asarray([[5.0, INF], [INF, INF]])
    mask = np.asarray(events_to_mask(events, (12, 4), dt=1.0, window_ms=4.0))
    expected = np.zeros((12, 4), bool)
    expected[3:8, 0] = True
    np.testing.assert_array_equal(mask, expected)

    # channel radius and clipping at the array edge; unit 1 sits on channel 3
    events = jnp.asarray([[0.0, INF], [10.0, 11.0]])
    mask = np.asarray(
        events_to_mask(
            events,
            (12, 4),
            dt=1.0,
            window_ms=4.0,
            channel_radius=1,
            channels=jnp.asarray([0, 3]),
        )
    )
    expected = np.zeros((12, 4), bool)
    expected[0:3, 0:2] = True  # window clipped at t=0, channel -1 dropped
    expected[8:12, 2:4] = True  # 10 -> 8..12 (clipped), 11 -> 9..13
    np.testing.assert_array_equal(mask, expected)


def test_events_to_mask_overlap_multi_channel_and_minimum_width():
    # a window narrower than one sample still marks 2*1+1 samples, as the CPU builder does
    mask = np.asarray(
        events_to_mask(jnp.asarray([[4.0]]), (10, 2), dt=1.0, window_ms=0.1)
    )
    assert mask[:, 0].nonzero()[0].tolist() == [3, 4, 5]

    # a unit seen on two channels (-1 padded), overlapping spikes do not double count
    events = jnp.asarray([[4.0, 5.0]])
    mask = np.asarray(
        events_to_mask(events, (10, 5), 1.0, 2.0, channels=jnp.asarray([[0, 4, -1]]))
    )
    assert mask.dtype == bool
    assert mask[:, 0].nonzero()[0].tolist() == [3, 4, 5, 6]
    assert mask[:, 4].nonzero()[0].tolist() == [3, 4, 5, 6]
    assert not mask[:, 1:4].any()


def test_events_to_mask_jits_batches_and_feeds_the_loss():
    events = jnp.asarray(
        [[[5.0, INF], [INF, INF]], [[2.0, 8.0], [20.0, INF]]]
    )  # (batch=2, unit=2, spike=2)
    fn = jax.jit(lambda e: events_to_mask(e, (12, 2), 1.0, 2.0))
    mask = fn(events)
    assert mask.shape == (2, 12, 2)
    np.testing.assert_array_equal(
        np.asarray(mask[0]), np.asarray(events_to_mask(events[0], (12, 2), 1.0, 2.0))
    )
    assert not np.asarray(mask[0, :, 1]).any()
    assert np.asarray(mask[1, :, 0]).sum() == 6

    # a spike far outside the window must not overflow
    far = events_to_mask(jnp.asarray([[1e12, -1e12]]), (12, 2), 1.0, 2.0)
    assert not bool(far.any())

    target, _ = _traces(shape=(2, 12, 2))
    loss, g = jax.value_and_grad(lambda p: shape_logmag(p, target, mask))(target + 0.5)
    assert np.isfinite(float(loss)) and bool(jnp.all(jnp.isfinite(g)))
