import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy import signal

from livn.io import Bandpass, detection
from livn.io.detection import (
    Detector,
    MethodError,
    threshold_method,
)
from livn.utils import canonical_hash

FS = 10_000.0


def _reference(x, fs, k, window, low, high, dead_ms, stride=5, sigma=None):
    """The rule written out per sample: ``window`` is ``[start, end)`` in frames."""
    hp = signal.butter(2, low, "highpass", fs=fs, output="sos")
    lp = signal.butter(2, high, "lowpass", fs=fs, output="sos")
    y = signal.sosfilt(lp, signal.sosfilt(hp, x, axis=0), axis=0)
    start = 0
    if sigma is None:
        s0, s1 = window
        head = np.abs(y[s0:s1:stride]).astype(np.float32).astype(np.float64)
        sigma = np.median(head, axis=0) / 0.6745
        start = s1
    dead = max(int(fs / 1000 * dead_ms), 1)
    events = []
    for row in range(y.shape[1]):
        thr, kept, refractory = -k * sigma[row], [], 0
        for n in range(start, len(y)):
            if refractory > 0:
                refractory -= 1
                continue
            if y[n, row] < thr:
                kept.append(n)
                refractory = dead
        events.append(kept)
    return y, sigma, events


def _noise(seed=0, n=20_000, rows=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=10.0, size=(n, rows))
    # negative spikes, some closer together than the dead time
    for row in range(rows):
        for at in rng.choice(np.arange(3_000, n - 20), 40, replace=False):
            x[at : at + 3, row] -= 120.0
        x[5_000:5_008:4, row] -= 150.0
    return x


def test_the_filter_is_a_causal_order_2_cascade():
    band = Bandpass.from_filter(threshold_method(k=5.0, sigma_window=100.0)["filter"])
    assert (band.low, band.high, band.order) == (300.0, 3000.0, 2)
    assert (band.form, band.phase) == ("cascade", "causal")


def test_the_bandpass_is_the_causal_reference_filter():
    x = _noise()
    method = threshold_method(k=4.0, sigma_window=500.0)
    y_ref, _, _ = _reference(x, FS, 4.0, (0, 5_000), 300.0, 3000.0, 1.0)
    np.testing.assert_allclose(
        Bandpass.from_filter(method["filter"]).filter(x, FS, axis=0), y_ref, atol=1e-9
    )


@pytest.mark.parametrize("window_ms", [250.0, (100.0, 400.0)])
def test_events_are_those_of_the_reference(window_ms):
    x = _noise()
    found = Detector.detect(x, FS, threshold_method(k=4.5, sigma_window=window_ms))
    window = (0, 2_500) if window_ms == 250.0 else (1_000, 4_000)
    _, sigma, events = _reference(x, FS, 4.5, window, 300.0, 3000.0, 1.0)
    np.testing.assert_allclose(found.sigma_uV, sigma, rtol=1e-9)
    np.testing.assert_allclose(found.threshold_uV, -4.5 * sigma, rtol=1e-9)
    assert found.state_from_frame == window[1]
    for got, want in zip(found.frames, events, strict=True):
        assert len(want) > 20
        np.testing.assert_array_equal(got, want)


def test_a_given_sigma_searches_from_the_first_frame():
    x = _noise(seed=2)
    sigma = np.array([3.0, 4.0, 5.0])
    found = Detector.detect(
        x, FS, threshold_method(k=5.0, sigma_window=250.0), sigma_uV=sigma
    )
    _, _, events = _reference(x, FS, 5.0, None, 300.0, 3000.0, 1.0, sigma=sigma)
    assert found.state_from_frame == 0
    np.testing.assert_array_equal(found.sigma_uV, sigma)
    for got, want in zip(found.frames, events, strict=True):
        np.testing.assert_array_equal(got, want)


def test_sigma_is_the_median_of_abs_over_strided_frames():
    rng = np.random.default_rng(4)
    x = rng.normal(scale=10.0, size=(4_000, 2))
    method = threshold_method(k=5.0, sigma_window=(10.0, 210.0), sigma_stride=7)
    found = Detector.detect(x, FS, method)
    y = Bandpass.from_filter(method["filter"]).filter(x, FS, axis=0)
    head = np.abs(y[100:2_100:7]).astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(found.sigma_uV, np.median(head, axis=0) / 0.6745)


def test_nothing_is_found_before_sigma_is_frozen_and_the_dead_time_holds():
    x = np.zeros((4_000, 1))
    x[100, 0] = -1e3  # inside the sigma window
    x[999, 0] = -1e3  # the window's last frame, which is not searched either
    x[2_000, 0] = x[2_005, 0] = x[2_010, 0] = x[2_011, 0] = -1e3
    x += np.random.default_rng(1).normal(scale=1.0, size=x.shape)
    (frames,) = Detector.detect(
        x, FS, threshold_method(k=5.0, sigma_window=100.0)
    ).frames
    assert frames[0] >= 1_000  # nothing in the window, the sigma window
    found = frames[(frames >= 2_000) & (frames < 2_020)]
    # an event at 2000 holds the next 10 frames (1 ms); 2011 is the first one free again
    assert (
        found[0] == 2_000
        and 2_011 in found
        and not ((found > 2_000) & (found <= 2_010)).any()
    )


def test_the_crossing_is_level_triggered():
    # a 400 Hz sine's troughs stay below the threshold for longer than the dead time (10
    # frames), so each trough gives a second event 11 frames after its first
    t = np.arange(3_000) / FS
    x = (1e3 * np.sin(2 * np.pi * 400.0 * t))[:, None]
    method = threshold_method(k=5.0, sigma_window=100.0)
    y = Bandpass.from_filter(method["filter"]).filter(x, FS, axis=0)[:, 0]
    (frames,) = Detector.detect(x, FS, method, sigma_uV=np.array([1.0])).frames
    below = y < -5.0
    first = int(frames[0])
    assert below[first : first + 12].all()  # one trough, longer than the dead time
    assert frames[1] == first + 11


def test_blocks_find_what_the_whole_finds():
    x = _noise(seed=3)
    method = threshold_method(k=4.0, sigma_window=300.0)
    whole = Detector.detect(x, FS, method)
    detector = Detector(method, FS, x.shape[1])
    for start in range(0, len(x), 1_234):
        detector.feed(x[start : start + 1_234])
    blocks = detector.finish()
    np.testing.assert_array_equal(blocks.sigma_uV, whole.sigma_uV)
    for a, b in zip(blocks.frames, whole.frames, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_method_the_detector_does_not_run_is_refused():
    method = threshold_method(k=4.0, sigma_window=300.0)
    for block, name, value, match in (
        ("filter", "causal", False, "causal"),
        ("filter", "order", 9, "order"),
        ("threshold", "polarity", "positive", "polarity"),
        ("threshold", "sigma", "rms", "sigma"),
    ):
        changed = json.loads(json.dumps(method))
        changed[block][name] = value
        with pytest.raises(MethodError, match=match):
            detection.check_method(changed)
    with pytest.raises(MethodError, match="window must fit"):
        Detector.detect(np.zeros((10, 1)), FS, method)
    with pytest.raises(MethodError, match="sigma_window_ms"):
        threshold_method(k=4.0, sigma_window=(300.0, 100.0))
    with pytest.raises(MethodError, match="Nyquist"):
        Detector.detect(np.zeros((10_000, 1)), 5_000.0, method)


def test_the_closed_loop_method_is_rsfs_example_and_its_hash_follows_the_rule():
    method = threshold_method(k=5.0, sigma_window=60_000.0)
    assert canonical_hash(method) == (
        "0b522a151faaed495789b00692c99e31480a633e04a3f9a09ef9605a9d963d1d"
    )
    assert "sigma_stride" not in method["threshold"]  # 5, the rule's own
    assert canonical_hash(
        threshold_method(k=5.0, sigma_window=60_000.0, sigma_stride=7)
    ) != canonical_hash(method)
    assert canonical_hash(threshold_method(k=5.5, sigma_window=60_000.0)) != (
        canonical_hash(method)
    )
