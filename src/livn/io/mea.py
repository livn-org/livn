from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as _np

from livn.backend import backend
from livn.io.base import (
    IO,
    calculate_cell_stimulus,
    calculate_distances,
    channel_recording,
    coupled_sections,
    induction_matrix,
    relative_distance,
    with_amplitude,
)
from livn.io.medium import Medium
from livn.io.transfer import Noise
from livn.utils import import_instance

if TYPE_CHECKING:
    from livn.stimulus import Stimulus
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


def _checked_profile(profile: Mapping) -> dict:
    distance = _np.asarray(profile["distance_um"], dtype=float)
    trough = _np.asarray(profile["trough_uv"], dtype=float)
    if (
        distance.ndim != 1
        or distance.shape != trough.shape
        or (_np.diff(distance) <= 0).any()
    ):
        raise ValueError(
            "a spike profile needs increasing `distance_um` and a `trough_uv` "
            "of the same length"
        )
    return {"distance_um": distance.tolist(), "trough_uv": trough.tolist()}


class MEA(IO):
    """
    A multi-electrode array

    # Arguments

    electrode_coordinates
        Array of shape (n_electrodes, 4) containing the coordinates of electrodes.
        Each row represents [id, x, y, z] for an electrode.
    input_radius
        Radius within which an electrode can stimulate neurons, in micrometers.
        Default is 250.
    output_radius
        Radius within which an electrode can record from neurons, in micrometers.
        Default is 250.
    detection_threshold
        Smallest detectable signal, in the same relative units as the
        `amplitude` passed to `channel_recording`. Default is 0, which
        records every unit inside `output_radius`.
    threshold_sigma
        Physical detection, used instead of `detection_threshold` when the
        composition's stack has `Noise` stages (`detection_noise_uv > 0`): a cell
        is on a channel when its spike there reaches `threshold_sigma` times the
        stack's noise (the variances of its `Noise` stages summed). A bare MEA
        has no stack and so no physical detection.
    electrode_radius_um
        The electrode's radius: it sets the field the electrode imposes in
        stimulation (the medium supplies the height and the conductivity).
    medium
        The `Medium` (or its document); the electrical part carries recording
        and stimulation. A `ComposedIO` replaces it with the one it owns.
    spike_profile
        The spike's trough against distance (`{"distance_um", "trough_uv"}`).
    dead_time_ms
        A detector's refractory period per channel: spikes on one channel
        closer than this to the previous kept one are dropped
        (`apply_dead_time`). 0 keeps every spike.

    # Computed attributes

    cell_measurement
        Array containing the relative measurement weights for each neuron-electrode pair,
        filtered based on output_radius.
    """

    def __init__(
        self,
        electrode_coordinates: Float[Array, "n_electrodes ixyz=4"] | None = None,
        input_radius=250,
        output_radius=250,
        electrode_radius_um: float = 15.0,
        medium=None,
        detection_threshold: float = 0.0,
        threshold_sigma: float = 5.0,
        dead_time_ms: float = 0.0,
        spike_profile: Mapping | None = None,
    ):
        if electrode_coordinates is None:
            electrode_coordinates = self.default_electrode_coordinates()
        self.electrode_coordinates = np.array(electrode_coordinates)
        self.input_radius = input_radius
        self.output_radius = output_radius
        self.detection_threshold = float(detection_threshold)
        self.threshold_sigma = float(threshold_sigma)
        self.dead_time_ms = float(dead_time_ms)
        self.spike_profile = (
            None if spike_profile is None else _checked_profile(spike_profile)
        )
        self._attached_profile = None
        self.electrode_radius_um = float(electrode_radius_um)
        self.medium = Medium() if medium is None else import_instance(medium)
        self._stack = ()

        self.cell_measurement = None
        self._cell_induction = None
        self._induction_cache = None

    def parameter_children(self) -> dict:
        return {"medium": self.medium}

    @property
    def electrical(self):
        return self.medium.require("electrical")

    @property
    def needs_dt(self) -> bool:
        return bool(self.electrical.needs_dt)

    @property
    def detection_noise_uv(self) -> float:
        variance = sum(
            float(t.sigma_uv) ** 2 for t in self._stack if isinstance(t, Noise)
        )
        return variance**0.5

    @property
    def num_channels(self) -> int:
        return len(self.electrode_coordinates)

    @property
    def input_units(self) -> str:
        return "uA"

    @property
    def channel_ids(self) -> Int[Array, " n_channel_ids"]:
        return self.electrode_coordinates[:, 0].astype(np.int32)

    def serialize(self) -> dict:
        return {
            "electrode_coordinates": _np.asarray(self.electrode_coordinates).tolist(),
            "input_radius": self.input_radius,
            "output_radius": self.output_radius,
            "electrode_radius_um": self.electrode_radius_um,
            "detection_threshold": self.detection_threshold,
            "threshold_sigma": self.threshold_sigma,
            "dead_time_ms": self.dead_time_ms,
            **(
                {"spike_profile": self.spike_profile}
                if self.spike_profile is not None
                else {}
            ),
            "medium": self.medium.serialize(),
        }

    @property
    def physical_detection(self) -> bool:
        """Whether cells are detected against the stack's channel noise."""
        return self.detection_noise_uv > 0.0

    @property
    def profile(self) -> dict | None:
        return (
            self.spike_profile
            if self.spike_profile is not None
            else self._attached_profile
        )

    def attach_spike_profile(self, profile: Mapping | None) -> None:
        self._attached_profile = None if profile is None else _checked_profile(profile)

    def signal(
        self,
        cell_measurement: Float[Array, "n cip=3"],
        amplitude: Mapping | Float[Array, " n_gids"] | None = None,
    ) -> Float[Array, " n"]:
        """Each measurement row's signal in uV for physical detection, else relative.

        Relative: `amplitude * (1 - distance / output_radius)`. Physical:
        `amplitude * spike_profile(distance)`.
        """
        rows = _np.asarray(cell_measurement)
        if amplitude is None:
            scale = _np.ones(rows.shape[0])
        elif isinstance(amplitude, Mapping):
            scale = _np.array([float(amplitude.get(int(g), 1.0)) for g in rows[:, 1]])
        else:
            values = _np.asarray(amplitude)
            gids = rows[:, 1].astype(int)
            scale = (
                values[gids] if values.ndim else _np.full(rows.shape[0], float(values))
            )
        if self.physical_detection:
            distance = rows[:, -1] * float(self.output_radius)
            profile = self.profile
            if profile is None:
                raise ValueError(
                    "physical detection (a Noise stage in the stack) needs the cells' spike profile"
                )
            trough = _np.interp(
                distance,
                profile["distance_um"],
                profile["trough_uv"],
                right=0.0,
            )
            return scale * trough
        return scale * (1.0 - rows[:, -1])

    def apply_dead_time(self, per_channel: Mapping) -> dict:
        """Per-channel spike times with the detector's dead time applied."""
        if self.dead_time_ms <= 0.0:
            return dict(per_channel)
        out = {}
        for channel, times in per_channel.items():
            times = _np.sort(_np.asarray(times, dtype=float))
            kept, last = [], -_np.inf
            for t in times:
                if t - last >= self.dead_time_ms:
                    kept.append(t)
                    last = t
            out[channel] = _np.asarray(kept, dtype=float)
        return out

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
        dt: float = 1.0,
    ) -> Stimulus:
        """Map per-channel inputs (uA) to extracellular voltages (mV)."""
        from livn.stimulus import Stimulus

        autobatch = False
        if len(channel_inputs.shape) == 2:
            autobatch = True
            channel_inputs = channel_inputs[np.newaxis, ...]

        if self._cell_induction is None:
            distances = self.distances(neuron_coordinates)

            self._cell_induction = self.cell_induction(distances)

        dtype = np.promote_types(channel_inputs.dtype, np.float32)
        key = (len(neuron_coordinates), int(channel_inputs.shape[-1]), dtype.str)
        cached = getattr(self, "_induction_cache", None)
        if cached is None or cached[0] != key:
            gids, sections, keep = coupled_sections(
                neuron_coordinates, self._cell_induction
            )
            matrix, n_reached = induction_matrix(
                self._cell_induction,
                n_gids=len(neuron_coordinates),
                n_channels=int(channel_inputs.shape[-1]),
                keep=keep,
                dtype=dtype,
            )
            cached = (key, gids, sections, keep, matrix, n_reached)
            self._induction_cache = cached
        _key, gids, sections, keep, matrix, n_reached = cached

        stimulus = calculate_cell_stimulus(
            channel_inputs,
            matrix=matrix,
            n_reached=n_reached,
        )

        return Stimulus(
            stimulus[0] if autobatch else stimulus,
            dt=dt,
            gids=gids,
            sections=sections,
            input_mode="extracellular",
            units="mV",
        )

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"] | None,
        ii: Float[Array, " i"],
        *recordings: Float[Array, " _"],
        amplitude: Mapping | Float[Array, " n_gids"] | None = None,
    ) -> tuple[dict[int, Array], ...]:
        """Per-channel recordings, keeping only the units an electrode detects."""
        if ii is None:
            ii = np.unique(neuron_coordinates[:, 0])

        if self.cell_measurement is None:
            self.cell_measurement = relative_distance(
                self.distances(neuron_coordinates),
                self.output_radius,
                filter_out_of_bounds=True,
            )

        return channel_recording(
            self.detected(self.cell_measurement, amplitude), ii, *recordings
        )

    def detected(
        self,
        cell_measurement: Float[Array, "n cip=3"],
        amplitude: Mapping | Float[Array, " n_gids"] | None = None,
    ) -> Float[Array, "n cip=3"]:
        """The measurement rows whose signal reaches the detection threshold.

        `threshold_sigma * detection_noise_uv` with physical detection, `detection_threshold`
        otherwise.
        """
        if self.physical_detection:
            rows = _np.asarray(cell_measurement)
            limit = self.threshold_sigma * self.detection_noise_uv
            return rows[self.signal(rows, amplitude) >= limit]
        if self.detection_threshold <= 0.0 and amplitude is None:
            return cell_measurement

        rows = _np.asarray(cell_measurement)
        return rows[self.signal(rows, amplitude) >= self.detection_threshold]

    def strongest_units(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        amplitude: Mapping | Float[Array, " n_gids"] | None = None,
    ) -> dict[int, int]:
        """`{channel: gid}`: the detected cell with the largest signal on each electrode."""
        rows = _np.asarray(
            self.detected(
                relative_distance(
                    self.distances(neuron_coordinates),
                    self.output_radius,
                    filter_out_of_bounds=True,
                ),
                amplitude,
            )
        )
        if rows.size == 0:
            return {}
        gids = rows[:, 1].astype(int)
        signal = self.signal(rows, amplitude)
        best: dict[int, tuple[float, int]] = {}
        for channel, gid, value in zip(
            rows[:, 0].astype(int), gids, signal, strict=True
        ):
            if channel not in best or value > best[channel][0]:
                best[int(channel)] = (float(value), int(gid))
        return {channel: gid for channel, (_v, gid) in sorted(best.items())}

    def _pair_distances(self, distances):
        n_electrodes = int(self.num_channels)
        d = distances[:, -1]
        if d.size % n_electrodes != 0:
            raise ValueError(
                f"distances size mismatch: expected a multiple of {n_electrodes}, got {d.size}"
            )
        return d.reshape(n_electrodes, d.size // n_electrodes)

    def source_gain(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_channels n_recording_coords"]:
        """
        Compute per-recording-coordinate gain for electrode potential.

        Only recording coordinates within output_radius get non-zero gain.

        Arguments
        - distances: as returned by `MEA.distances(neuron_coordinates)`
        """
        d = self._pair_distances(distances)
        gain = self.electrical.potential_gain(d)
        mask = d <= float(self.output_radius)
        return np.where(mask, gain, 0.0)

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "n_neurons timestep"],
        dt: float | None = None,
        key=None,
    ) -> Float[Array, "n_channels timestep"]:
        """
        Estimate electrode potentials from membrane currents through the
        medium (a point source: `V [uV] = sum_i g(r_i) I_i [uA]`).

        Only neurons within output_radius contribute.

        Arguments
        - distances: as returned by `MEA.distances(neuron_coordinates)`
        - membrane_currents: in uA, shape [n_neurons, timestep]
        - dt: sampling interval (ms); needed only by a medium that filters in time
        - key: unused (a bare MEA has no stochastic stage)

        Returns: microvolts array (uV), shape [n_channels, timestep]
        """
        gain = self.source_gain(distances)
        return self.electrical.project(
            gain,
            self._pair_distances(distances),
            membrane_currents,
            dt,
            output_radius_um=float(self.output_radius),
        )

    def distances(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_sources*n_coords cip=3"]:
        return calculate_distances(self.electrode_coordinates, neuron_coordinates)

    def default_electrode_coordinates(self):
        return electrode_array_coordinates()

    def cell_induction(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_amplitudes cip=3"]:
        return with_amplitude(
            distances,
            self.electrical.field_gain(distances[:, -1], self.electrode_radius_um),
        )

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        """mV induced per uA at each coordinate row, per electrode."""
        induction = np.asarray(self.cell_induction(self.distances(neuron_coordinates)))
        return induction[:, -1].reshape(len(self.channel_ids), -1)

    def channel_contribution(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_id: int,
    ):
        import pandas as pd

        distances = self.distances(neuron_coordinates)
        full_gain = self.source_gain(distances)  # (n_channels, n_recording_coords)

        ch_idx = int(np.where(self.channel_ids == channel_id)[0][0])
        ch_gain = full_gain[ch_idx]

        mask = distances[:, 0] == channel_id
        ch = distances[mask]

        within = ch_gain > 0
        ch = ch[within]
        ch_gain = ch_gain[within]

        order = ch[:, 2].argsort()
        ch = ch[order]
        ch_gain = ch_gain[order]

        total = ch_gain.sum()
        contribution_pct = ch_gain / total * 100 if total > 0 else ch_gain * 0

        return pd.DataFrame(
            {
                "gid": ch[:, 1].astype(int),
                "distance_um": ch[:, 2],
                "gain": ch_gain,
                "contribution_pct": contribution_pct,
            }
        )


if _USES_JAX:

    def _mea_tree_flatten(mea):
        aux = {
            "electrode_coordinates": mea.electrode_coordinates,
            "input_radius": mea.input_radius,
            "output_radius": mea.output_radius,
            "electrode_radius_um": mea.electrode_radius_um,
            "cell_measurement": mea.cell_measurement,
            "_cell_induction": mea._cell_induction,
            "detection_threshold": mea.detection_threshold,
            "threshold_sigma": mea.threshold_sigma,
            "dead_time_ms": mea.dead_time_ms,
            "spike_profile": mea.spike_profile,
            "_attached_profile": mea._attached_profile,
            "_stack": mea._stack,
        }
        return (mea.medium,), aux

    def _mea_tree_unflatten(aux, children):
        (medium,) = children
        mea = MEA.__new__(MEA)
        mea.medium = medium
        for name, value in aux.items():
            setattr(mea, name, value)
        mea._induction_cache = None
        return mea

    jax.tree_util.register_pytree_node(MEA, _mea_tree_flatten, _mea_tree_unflatten)


def electrode_array_coordinates(
    pitch: float = 1000, xs: int = 4, ys: int = 4, xoffset=500, yoffset=500, z=350 / 2
) -> Float[Array, "n_electrodes ixyz=4"]:
    """
    Generate regular electrode array coordinates.
    """
    coordinates = np.array(
        [
            [xoffset + x * pitch, yoffset + y * pitch, z]
            for y in range(ys)
            for x in range(xs)
        ]
    )

    return np.hstack((np.arange(coordinates.shape[0]).reshape(-1, 1), coordinates))


def electrode_array_coordinates_for_area(
    pitch: float, area: tuple[tuple[float, float], tuple[float, float]], z=0
) -> Float[Array, "n_electrodes ixyz=4"]:
    """
    Generate electrode array coordinates that fit within a given area

    The number of electrodes in each dimension is the largest centered power-of-2
    that fits within the area given the pitch

    Arguments:
        pitch: spacing between electrodes
        area: ((x_min, y_min), (x_max, y_max))
        z: z coordinate for all electrodes
    """
    (x_min, y_min), (x_max, y_max) = area
    width = x_max - x_min
    height = y_max - y_min

    max_xs = int(width // pitch) + 1
    max_ys = int(height // pitch) + 1

    xs = 1 << (max_xs.bit_length() - 1) if max_xs >= 1 else 1
    ys = 1 << (max_ys.bit_length() - 1) if max_ys >= 1 else 1

    grid_width = (xs - 1) * pitch if xs > 1 else 0
    grid_height = (ys - 1) * pitch if ys > 1 else 0

    xoffset = x_min + (width - grid_width) / 2
    yoffset = y_min + (height - grid_height) / 2

    return electrode_array_coordinates(
        pitch=pitch, xs=xs, ys=ys, xoffset=xoffset, yoffset=yoffset, z=z
    )
