from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in os.environ.get("LIVN_BACKEND", ""):
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


def section_positions(gids):
    seen: dict[int, int] = {}
    out = []
    for gid in gids:
        gid = int(gid)
        out.append(seen.get(gid, 0))
        seen[gid] = out[-1] + 1
    return np.asarray(out)


class Stimulus:
    def __init__(
        self,
        array: Float[Array, "timestep n_gids"],
        dt: float = 1.0,
        gids: Int[Array, "n_gids"] | None = None,
        input_mode: str = "extracellular",
        units: str | None = None,
        sections: Int[Array, "n_gids"] | None = None,
        **extra,
    ):
        self.array = array
        if dt <= 0:
            raise ValueError("Stimulus dt must be positive")
        self.dt = dt
        if gids is not None and len(gids) != array.shape[-1]:
            raise ValueError(
                f"stimulus has {array.shape[-1]} channels but {len(gids)} gids; "
                "the last axis of the array and the gid list must describe the "
                "same cells, in the same order"
            )
        if sections is not None:
            if gids is None:
                raise ValueError(
                    "`sections` names the section of each gid, so it "
                    "cannot be given without `gids`"
                )
            if len(sections) != len(gids):
                raise ValueError(
                    f"{len(sections)} sections for {len(gids)} gids; a column is "
                    "one section of one cell, so the two must line up"
                )
        self.gids = gids
        self.sections = sections
        self.input_mode = input_mode
        self.units = units
        self.extra = extra

    @property
    def duration(self) -> float:
        return self.array.shape[0] * self.dt

    def __iter__(self):
        """`(gid, section, series)` per column."""
        gids = self.gids
        if self.sections is not None:
            yield from zip(gids, self.sections, self.array.T)
            return

        per_cell = self.sections_per_cell
        for column, (gid, series) in enumerate(zip(gids, self.array.T)):
            yield gid, column % per_cell, series

    @property
    def sections_per_cell(self) -> int:
        """Columns each cell has, for a stimulus that does not say."""
        if self.gids is None:
            return 1
        import builtins

        distinct = len(builtins.set(int(g) for g in self.gids))
        return max(1, len(self.gids) // max(1, distinct))

    def __len__(self):
        return self.array.shape[-1]

    @classmethod
    def from_arg(cls, stimulus, env=None, duration=None) -> "Stimulus | None":
        if stimulus is None:
            return None

        if isinstance(stimulus, cls):
            return stimulus

        from livn.policy import Policy

        if isinstance(stimulus, Policy):
            if env is None or duration is None:
                raise ValueError(
                    "a Policy commands channels, not cells, and only covers "
                    "the window being simulated. Pass it to "
                    "`env.run(duration, stimulus=policy)`, which supplies both "
                    "or call `env.cell_stimulus(policy())` to convert one by "
                    "hand"
                )
            extent = stimulus.extent_ms
            if extent is not None and extent > duration + 1e-9:
                raise ValueError(
                    f"a {duration:g} ms run cannot deliver a {extent:g} ms "
                    f"{type(stimulus).__name__}: the pulses past the end would "
                    "be dropped and the run would look like a protocol that "
                    "was delivered in full. Run for at least the policy's own "
                    "length, or shorten the policy"
                )
            return env.cell_stimulus(
                stimulus.window(0.0, duration, stimulus.dt), dt=stimulus.dt
            )

        if hasattr(stimulus, "shape"):
            return cls(stimulus)

        if isinstance(stimulus, (tuple, list)):
            return cls(*stimulus)

        if isinstance(stimulus, dict):
            return cls(**stimulus)

        raise ValueError("Invalid stimulus", stimulus)

    @classmethod
    def from_conductance(
        cls,
        conductance: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **extra,
    ) -> "Stimulus":
        """Create stimulus from synaptic conductance values

        Args:
            conductance: in uS
            dt: Time step in ms
            gids: Neuron GIDs
            **extra: Additional metadata
        """
        return cls(
            array=conductance,
            dt=dt,
            gids=gids,
            input_mode="conductance",
            units="uS",
            **extra,
        )

    @classmethod
    def from_current(
        cls,
        current: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **extra,
    ) -> "Stimulus":
        """Create stimulus from direct current injection

        Args:
            current: Current values in nA
            dt: Time step in ms
            gids: Neuron GIDs
            **extra: Additional metadata
        """
        return cls(
            array=current, dt=dt, gids=gids, input_mode="current", units="nA", **extra
        )

    @classmethod
    def from_current_density(
        cls,
        current_density: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **extra,
    ) -> "Stimulus":
        """Create stimulus from current density

        Args:
            current_density: Current density values in mA/cm2
            dt: Time step in ms
            gids: Neuron GIDs
            **extra: Additional metadata
        """
        return cls(
            array=current_density,
            dt=dt,
            gids=gids,
            input_mode="current_density",
            units="mA/cm2",
            **extra,
        )

    @classmethod
    def from_extracellular(
        cls,
        voltage: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **extra,
    ) -> "Stimulus":
        return cls(
            array=voltage,
            dt=dt,
            gids=gids,
            input_mode="extracellular",
            units="mV",
            **extra,
        )

    @classmethod
    def from_irradiance(
        cls,
        irradiance: Float[Array, "timestep n_gids"],
        dt: float = 0.1,
        gids: Int[Array, "n_gids"] | None = None,
        **extra,
    ) -> "Stimulus":
        """Optical stimulus as irradiance at each neuron (mW/mm^2).

        Args:
            irradiance: Light power density at each neuron, shape [timestep, n_gids].
            **extra: Additional metadata, e.g. wavelength_nm=473.0.
        """
        return cls(
            array=irradiance,
            dt=dt,
            gids=gids,
            input_mode="irradiance",
            units="mW/mm2",
            **extra,
        )

    def convert_to(self, target_units: str) -> "Stimulus":
        """Convert stimulus to equivalent units

        Supported conversions:
            "mW/mm2" -> "photon_flux"  (irradiance -> photons/s/mm^2)
            "photon_flux" -> "mW/mm2"  (photons/s/mm^2 -> irradiance)

        Wavelength is read from extra["wavelength_nm"]
        """
        current_units = self.units
        if current_units == target_units:
            return self

        wavelength_nm = self.extra.get("wavelength_nm", 473.0)
        E_photon = 6.626e-34 * 3e8 / (wavelength_nm * 1e-9) * 1e3  # mW*s

        if current_units == "mW/mm2" and target_units == "photon_flux":
            converted = self.array / E_photon
        elif current_units == "photon_flux" and target_units == "mW/mm2":
            converted = self.array * E_photon
        else:
            raise ValueError(
                f"No conversion from '{current_units}' to '{target_units}'"
            )

        return Stimulus(
            array=converted,
            dt=self.dt,
            gids=self.gids,
            input_mode=self.input_mode,
            units=target_units,
            **self.extra,
        )

    @staticmethod
    def align_gids(
        stimulus: "Stimulus",
        all_gids: Int[Array, "n_total_gids"],
    ) -> "Stimulus":
        """Expand stimulus array to cover all_gids, zero-padding missing neurons"""
        if stimulus.gids is None:
            assert stimulus.array.shape[-1] == len(all_gids), (
                f"Stimulus has {stimulus.array.shape[-1]} columns but system has "
                f"{len(all_gids)} neurons. Set gids= explicitly."
            )
            return stimulus

        gid_to_idx = {int(g): i for i, g in enumerate(all_gids)}
        n_timesteps = stimulus.array.shape[0]
        expanded = np.zeros((n_timesteps, len(all_gids)), dtype=stimulus.array.dtype)
        for col_idx, gid in enumerate(stimulus.gids):
            sys_idx = gid_to_idx.get(int(gid))
            if sys_idx is None:
                raise ValueError(
                    f"Stimulus targets GID {gid} which is not in the system"
                )
            if _USES_JAX:
                expanded = expanded.at[:, sys_idx].add(stimulus.array[:, col_idx])
            else:
                expanded[:, sys_idx] += stimulus.array[:, col_idx]

        return Stimulus(
            array=expanded,
            dt=stimulus.dt,
            gids=all_gids,
            input_mode=stimulus.input_mode,
            units=stimulus.units,
            **stimulus.extra,
        )

    @staticmethod
    def resample(
        stimulus: "Stimulus",
        target_dt: float,
        duration: float,
    ) -> "Stimulus":
        """Resample stimulus to a common dt via linear interpolation"""
        if np.isclose(stimulus.dt, target_dt):
            return stimulus

        n_target_steps = int(round(duration / target_dt))
        t_target = np.linspace(0.0, duration, n_target_steps, endpoint=False)
        t_src = np.arange(stimulus.array.shape[0]) * stimulus.dt
        resampled = np.stack(
            [
                np.interp(t_target, t_src, stimulus.array[:, col])
                for col in range(stimulus.array.shape[-1])
            ],
            axis=-1,
        )
        return Stimulus(
            array=resampled,
            dt=target_dt,
            gids=stimulus.gids,
            input_mode=stimulus.input_mode,
            units=stimulus.units,
            **stimulus.extra,
        )

    def expand(self, gids, sections=None) -> "Stimulus":
        """Widen to the full column ordering `gids`/`sections` describe."""
        target_gids = np.asarray(gids)
        if self.gids is None:
            raise ValueError(
                "a stimulus with no `gids` does not say which columns it holds, "
                "so it cannot be widened; it is already assumed to be full width"
            )

        target_sections = (
            np.asarray(sections)
            if sections is not None
            else section_positions(target_gids)
        )
        own_sections = (
            self.sections
            if self.sections is not None
            else section_positions(np.asarray(self.gids))
        )

        width = len(target_gids)
        if (
            width == self.array.shape[-1]
            and np.array_equal(np.asarray(self.gids), target_gids)
            and np.array_equal(np.asarray(own_sections), target_sections)
        ):
            return self

        column = {
            (int(g), int(s)): i
            for i, (g, s) in enumerate(zip(target_gids, target_sections))
        }
        take = []
        into = []
        for own, (g, s) in enumerate(zip(np.asarray(self.gids), own_sections)):
            found = column.get((int(g), int(s)))
            if found is not None:
                take.append(own)
                into.append(found)

        array = np.asarray(self.array)
        full = np.zeros(array.shape[:-1] + (width,), dtype=array.dtype)
        if take:
            if _USES_JAX:
                full = full.at[..., np.asarray(into)].set(array[..., np.asarray(take)])
            else:
                full[..., np.asarray(into)] = array[..., np.asarray(take)]

        return Stimulus(
            full,
            dt=self.dt,
            gids=target_gids,
            sections=target_sections,
            input_mode=self.input_mode,
            units=self.units,
            **self.extra,
        )

    def to_array(self, duration: float, dt: float):
        """Resample and pad/trim to simulation time grid.

        Returns an array with ``int(duration / dt) + 1`` rows.
        Compatible with JAX tracers inside JIT.
        """
        arr = np.asarray(self.array)
        expected_steps = int(duration / dt) + 1
        original_ndim = arr.ndim

        if original_ndim == 1:
            arr = arr[:, None]

        if not _USES_JAX:
            if arr.shape[0] != expected_steps or not np.isclose(self.dt, dt):
                time_src = np.arange(arr.shape[0]) * self.dt
                time_target = np.linspace(0.0, duration, expected_steps)
                arr = np.stack(
                    [
                        np.interp(time_target, time_src, arr[:, col])
                        for col in range(arr.shape[1])
                    ],
                    axis=1,
                )

        if arr.shape[0] < expected_steps:
            pad = np.zeros(
                (expected_steps - arr.shape[0], arr.shape[1]), dtype=arr.dtype
            )
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > expected_steps:
            arr = arr[:expected_steps]

        if original_ndim == 1:
            arr = arr[:, 0]

        return arr

    def tree_flatten(self):
        children = [self.array]
        aux = (
            self.dt,
            self.gids,
            self.input_mode,
            self.units,
            self.sections,
            self.extra,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, gids, input_mode, units, sections, extra = aux
        return cls(
            array=children[0],
            dt=dt,
            gids=gids,
            sections=sections,
            input_mode=input_mode,
            units=units,
            **extra,
        )


if _USES_JAX:
    try:
        import jax

        jax.tree_util.register_pytree_node(
            Stimulus,
            Stimulus.tree_flatten,
            Stimulus.tree_unflatten,
        )
    except ImportError:
        pass
