from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as _np

from livn.backend import backend
from livn.io._component import (
    _USES_JAX,
    Component,
    _Base,
    cls_path,
    module,
    resolve_component,
    static_field,
    with_fields,
)
from livn.utils import is_traced

if "ax" in backend():
    import jax
    import jax.numpy as np
else:
    import numpy as np

_PI = 3.141592653589793
_DEFAULT = object()


def _induction_shape(distances_um, electrode_radius_um, culture_height_um):
    """1 at the electrode surface, falling as 1/sqrt(r^2 + h^2)."""
    xp = (
        _np
        if not is_traced(distances_um, electrode_radius_um, culture_height_um)
        else np
    )
    h_eff = (electrode_radius_um**2 + culture_height_um**2) ** 0.5
    return h_eff / xp.sqrt(distances_um**2 + h_eff**2)


def _source_shape(distances_um, min_distance_um):
    """1 at the electrode surface, falling as 1/r with a floor on r."""
    return min_distance_um / (distances_um + min_distance_um)


@runtime_checkable
class ElectricalMedium(Protocol):
    needs_dt: bool

    def potential_gain(self, distances_um):
        """uV per uA at each source-channel distance (recording)."""

    def field_gain(self, distances_um, electrode_radius_um):
        """mV per uA at each channel-cell distance (stimulation)."""

    def project(
        self, gain, distances_um, membrane_currents, dt=None, output_radius_um=None
    ):
        """Channel potentials `[C, T]` from the masked `gain` `[C, N]`, the
        distances `[C, N]` and `membrane_currents` `[N, T]`."""

    def serialize(self) -> dict: ...

    def set_params(self, params: dict): ...


@runtime_checkable
class OpticalMedium(Protocol):
    def irradiance_gain(self, offsets_um, numerical_aperture, fiber_radius_um):
        """Irradiance per unit source power from fiber-to-cell offsets `[F, N, 3]`."""

    def serialize(self) -> dict: ...

    def set_params(self, params: dict): ...


@module
class PointSource(Component):
    """Point current sources in a homogeneous conductor."""

    sigma_s_per_m: float = 0.3
    height_um: float = 50.0
    min_distance_um: float = 5.0
    stimulation_resistivity_ohm_m: float | None = 3.5

    needs_dt = False

    @property
    def recording_gain(self):
        sigma_per_mm = self.sigma_s_per_m / 1000.0
        return 1000.0 / (4.0 * _PI * sigma_per_mm * self.min_distance_um)

    def stimulation_gain(self, electrode_radius_um):
        rho = self.stimulation_resistivity_ohm_m
        if rho is None:
            rho = 1.0 / self.sigma_s_per_m
        h_eff = (electrode_radius_um**2 + self.height_um**2) ** 0.5
        return 1000.0 * rho / (4.0 * _PI * h_eff)

    def potential_gain(self, distances_um):
        return self.recording_gain * _source_shape(distances_um, self.min_distance_um)

    def field_gain(self, distances_um, electrode_radius_um):
        return self.stimulation_gain(electrode_radius_um) * _induction_shape(
            distances_um, electrode_radius_um, self.height_um
        )

    def project(
        self, gain, distances_um, membrane_currents, dt=None, output_radius_um=None
    ):
        return np.matmul(gain, membrane_currents)


@module
class DispersivePointSource(PointSource):
    """A point source whose tissue also low-passes."""

    tissue_alpha_ms_per_um: float = 0.001
    min_sigma_ms: float = static_field(0.0)
    max_sigma_ms: float = static_field(2.0)
    kernel_sigma_ratio: int = static_field(3)
    per_pair: bool = static_field(False)
    mean_distance_um: Any = None

    needs_dt = True

    def project(
        self, gain, distances_um, membrane_currents, dt=None, output_radius_um=None
    ):
        if dt is None:
            raise ValueError(
                "DispersivePointSource filters in time, pass a dt to potential_recording"
            )
        gain = np.asarray(gain)
        currents = np.asarray(membrane_currents)
        d = np.asarray(distances_um)
        if self.per_pair:
            return self._project_per_pair(gain, d, currents, dt)

        v = np.matmul(gain, currents)
        if self.mean_distance_um is not None:
            mean = np.asarray(self.mean_distance_um)
        elif output_radius_um is None:
            mean = d.mean(axis=1)
        else:
            inside = d <= output_radius_um
            counts = inside.sum(axis=1)
            sums = np.where(inside, d, 0.0).sum(axis=1)
            mean = np.where(
                counts > 0, sums / np.maximum(counts, 1), float(output_radius_um)
            )
        sigma_ms = np.maximum(self.tissue_alpha_ms_per_um * mean, self.min_sigma_ms)
        half = int(self.kernel_sigma_ratio * (self.max_sigma_ms / dt)) + 1
        x = np.arange(-half, half + 1, dtype=np.float32)

        def blur(v_ch, sigma):
            k = np.exp(-0.5 * (x / (sigma / dt + 1e-8)) ** 2)
            full = np.convolve(v_ch, k / np.sum(k), mode="full")
            return full[half : half + v_ch.shape[-1]]

        if _USES_JAX:
            return jax.vmap(blur)(v, sigma_ms)
        return np.stack([blur(a, b) for a, b in zip(v, sigma_ms, strict=True)])

    def _project_per_pair(self, gain, d, currents, dt):
        n_time = currents.shape[-1]
        n_fft = n_time + math.ceil(4.0 * self.max_sigma_ms / dt) + 1
        spectrum = np.fft.rfft(currents, n=n_fft, axis=-1)
        freqs = np.fft.rfftfreq(n_fft, d=dt)
        sigma_ms = np.clip(
            self.tissue_alpha_ms_per_um * d, self.min_sigma_ms, self.max_sigma_ms
        )

        def channel(args):
            g, s = args
            response = np.exp(-2.0 * _PI**2 * (freqs[None, :] * s[:, None]) ** 2)
            mixed = np.sum(g[:, None] * response * spectrum, axis=0)
            return np.fft.irfft(mixed, n=n_fft)[:n_time]

        if _USES_JAX:
            return jax.lax.map(channel, (gain, sigma_ms))
        return np.stack([channel((g, s)) for g, s in zip(gain, sigma_ms, strict=True)])


@module
class KubelkaMunk(Component):
    """Light spreading from a fiber by cone divergence and scattering
    (Aravanis et al. 2007):

        I(r, z) = NA^2 / (n^2 (z + z0)^2) exp(-mu_s sqrt(r^2 + z^2)) / (4 pi)

    with `z0` the depth at which the cone's area equals the fiber's.
    """

    scattering_per_mm: float = 11.2
    refractive_index: float = 1.36

    def irradiance_gain(self, offsets_um, numerical_aperture, fiber_radius_um):
        NA = numerical_aperture
        n_tissue = self.refractive_index
        mu_s = self.scattering_per_mm
        fiber_r_mm = fiber_radius_um / 1000.0

        # z0: distance from fiber tip where cone area equals fiber area
        z0 = fiber_r_mm * n_tissue / NA

        diff_mm = offsets_um / 1000.0

        r = np.sqrt(diff_mm[..., 0] ** 2 + diff_mm[..., 1] ** 2)
        z = np.abs(diff_mm[..., 2])

        dist = np.sqrt(r**2 + z**2)
        denom = np.maximum((z + z0) ** 2, 1e-12)

        return (NA**2 / (n_tissue**2 * denom)) * np.exp(-mu_s * dist) / (4 * np.pi)


class Medium(_Base):
    electrical: ElectricalMedium | None
    optical: OpticalMedium | None

    def __init__(self, electrical=_DEFAULT, optical=_DEFAULT):
        self.electrical = (
            PointSource() if electrical is _DEFAULT else resolve_component(electrical)
        )
        self.optical = (
            KubelkaMunk() if optical is _DEFAULT else resolve_component(optical)
        )

    def require(self, modality: str):
        part = getattr(self, modality)
        if part is None:
            raise ValueError(
                f"this medium has no {modality} part, which a {modality} device "
                f"needs. Use Medium({modality}=...)"
            )
        return part

    def serialize(self) -> dict:
        return {
            "cls": cls_path(self),
            "electrical": None
            if self.electrical is None
            else self.electrical.serialize(),
            "optical": None if self.optical is None else self.optical.serialize(),
        }

    def set_params(self, params: dict) -> Medium:
        nested: dict = {}
        for key, value in params.items():
            head, _, rest = key.partition("-")
            if not rest or head not in ("electrical", "optical"):
                raise KeyError(
                    f"{key!r} is not a parameter of Medium; address a part as "
                    "'electrical-<name>' or 'optical-<name>'"
                )
            if getattr(self, head) is None:
                raise KeyError(f"this medium has no {head} part to set {rest!r} on")
            nested.setdefault(head, {})[rest] = value
        return with_fields(
            self,
            **{head: getattr(self, head).set_params(v) for head, v in nested.items()},
        )

    def __repr__(self) -> str:
        return f"Medium(electrical={self.electrical!r}, optical={self.optical!r})"


__all__ = [
    "DispersivePointSource",
    "ElectricalMedium",
    "KubelkaMunk",
    "Medium",
    "OpticalMedium",
    "PointSource",
]
