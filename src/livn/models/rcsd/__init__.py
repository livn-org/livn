from __future__ import annotations

import math
import os

import numpy as _onp

from livn import types
from livn.backend import backend
from livn.types import Model
from livn.utils import P

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


def hold_potential(params: dict) -> float:
    return float(params.get("V_hold", params["V_rest"]))


class ReducedCalciumSomaDendrite(Model):
    def __init__(
        self,
        input_mode: str | None = None,
        refractory_period: float = 2.0,
        short_term_depression: bool = False,
        dendrite_offset: float = 60.0,
        dendrite_orientation: str = "random",
        dendrite_orientation_seed: int = 20260824,
        size_cv: float | dict[str, float] = 0.0,
        size_seed: int = 20260827,
    ):
        if input_mode is not None and input_mode not in {
            "current_density",
            "conductance",
            "current",
            "irradiance",
        }:
            raise ValueError(
                f"Unknown input_mode {input_mode!r}; expected one of "
                f"'current_density', 'conductance', 'current', 'irradiance'."
            )
        self.input_mode = input_mode
        if refractory_period < 0:
            raise ValueError(f"refractory_period must be >= 0, got {refractory_period}")
        self.refractory_period = float(refractory_period)
        self.short_term_depression = bool(short_term_depression)
        self.dendrite_offset = float(dendrite_offset)
        if dendrite_orientation not in {"random", "aligned"}:
            raise ValueError(
                f"Unknown dendrite_orientation {dendrite_orientation!r}; "
                "expected 'random' (each cell points its own way) or 'aligned' "
                "(every cell along +x)"
            )
        self.dendrite_orientation = dendrite_orientation
        self.dendrite_orientation_seed = int(dendrite_orientation_seed)
        self.size_cv = size_cv
        self.size_seed = int(size_seed)
        for population, cv in self._size_cv_map().items():
            if cv < 0.0:
                raise ValueError(f"size_cv for {population!r} must be >= 0, got {cv}")

    def _size_cv_map(self) -> dict[str, float]:
        if isinstance(self.size_cv, dict):
            return {k: float(v) for k, v in self.size_cv.items()}
        return {"EXC": float(self.size_cv), "INH": float(self.size_cv)}

    def size_cv_for(self, population: str) -> float:
        return self._size_cv_map().get(population, 0.0)

    def size_scales(self, population: str, gids, cv: float | None = None):
        gids = _onp.asarray(gids, dtype=_onp.int64).ravel()
        cv = self.size_cv_for(population) if cv is None else float(cv)
        if cv <= 0.0 or gids.size == 0:
            return _onp.ones(gids.size, dtype=_onp.float64)

        from statistics import NormalDist

        seed = self.size_seed + (0 if population == "EXC" else 1)
        unit = _onp.clip(P.stable_uniform(gids, seed=seed), 1e-9, 1.0 - 1e-9)
        normal = NormalDist()
        z = _onp.array([normal.inv_cdf(float(u)) for u in unit])
        sigma = math.sqrt(math.log1p(cv * cv))
        return _onp.exp(sigma * z - 0.5 * sigma * sigma)

    def _inh_params_name(self) -> str:
        return "V1In-Renshaw-InVitro"

    def prepare_stimulus(self, stimulus):
        from livn.stimulus import check_bounds

        modes = {
            "extracellular",
            "current",
            "current_density",
            "conductance",
            "irradiance",
        }
        if stimulus.input_mode not in modes:
            raise ValueError(
                f"ReducedCalciumSomaDendrite does not support input_mode "
                f"'{stimulus.input_mode}'. Supported: {modes}"
            )
        if not stimulus.deferred:
            check_bounds(
                stimulus.array,
                self.stimulus_bounds(stimulus.input_mode),
                stimulus.input_mode,
                stimulus.units,
            )
        return stimulus

    def stimulus_bounds(self, input_mode: str) -> tuple[float, float] | None:
        if input_mode == "extracellular":
            return (-1000.0, 1000.0)
        return None

    def opsin_config(self):
        return {
            "mechanism": "RhO3c",
            "sections": ["soma"],
            "wavelength_nm": 473.0,
        }

    def neuron_opsin_config(self):
        return self.opsin_config()

    def diffrax_opsin_config(self):
        return self.opsin_config()

    def stimulus_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> types.Float[types.Array, "n_stim_coords ixyz=4"]:
        """
        Where each section samples the extracellular field.

            gid, x, y, z -> gid, x + dx(gid), y + dy(gid), z

        One row per section, interleaved per neuron, so the width follows the
        population.

        Returns:
            [width*n_neurons, 4], `width` sections per neuron:
            soma0, dend0, axon0.., soma1, dend1, axon1.., ...
        """
        n_neurons = neuron_coordinates.shape[0]

        if self.dendrite_orientation == "aligned":
            dx = _onp.full(n_neurons, self.dendrite_offset)
            dy = _onp.zeros(n_neurons)
        else:
            angles = P.stable_uniform(
                _onp.asarray(neuron_coordinates)[:, 0],
                seed=self.dendrite_orientation_seed,
                high=2.0 * math.pi,
            )
            dx = self.dendrite_offset * _onp.cos(angles)
            dy = self.dendrite_offset * _onp.sin(angles)

        dend_coords = neuron_coordinates.copy()
        if _USES_JAX:
            dend_coords = dend_coords.at[:, 1].add(dx)
            dend_coords = dend_coords.at[:, 2].add(dy)
        else:
            dend_coords[:, 1] += dx
            dend_coords[:, 2] += dy

        rows = [neuron_coordinates]
        if self._has_dendrite(population):
            rows.append(dend_coords)
        unit = np.stack([dx, dy], axis=1) / self.dendrite_offset
        for offset in self._axon_offsets(population):
            link = neuron_coordinates.copy()
            step = unit * offset
            if _USES_JAX:
                link = link.at[:, 1].add(-step[:, 0]).at[:, 2].add(-step[:, 1])
            else:
                link[:, 1] -= step[:, 0]
                link[:, 2] -= step[:, 1]
            rows.append(link)

        width = len(rows)
        stacked = np.stack(rows, axis=1)  # [n, width, 4]
        return stacked.reshape(width * n_neurons, 4)

    def _has_dendrite(self, population: str | None) -> bool:
        return population != "INH"

    def _axon_offsets(self, population: str | None) -> list[float]:
        from livn.models.rcsd.neuron.templates import axon as _axon

        name = self._inh_params_name() if population == "INH" else "BoothRinzelKiehn-MN"
        return _axon.sampling_offsets(self.params(name))

    def recording_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> types.Float[types.Array, "n_stim_coords ixyz=4"]:
        coordinates = np.asarray(neuron_coordinates)
        n = coordinates.shape[0]
        rows = self.stimulus_coordinates(coordinates, population=population)
        width = len(rows) // max(n, 1)
        keep = min(2, width)
        rows = np.asarray(rows).reshape(n, width, 4)[:, :keep]
        if keep < 2:
            rows = np.concatenate([rows, rows[:, -1:]], axis=1)
        return rows.reshape(n * 2, 4)

    def expand_stimulus_currents(
        self,
        currents: types.Float[types.Array, "batch timestep n_neurons"],
    ) -> types.Float[types.Array, "batch timestep n_stimulus_coords"]:
        """Interleave [soma_curr, 0, soma_curr, 0, ...] for BRK soma-only drive."""
        zeros = np.zeros_like(currents)
        stacked = np.stack([currents, zeros], axis=-1)  # [..., n_neurons, 2]
        new_shape = (*currents.shape[:-1], currents.shape[-1] * 2)
        return stacked.reshape(new_shape)

    # neuron

    def params(self, name: str):
        base = {
            "BoothRinzelKiehn-MN": {
                "Ltotal": 120.0,
                "e_pas": -62.0,
                "pp": 0.1,
                "Ra": 190.0,
                "gc": 8.255338188577891,
                "cm_ratio": 8.192411084109766,
                "global_cm": 1.2959372997283936,
                "global_diam": 8.13117790222168,
                "soma_g_pas": 0.00014519958335660783,
                "soma_gmax_Na": 0.10499561578035355,
                "soma_gmax_K": 0.04515171924361102,
                "soma_gmax_KCa": 0.00029591686011223015,
                "soma_gmax_CaN": 0.001067815023922727,
                "soma_f_Caconc": 0.0010052554136157328,
                "soma_alpha_Caconc": 0.21016412395712364,
                "soma_kCa_Caconc": 1.5283260741327587,
                "dend_g_pas": 1.39040483614949e-05,
                "dend_gmax_CaL": 1e-05,
                "dend_gmax_CaN": 0.00028807198690005147,
                "dend_gmax_KCa": 0.00023696468003479583,
                "dend_f_Caconc": 0.0011954806290105971,
                "dend_alpha_Caconc": 0.6776084695609715,
                "dend_kCa_Caconc": 3.3313010601193564,
                "axon_segments": 5.0,
                "axon_segment_um": 30.0,
                "axon_diam": 2.5669692667650583,
                "axon_Ra": 70.0,
                "axon_cm": 0.07990137878575652,
                "axon_gmax_Na_ratio": 1.0,
                "axon_gmax_K": 0.3,
                "axon_g_pas": 5.1745027014177245e-05,
                "V_hold": -60.0,
                "V_rest": -53.0,
                "V_threshold": -37.0,
            },
            "BoothRinzelKiehn-MN-v2": {
                "Ltotal": 120.0,
                "e_pas": -62.0,
                "pp": 0.1,
                "Ra": 190.0,
                "gc": 5.977336484321611,
                "cm_ratio": 8.629566159318266,
                "global_cm": 0.8999999761581421,
                "global_diam": 5.887437343597412,
                "soma_g_pas": 7.584103831794137e-05,
                "soma_gmax_Na": 0.2678287625312805,
                "soma_gmax_K": 0.0184348523149869,
                "soma_gmax_KCa": 0.005346261924514746,
                "soma_gmax_CaN": 3.99317953058386e-05,
                "soma_f_Caconc": 0.0010460265169067887,
                "soma_alpha_Caconc": 0.1999999928705826,
                "soma_kCa_Caconc": 1.0,
                "dend_g_pas": 1e-05,
                "dend_gmax_CaL": 1.4611178597834319e-05,
                "dend_gmax_CaN": 0.00017915970839678846,
                "dend_gmax_KCa": 0.002108731761656938,
                "dend_f_Caconc": 0.00642172580623691,
                "dend_alpha_Caconc": 0.29306756876935247,
                "dend_kCa_Caconc": 5.118716103286291,
                "axon_segments": 5.0,
                "axon_segment_um": 30.0,
                "axon_diam": 2.8238298892974854,
                "axon_Ra": 70.0,
                "axon_cm": 0.13383914666265867,
                "axon_gmax_Na_ratio": 0.22069636509544605,
                "axon_gmax_K": 0.3,
                "axon_g_pas": 1.9995308811205503e-05,
                "V_hold": -60.0,
                "V_rest": -53.0,
                "V_threshold": -37.0,
            },
            "BoothRinzelKiehn-MN-v1": {
                "Ltotal": 120.0,
                "e_pas": -62.0,
                "pp": 0.1,
                "cm_ratio": 1.777068615934255,
                "dend_alpha_Caconc": 4.127872667411456,
                "dend_f_Caconc": 0.001348028244331323,
                "dend_g_pas": 4.429519140261562e-05,
                "dend_gmax_CaL": 1.3888775426905302e-05,
                "dend_gmax_CaN": 0.001,
                "dend_gmax_KCa": 0.0004750438852334796,
                "dend_kCa_Caconc": 15.900063160386765,
                "gc": 7.206964498910834,
                "global_cm": 1.0312566757202148,
                "global_diam": 7.0965752601623535,
                "soma_alpha_Caconc": 3.2002168327027687,
                "soma_f_Caconc": 0.013447929809072348,
                "soma_g_pas": 9.302226751616419e-05,
                "soma_gmax_CaN": 0.0003060516904561799,
                "soma_gmax_K": 0.06444230715646704,
                "soma_gmax_KCa": 0.001865702157561832,
                "soma_gmax_Na": 0.16667339205741882,
                "soma_kCa_Caconc": 1.0,
                "V_hold": -60.0,
                "V_rest": -57.4,
                "V_threshold": -37.0,
            },
            "PinskyRinzel-PVBC": {
                "Ltotal": 37.62028884887695,
                "cm_ratio": 3.903846025466919,
                "dend_beta_Caconc": 0.03191220387816429,
                "dend_d_Caconc": 17.446317672729492,
                "dend_g_pas": 0.0004252658982295543,
                "dend_gmax_Ca": 0.8048859238624573,
                "dend_gmax_KCa": 1.0,
                "gc": 23.3135986328125,
                "pp": 0.10000000149011612,
                "soma_g_pas": 0.0016516740433871746,
                "soma_gmax_K": 0.0010000000474974513,
                "soma_gmax_Na": 0.898166298866272,
                "e_pas": -62,
                "global_cm": 3.0,
                "global_diam": 10.0,
                "ic_constant": 0.013448839558146165,
                "V_rest": -60.0,
                "V_threshold": -37.0,
            },
            "V1In-Renshaw-Perry": {
                "global_diam": 15.670801162719727,
                "global_cm": 1.0314580202102661,
                "e_pas": -74.01708221435547,
                "soma_g_pas": 2.9999999242136255e-05,
                "soma_gmax_Na": 0.050526563078165054,
                "soma_gmax_K": 0.10332200676202774,
                "soma_gmax_Ka": 0.019093167036771774,
                "soma_gmax_KCa": 0.002753025765930022,
                "soma_gmax_CaN": 0.009249742142856121,
                "soma_f_Caconc": 0.012550535520847662,
                "soma_alpha_Caconc": 1.0873581573352384,
                "soma_kCa_Caconc": 5.057104716106808,
                "V_hold": -60.0,
                "V_rest": -50.5,
                "V_threshold": -30.8,
            },
            "V1In-Renshaw-InVitro": {
                "global_diam": 24.801532745361328,
                "global_cm": 1.0757436752319336,
                "e_pas": -69.2257080078125,
                "soma_g_pas": 0.00011933463247260079,
                "soma_gmax_Na": 0.13920024037361145,
                "soma_gmax_K": 0.14060243964195251,
                "soma_gmax_Ka": 0.0,
                "soma_gmax_KCa": 0.00033676564130083654,
                "soma_gmax_CaN": 0.007877439260482788,
                "soma_f_Caconc": 0.008864013430715868,
                "soma_alpha_Caconc": 5.0000001782354415,
                "soma_kCa_Caconc": 6.199637067638542,
                "axon_segments": 5.0,
                "axon_segment_um": 30.0,
                "axon_diam": 1.5779502391815186,
                "axon_Ra": 70.0,
                "axon_cm": 1.0,
                "axon_gmax_Na_ratio": 0.6172392336752061,
                "axon_gmax_K": 0.3,
                "axon_g_pas": 8.444644718803868e-06,
                "V_hold": -60.0,
                "V_rest": -60.0,
                "V_threshold": -50.0,
            },
        }[name]
        if self.input_mode is not None:
            base = {**base, "input_mode": self.input_mode}
        return base

    def neuron_template_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "templates")

    def neuron_mechanisms_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "mechanisms")

    def neuron_refractory_period(self) -> float:
        return self.refractory_period

    def neuron_cells(self):
        from livn.backend.neuron.cells import ReducedCell
        from livn.models.rcsd.neuron.templates.BRK import BRK
        from livn.models.rcsd.neuron.templates.V1In import V1In

        brk_params = self.params("BoothRinzelKiehn-MN")
        inh_params = self.params(self._inh_params_name())

        def scaled(params, population, gid):
            """`params` with the cell's own size factor folded into its diameter."""
            if gid is None or self.size_cv_for(population) <= 0.0:
                return params
            scale = float(self.size_scales(population, [gid])[0])
            out = dict(params)
            for key in ("global_diam", "axon_diam"):
                if params.get(key) is not None:
                    out[key] = float(params[key]) * scale
            return out

        def make_exc(morphology=None, gid=None):
            params = scaled(brk_params, "EXC", gid)
            cell = BRK({"BoothRinzelKiehn": params})
            return ReducedCell(
                cell,
                threshold=params["V_threshold"],
                v_rest=hold_potential(params),
                dend_type="dend",
            )

        def make_inh(morphology=None, gid=None):
            params = scaled(inh_params, "INH", gid)
            cell = V1In(params)
            return ReducedCell(
                cell,
                threshold=params["V_threshold"],
                v_rest=hold_potential(params),
                dend_type="dend",
            )

        return {"EXC": make_exc, "INH": make_inh}

    def neuron_celltypes(self, celltypes):
        if "EXC" in celltypes:
            celltypes["EXC"]["template class"] = (
                "livn.models.rcsd.neuron.templates.BRK.BRK"
            )
            celltypes["EXC"]["template"] = "@" + celltypes["EXC"]["template class"]
            celltypes["EXC"]["mechanism"] = {
                "BoothRinzelKiehn": self.params("BoothRinzelKiehn-MN")
            }

        if "INH" in celltypes:
            celltypes["INH"]["template class"] = (
                "livn.models.rcsd.neuron.templates.V1In.V1In"
            )
            celltypes["INH"]["template"] = "@" + celltypes["INH"]["template class"]
            celltypes["INH"]["mechanism"] = self.params(self._inh_params_name())

    def neuron_synapse_mechanisms(self):
        return {
            "AMPA": (
                "DepLinExp2Syn" if self.short_term_depression else "StdpLinExp2Syn"
            ),
            "NMDA": "StdpLinExp2SynNMDA",
            "GABA_A": "StdpLinExp2SynInh",
            "GABA_B": "LinExp2Syn",
        }

    def neuron_synapse_rules(self):
        return {
            "Exp2Syn": {
                "mech_file": "exp2syn.mod",
                "mech_params": ["tau1", "tau2", "e"],
                "netcon_params": {"weight": 0},
                "netcon_state": {},
            },
            "LinExp2Syn": {
                "mech_file": "lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "LinExp2SynNMDA": {
                "mech_file": "lin_exp2synNMDA.mod",
                "mech_params": [
                    "tau_rise",
                    "tau_decay",
                    "e",
                    "mg",
                    "Kd",
                    "gamma",
                    "vshift",
                ],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "DepLinExp2Syn": {
                # `LinExp2Syn` plus Tsodyks-Markram depression, per stream.
                # `R` (available resources) and `tlast` are per-connection
                # state carried in the NetCon weight vector, so each source
                # depresses independently despite sharing the point process.
                "mech_file": "dep_lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e", "U", "tau_rec"],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "R": 2,
                    "tlast": 3,
                },
                "netcon_state": {},
            },
            "StdpLinExp2Syn": {
                "mech_file": "stdp_lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
            "StdpLinExp2SynNMDA": {
                "mech_file": "stdp_lin_exp2synNMDA.mod",
                "mech_params": [
                    "tau_rise",
                    "tau_decay",
                    "e",
                    "mg",
                    "Kd",
                    "gamma",
                    "vshift",
                ],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
            "StdpLinExp2SynInh": {
                "mech_file": "stdp_lin_exp2syn_inh.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
        }

    def neuron_plasticity_defaults(self):
        """Default plasticity parameters by population

        Returns a nested dict ``{population_name: {param: value}}`` that will be applied
        to matching point processes when ``enable_plasticity()`` is called.

        Populations are mapped to mechanism types via ``neuron_plasticity_mechanism_groups()``

        Default values are taken from the Sigma3Exp2Syn mechanisms in neuronpp
        (https://github.com/ziemowit-s/neuronpp).
        """
        return {
            "EXC": {
                "A_ltp": 1.0,
                "A_ltd": 1.0,
                "theta_ltp": -45.0,
                "theta_ltd": -60.0,
                "ltp_sigmoid_half": -40.0,
                "ltd_sigmoid_half": -55.0,
                "learning_slope": 1.3,
                "learning_tau": 20.0,
                "w_max": 5.0,
                "w_min": 0.0001,
            },
            "INH": {
                "A_ltp": 1.0,
                "A_ltd": 1.0,
                "theta_ltp": -77.0,
                "theta_ltd": -70.0,
                "ltp_sigmoid_half": -80.0,
                "ltd_sigmoid_half": -73.0,
                "learning_slope": 1.2,
                "learning_tau": 20.0,
                "w_max": 5.0,
                "w_min": 0.0001,
            },
        }

    def neuron_plasticity_mechanism_groups(self):
        """Maps population/group names to sets of STDP mechanism class names.

        Used by ``enable_plasticity()`` to decide which parameter config
        to apply to each point process based on its mechanism type.
        """
        return {
            "EXC": {"StdpLinExp2Syn", "StdpLinExp2SynNMDA"},
            "INH": {"StdpLinExp2SynInh"},
        }

    def neuron_noise_mechanism(self, section, gid=None, index=None, seed=None):
        from neuron import h

        fluct = h.Gfluct3(section)
        if gid is not None:
            fluct.noiseFromRandom123(
                int(gid) + 1, int(index or 0) + 1, int(seed if seed is not None else 0)
            )
        return fluct, None

    def neuron_noise_configure(
        self,
        population,
        mechanism,
        state,
        std_e=0.0030,
        std_i=0.0066,
        g_e0=0.0121,
        g_i0=0.0573,
        tau_e=2.728,
        tau_i=10.49,
        E_e=0,
        E_i=-75,
    ):
        import math

        sec_name = mechanism.get_segment().sec.name()
        is_soma = "soma" in sec_name
        is_axon = "ais" in sec_name

        mechanism.tau_e = tau_e
        mechanism.tau_i = tau_i
        mechanism.E_e = E_e
        mechanism.E_i = E_i

        if is_axon:
            mechanism.std_e = 0
            mechanism.g_e0 = 0
            mechanism.std_i = 0
            mechanism.g_i0 = 0
        elif is_soma and population == "INH":
            # The V1In Renshaw INH cell is single-compartment: its soma is the only
            # site, so it must carry BOTH the excitatory and inhibitory background.
            # With the soma-only inhibitory split below it would be pinned near E_i
            # (-75 mV) and never fire.
            mechanism.std_e = std_e
            mechanism.g_e0 = g_e0
            mechanism.std_i = std_i
            mechanism.g_i0 = g_i0
        elif is_soma:
            # two-compartment EXC: inhibition on the soma
            mechanism.std_e = 0
            mechanism.g_e0 = 0
            mechanism.std_i = std_i
            mechanism.g_i0 = g_i0
        else:
            # two-compartment EXC: excitation on the dendrite
            mechanism.std_e = std_e
            mechanism.g_e0 = g_e0
            mechanism.std_i = 0
            mechanism.g_i0 = 0

        mechanism.on = 1 if (mechanism.std_e > 0 or mechanism.std_i > 0) else 0

        # recompute INITIAL variables manually to ensure changes propagate mid-simulation
        h_val = mechanism.h
        if mechanism.tau_e > 0:
            mechanism.D_e = 2 * mechanism.std_e**2 / mechanism.tau_e
            mechanism.exp_e = math.exp(-h_val / mechanism.tau_e)
            mechanism.amp_e = mechanism.std_e * math.sqrt(
                max(0.0, 1.0 - math.exp(-2 * h_val / mechanism.tau_e))
            )
        else:
            mechanism.D_e = 0.0
            mechanism.exp_e = 0.0
            mechanism.amp_e = 0.0

        if mechanism.tau_i > 0:
            mechanism.D_i = 2 * mechanism.std_i**2 / mechanism.tau_i
            mechanism.exp_i = math.exp(-h_val / mechanism.tau_i)
            mechanism.amp_i = mechanism.std_i * math.sqrt(
                max(0.0, 1.0 - math.exp(-2 * h_val / mechanism.tau_i))
            )
        else:
            mechanism.D_i = 0.0
            mechanism.exp_i = 0.0
            mechanism.amp_i = 0.0

    def neuron_default_noise(self, system: str):
        return {
            "EI1": {
                "g_e0": 1.0,
                "g_i0": 1.2172681093215942,
                "std_e": 0.3290764391422272,
                "std_i": 0.35633188486099243,
                "tau_e": 33.00786209106445,
                "tau_i": 28.50772476196289,
            },
            "EI2": {
                "g_e0": 3.409418821334839,
                "g_i0": 1.0573457479476929,
                "std_e": 0.49486637115478516,
                "std_i": 0.23988725244998932,
                "tau_e": 31.219661712646484,
                "tau_i": 16.700607299804688,
            },
            "EI3": {},
            "EI4": {},
        }[system]

    def neuron_default_weights(self, system: str):
        return {
            "EI1": {
                "EXC_EXC-dend-AMPA-weight": 0.0010000000254350994,
                "EXC_EXC-dend-NMDA-weight": 0.37764625228307414,
                "INH_EXC-dend-AMPA-weight": 2.9091933347646908,
                "INH_EXC-dend-NMDA-weight": 0.0010000000254350994,
                "EXC_INH-soma-GABA_A-weight": 9.406616405134113,
                "INH_INH-soma-GABA_A-weight": 8.710510071227473,
            },
            "EI2": {
                "EXC_EXC-dend-AMPA-weight": 0.8598201979147386,
                "EXC_EXC-dend-NMDA-weight": 1.2337499089211241,
                "INH_EXC-dend-AMPA-weight": 1.1851855878120792,
                "INH_EXC-dend-NMDA-weight": 0.056837208512839466,
                "EXC_INH-soma-GABA_A-weight": 1.5785464331652075,
                "INH_INH-soma-GABA_A-weight": 4.262910407764182,
            },
            "EI3": {},
            "EI4": {},
        }[system]

    # diffrax

    def diffrax_module(self, env, key):
        from livn.models.rcsd.diffrax.culture import MotoneuronCulture

        return MotoneuronCulture(
            num_neurons=len(env.simulated_gids(everywhere=True)),
            params=self.params("BoothRinzelKiehn-MN"),
            key=key,
        )

    # brian2

    def _brian2_synapse_block(self, suffix: str, v: str, area_cm2: float) -> str:
        s = suffix
        # synaptic conductance in uS, voltage in mV: g * V is nA -> mA/cm2
        syn_factor = 1e-6 / area_cm2

        return f"""
        # --- Synaptic conductances (dual-exponential, in uS) ---
        dA_ampa{s}/dt = -A_ampa{s} / (tau_rise_ampa{s} * ms) : 1
        dB_ampa{s}/dt = -B_ampa{s} / (tau_decay_ampa{s} * ms) : 1
        dA_nmda{s}/dt = -A_nmda{s} / (tau_rise_nmda{s} * ms) : 1
        dB_nmda{s}/dt = -B_nmda{s} / (tau_decay_nmda{s} * ms) : 1
        dA_gaba_a{s}/dt = -A_gaba_a{s} / (tau_rise_gaba_a{s} * ms) : 1
        dB_gaba_a{s}/dt = -B_gaba_a{s} / (tau_decay_gaba_a{s} * ms) : 1
        dA_gaba_b{s}/dt = -A_gaba_b{s} / (tau_rise_gaba_b{s} * ms) : 1
        dB_gaba_b{s}/dt = -B_gaba_b{s} / (tau_decay_gaba_b{s} * ms) : 1

        g_ampa{s} = B_ampa{s} - A_ampa{s} : 1
        g_nmda{s} = B_nmda{s} - A_nmda{s} : 1
        g_gaba_a{s} = B_gaba_a{s} - A_gaba_a{s} : 1
        g_gaba_b{s} = B_gaba_b{s} - A_gaba_b{s} : 1

        # NMDA Mg2+ block (Jahr & Stevens)
        mgblock{s} = 1.0 / (1.0 + exp(nmda_gamma{s} * -({v} + nmda_vshift{s})) * (nmda_mg{s} / nmda_Kd{s})) : 1
        nmda_mg{s} : 1
        nmda_Kd{s} : 1
        nmda_gamma{s} : 1
        nmda_vshift{s} : 1

        # Synaptic time constants (set per projection during init)
        tau_rise_ampa{s} : 1
        tau_decay_ampa{s} : 1
        tau_rise_nmda{s} : 1
        tau_decay_nmda{s} : 1
        tau_rise_gaba_a{s} : 1
        tau_decay_gaba_a{s} : 1
        tau_rise_gaba_b{s} : 1
        tau_decay_gaba_b{s} : 1

        # Synaptic reversal potentials
        e_ampa{s} : 1
        e_nmda{s} : 1
        e_gaba_a{s} : 1
        e_gaba_b{s} : 1

        # Synaptic current (mA/cm2, sign: negative = inward = depolarizing for exc)
        I_syn_ampa{s} = g_ampa{s} * ({v} - e_ampa{s}) * {syn_factor} : 1
        I_syn_nmda{s} = g_nmda{s} * mgblock{s} * ({v} - e_nmda{s}) * {syn_factor} : 1
        I_syn_gaba_a{s} = g_gaba_a{s} * ({v} - e_gaba_a{s}) * {syn_factor} : 1
        I_syn_gaba_b{s} = g_gaba_b{s} * ({v} - e_gaba_b{s}) * {syn_factor} : 1
        I_syn{s} = -(I_syn_ampa{s} + I_syn_nmda{s} + I_syn_gaba_a{s} + I_syn_gaba_b{s}) : 1

        # --- STDP learning signal (excitatory rule, shared across connections) ---
        exc_ltd{s} = int({v} > theta_ltd_exc) * (1.0 / (1.0 + exp(clip(-({v} - half_ltd_exc) * log(slope_exc), -500, 500)))) : 1
        exc_ltp{s} = int({v} > theta_ltp_exc) * (1.0 / (1.0 + exp(clip(-({v} - half_ltp_exc) * log(slope_exc), -500, 500)))) : 1
        sig_sat_exc{s} = 2.0 / (1.0 + exp(clip(-(-A_ltd_exc * exc_ltd{s} + A_ltp_exc * 2.0 * exc_ltp{s}) / (learning_tau_exc + 1e-20) * log(slope_exc), -500, 500))) - 1.0 : 1
        dlearning_w_exc{s}/dt = -learning_w_exc{s} / (4.0 * ms) + plasticity_on * sig_sat_exc{s} / (125.0 * ms) : 1
        dlearn_int_exc{s}/dt = learning_w_exc{s} / ms : 1

        # --- STDP learning signal (inhibitory rule, shared across connections) ---
        inh_ltd{s} = int({v} < theta_ltd_inh) * (1.0 / (1.0 + exp(clip(-({v} - half_ltd_inh) * log(slope_inh), -500, 500)))) : 1
        inh_ltp{s} = int({v} < theta_ltp_inh) * (1.0 / (1.0 + exp(clip(-({v} - half_ltp_inh) * log(slope_inh), -500, 500)))) : 1
        sig_sat_inh{s} = 2.0 / (1.0 + exp(clip(-(-A_ltd_inh * inh_ltd{s} + A_ltp_inh * 2.0 * inh_ltp{s}) / (learning_tau_inh + 1e-20) * log(slope_inh), -500, 500))) - 1.0 : 1
        dlearning_w_inh{s}/dt = -learning_w_inh{s} / ms + plasticity_on * sig_sat_inh{s} / (125.0 * ms) : 1
        dlearn_int_inh{s}/dt = learning_w_inh{s} / ms : 1
"""

    def _brian2_stdp_parameters(self) -> str:
        """STDP configuration, shared by every compartment of a cell."""
        return """
        # STDP parameters (set by enable_plasticity)
        plasticity_on : 1
        theta_ltp_exc : 1
        theta_ltd_exc : 1
        half_ltp_exc : 1
        half_ltd_exc : 1
        slope_exc : 1
        A_ltp_exc : 1
        A_ltd_exc : 1
        learning_tau_exc : 1
        theta_ltp_inh : 1
        theta_ltd_inh : 1
        half_ltp_inh : 1
        half_ltd_inh : 1
        slope_inh : 1
        A_ltp_inh : 1
        A_ltd_inh : 1
        learning_tau_inh : 1
"""

    def _brk_equations(self, params):
        p = params

        # Geometry: coupling conductance
        # In NEURON: Ra is set so that the coupling between soma and dend gives gc
        # gc is in mS/cm2, we convert to brian2 units in the equations
        # gc_coupling = gc / (p * (1-p)) between compartments
        pp = p["pp"]
        Ltotal = p["Ltotal"]
        gc = p["gc"]
        cm = p["global_cm"]
        cm_ratio = p["cm_ratio"]
        diam = p["global_diam"]
        e_pas = p["e_pas"]

        # Compartment areas (um^2) and lengths
        L_soma = pp * Ltotal  # um
        L_dend = (1 - pp) * Ltotal  # um
        area_soma = math.pi * diam * L_soma  # um^2
        area_dend = math.pi * diam * L_dend  # um^2

        # Axial resistance: same formula as in BRK.py biophys()
        # Ra is set so coupling effective conductance = gc (mS/cm2)
        # The coupling current: I_coupling = gc/(p*(1-p)) * (V_other - V_self) in mS/cm2 * mV = uA/cm2
        # Convert to nA: I_nA = I_mA_per_cm2 * area_cm2 * 1000
        # But in brian2 we work with mV directly in the ODE
        # dV/dt = ... / (cm * area) where cm is uF/cm2
        # The voltage equation: C * dV/dt = -I_ionic + I_coupling + I_stim (all in mA/cm2 or equivalent)
        # Since brian2 works with units, we express everything in mV and ms

        # KTF factor for GHK (CaN uses 36/293.15, CaL uses 25/293.15)
        celsius = 6.3
        fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
        fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
        cao = 2.0  # mM external calcium

        # Soma-dendrite coupling.
        #
        # In the ODE: 1/C * I [mA/cm2] gives mV/ms, since
        # mA/cm2 / (uF/cm2) = mA/uF = 1000 V/s = mV/ms.
        area_soma_cm2 = area_soma * 1e-8
        area_dend_cm2 = area_dend * 1e-8
        # `1/(2*ri)` is what BRK.py sets, and NEURON's `ri(0.5)` is already
        # the section's own half-segment resistance -- so the half-section
        # conductance is `1/ri`, twice that. Halving it here would double the
        # separation between the compartments under load.
        g_half_s = 2.0 * (gc / pp) * area_soma_cm2 * 1e3  # uS
        g_half_d = 2.0 * (gc / pp) * area_dend_cm2 * 1e3  # uS
        g_coup = 1.0 / (1.0 / g_half_s + 1.0 / g_half_d)  # uS
        gc_soma = g_coup * 1e-6 / area_soma_cm2  # S/cm2
        gc_dend = g_coup * 1e-6 / area_dend_cm2  # S/cm2

        ic_constant = p.get("ic_constant", 0.0)
        ic_constant_d = p.get("ic_constant_d", 0.0)

        synapse_blocks = self._brian2_synapse_block(
            "", "Vs", area_soma_cm2
        ) + self._brian2_synapse_block("_d", "Vd", area_dend_cm2)
        stdp_parameters = self._brian2_stdp_parameters()

        return f"""
        # Soma voltage
        dVs/dt = (1000.0/{cm * cm_ratio}) * (-I_Na - I_K - I_KCa_s - I_CaN_s - I_leak_s - I_coup_s + I_noise_s + I_stim_s + I_ext + I_syn + {ic_constant}) / ms : 1
        # Dendrite voltage
        dVd/dt = (1000.0/{cm}) * (-I_KCa_d - I_CaN_d - I_CaL_d - I_leak_d - I_coup_d + I_noise_d + I_stim_d + I_syn_d + {ic_constant_d}) / ms : 1

        # --- Soma currents (mA/cm2) ---
        I_Na = {p["soma_gmax_Na"]} * m_inf_s**3 * h_s * (Vs - E_Na) : 1
        I_K = {p["soma_gmax_K"]} * n_s**4 * (Vs - E_K) : 1
        I_KCa_s = {p["soma_gmax_KCa"]} * (Ca_s / (Ca_s + 0.0005)) * (Vs - E_K) : 1
        I_CaN_s = {p["soma_gmax_CaN"]} * mnS**2 * hnS * ghk_s : 1
        I_leak_s = {p["soma_g_pas"]} * (Vs - ({e_pas})) : 1

        # --- Dendrite currents (mA/cm2) ---
        I_KCa_d = {p["dend_gmax_KCa"]} * (Ca_d / (Ca_d + 0.0005)) * (Vd - E_K) : 1
        I_CaN_d = {p["dend_gmax_CaN"]} * mnD**2 * hnD * ghk_d_N : 1
        I_CaL_d = {p["dend_gmax_CaL"]} * ml_d * ghk_d_L : 1
        I_leak_d = {p["dend_g_pas"]} * (Vd - ({e_pas})) : 1

        # --- Coupling (mA/cm2) ---
        I_coup_s = {gc_soma} * (Vs - Vd) : 1
        I_coup_d = {gc_dend} * (Vd - Vs) : 1

        # --- GHK driving force for calcium ---
        ghk_s = -({fN}) * (1.0 - (Ca_s / {cao}) * exp(Vs / {fN})) * efun_s : 1
        efun_s = int(abs(Vs / {fN}) < 1e-4) * (1.0 - Vs / {fN} / 2.0) + int(abs(Vs / {fN}) >= 1e-4) * ((Vs / {fN}) / (exp(Vs / {fN}) - 1.0 + 1e-20)) : 1

        ghk_d_N = -({fN}) * (1.0 - (Ca_d / {cao}) * exp(Vd / {fN})) * efun_d_N : 1
        efun_d_N = int(abs(Vd / {fN}) < 1e-4) * (1.0 - Vd / {fN} / 2.0) + int(abs(Vd / {fN}) >= 1e-4) * ((Vd / {fN}) / (exp(Vd / {fN}) - 1.0 + 1e-20)) : 1

        ghk_d_L = -({fL}) * (1.0 - (Ca_d / {cao}) * exp(Vd / {fL})) * efun_d_L : 1
        efun_d_L = int(abs(Vd / {fL}) < 1e-4) * (1.0 - Vd / {fL} / 2.0) + int(abs(Vd / {fL}) >= 1e-4) * ((Vd / {fL}) / (exp(Vd / {fL}) - 1.0 + 1e-20)) : 1

        # --- Gating variables ---
        m_inf_s = 1.0 / (1.0 + exp(-(Vs + 35.0) / 7.8)) : 1
        dh_s/dt = (1.0 / (1.0 + exp((Vs + 55.0) / 7.0)) - h_s) / (30.0 / (exp((Vs + 50.0) / 15.0) + exp(-(Vs + 50.0) / 16.0))) / ms : 1
        dn_s/dt = (1.0 / (1.0 + exp(-(Vs + 28.0) / 15.0)) - n_s) / (7.0 / (exp((Vs + 40.0) / 40.0) + exp(-(Vs + 40.0) / 50.0))) / ms : 1

        # CaN gating (soma)
        dmnS/dt = (1.0 / (1.0 + exp((Vs + 30.0) / (-5.0))) - mnS) / (4.0 * ms) : 1
        dhnS/dt = (1.0 / (1.0 + exp((Vs + 45.0) / 5.0)) - hnS) / (40.0 * ms) : 1

        # CaN gating (dendrite)
        dmnD/dt = (1.0 / (1.0 + exp((Vd + 30.0) / (-5.0))) - mnD) / (4.0 * ms) : 1
        dhnD/dt = (1.0 / (1.0 + exp((Vd + 45.0) / 5.0)) - hnD) / (40.0 * ms) : 1

        # CaL gating (dendrite)
        dml_d/dt = (1.0 / (1.0 + exp((Vd + 40.0) / (-7.0))) - ml_d) / (60.0 * ms) : 1

        # --- Calcium dynamics ---
        dCa_s/dt = {p["soma_f_Caconc"]} * (clip(-{p["soma_alpha_Caconc"]} * (I_CaN_s - I_CaN_s_rest), 0, inf) - {p["soma_kCa_Caconc"]} * (Ca_s - 1e-5)) / ms : 1
        dCa_d/dt = {p["dend_f_Caconc"]} * (clip(-{p["dend_alpha_Caconc"]} * ((I_CaN_d + I_CaL_d) - I_Ca_d_rest), 0, inf) - {p["dend_kCa_Caconc"]} * (Ca_d - 1e-5)) / ms : 1

        I_CaN_s_rest : 1
        I_Ca_d_rest : 1

        # --- Reversal potentials ---
        E_Na : 1
        E_K : 1

        stim_index : integer (constant)

        # --- Extracellular stimulus as current density (mA/cm2) ---
        # V_ext enters through passive conductance: I = g_pas * V_ext
        # (linearized approximation of NEURON extracellular mechanism)
        V_ext = stim_v(t, stim_index) / mV : 1
        I_stim_s = {p["soma_g_pas"]} * V_ext + stim_i(t, stim_index)/amp * {1000.0 / area_soma_cm2} : 1
        I_stim_d = {p["dend_g_pas"]} * V_ext : 1

{synapse_blocks}{stdp_parameters}
        # --- Noise as current density (mA/cm2) ---
        # Split by compartment exactly as Gfluct3 is configured on this cell in
        # the NEURON backend: inhibitory background on the soma, excitatory on
        # the dendrite. Gfluct3 clips the conductance it uses at zero, so the
        # same clip is applied here -- without it a large std_e/g_e0 ratio
        # gives brian2 a lower effective mean conductance than NEURON.
        I_noise_s = (clip(g_noise_i, 0, inf) * (Vs - (-75))) * (-1e-6 / {area_soma_cm2}) : 1
        I_noise_d = (clip(g_noise_e, 0, inf) * (Vd - 0)) * (-1e-6 / {area_dend_cm2}) : 1

        # --- Noise conductances (updated externally via run_regularly) ---
        g_noise_e : 1
        g_noise_i : 1
        g_e0 : 1
        g_i0 : 1
        tau_e : 1
        tau_i : 1
        amp_e : 1
        amp_i : 1

        # --- External current for optogenetic injection ---
        I : amp
        I_ext = I/amp * {1000.0 / (area_soma * 1e-8)} : 1
        noise_amplitude : 1

        # --- Total membrane current per compartment (mA/cm2, positive outward) ---
        I_memb_s = I_Na + I_K + I_KCa_s + I_CaN_s + I_leak_s + I_coup_s : 1
        I_memb_d = I_KCa_d + I_CaN_d + I_CaL_d + I_leak_d + I_coup_d : 1

        # v alias for voltage monitoring (Vs is dimensionless in mV, convert to volt)
        v = Vs * mV : volt
        """

    def _v1in_equations(self, params):
        """Single-compartment V1 Renshaw cell, mirroring ``templates/V1In.py``.

        Same channel set as the NEURON template -- Nas, Kdr, Ka_v1in, KCa, CaN
        over a first-order calcium pool -- on a sphere-equivalent soma
        (``L = diam``). There is no dendrite, so the coupling, the second
        stimulus site and the dendritic recording slot all drop out.
        """
        p = params
        cm = p["global_cm"]
        diam = p["global_diam"]
        e_pas = p["e_pas"]

        # V1In.geometry(): L = diam, one segment
        area_soma = math.pi * diam * diam  # um^2

        celsius = 6.3
        fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
        cao = 2.0  # mM external calcium

        ic_constant = p.get("ic_constant", 0.0)

        synapse_blocks = self._brian2_synapse_block("", "Vs", area_soma * 1e-8)
        stdp_parameters = self._brian2_stdp_parameters()

        return f"""
        # Soma voltage (the only compartment)
        dVs/dt = (1000.0/{cm}) * (-I_Na - I_K - I_Ka - I_KCa_s - I_CaN_s - I_leak_s + I_noise_s + I_stim_s + I_ext + I_syn + {ic_constant}) / ms : 1

        # --- Soma currents (mA/cm2) ---
        I_Na = {p["soma_gmax_Na"]} * m_inf_s**3 * h_s * (Vs - E_Na) : 1
        I_K = {p["soma_gmax_K"]} * n_s**4 * (Vs - E_K) : 1
        I_Ka = {p["soma_gmax_Ka"]} * a_k * b_k * (Vs - E_K) : 1
        I_KCa_s = {p["soma_gmax_KCa"]} * (Ca_s / (Ca_s + 0.0005)) * (Vs - E_K) : 1
        I_CaN_s = {p["soma_gmax_CaN"]} * mnS**2 * hnS * ghk_s : 1
        I_leak_s = {p["soma_g_pas"]} * (Vs - ({e_pas})) : 1

        # --- GHK driving force for calcium ---
        ghk_s = -({fN}) * (1.0 - (Ca_s / {cao}) * exp(Vs / {fN})) * efun_s : 1
        efun_s = int(abs(Vs / {fN}) < 1e-4) * (1.0 - Vs / {fN} / 2.0) + int(abs(Vs / {fN}) >= 1e-4) * ((Vs / {fN}) / (exp(Vs / {fN}) - 1.0 + 1e-20)) : 1

        # --- Gating variables (Nas, Kdr; identical kinetics to the motoneuron) ---
        m_inf_s = 1.0 / (1.0 + exp(-(Vs + 35.0) / 7.8)) : 1
        dh_s/dt = (1.0 / (1.0 + exp((Vs + 55.0) / 7.0)) - h_s) / (30.0 / (exp((Vs + 50.0) / 15.0) + exp(-(Vs + 50.0) / 16.0))) / ms : 1
        dn_s/dt = (1.0 / (1.0 + exp(-(Vs + 28.0) / 15.0)) - n_s) / (7.0 / (exp((Vs + 40.0) / 40.0) + exp(-(Vs + 40.0) / 50.0))) / ms : 1

        # --- A-type potassium (Ka_v1in): fast activation, slow inactivation ---
        da_k/dt = (1.0 / (1.0 + exp(-(Vs + 36.0) / 8.0)) - a_k) / (1.0 * ms) : 1
        db_k/dt = (1.0 / (1.0 + exp((Vs + 66.0) / 8.0)) - b_k) / (15.0 * ms) : 1

        # CaN gating
        dmnS/dt = (1.0 / (1.0 + exp((Vs + 30.0) / (-5.0))) - mnS) / (4.0 * ms) : 1
        dhnS/dt = (1.0 / (1.0 + exp((Vs + 45.0) / 5.0)) - hnS) / (40.0 * ms) : 1

        # --- Calcium dynamics ---
        dCa_s/dt = {p["soma_f_Caconc"]} * (clip(-{p["soma_alpha_Caconc"]} * (I_CaN_s - I_CaN_s_rest), 0, inf) - {p["soma_kCa_Caconc"]} * (Ca_s - 1e-5)) / ms : 1

        I_CaN_s_rest : 1

        # --- Reversal potentials ---
        E_Na : 1
        E_K : 1

        stim_index : integer (constant)

        # --- Extracellular stimulus as current density (mA/cm2) ---
        V_ext = stim_v(t, stim_index) / mV : 1
        I_stim_s = {p["soma_g_pas"]} * V_ext + stim_i(t, stim_index)/amp * {1000.0 / (area_soma * 1e-8)} : 1

{synapse_blocks}{stdp_parameters}
        # --- Noise as current density (mA/cm2) ---
        # The soma is the only site, so unlike the motoneuron it carries both
        # the excitatory and the inhibitory background -- as Gfluct3 does on
        # this cell in the NEURON backend, clip at zero included
        I_noise_s = (clip(g_noise_e, 0, inf) * (Vs - 0) + clip(g_noise_i, 0, inf) * (Vs - (-75))) * (-1e-6 / {area_soma * 1e-8}) : 1

        # --- Noise conductances (updated externally via run_regularly) ---
        g_noise_e : 1
        g_noise_i : 1
        g_e0 : 1
        g_i0 : 1
        tau_e : 1
        tau_i : 1
        amp_e : 1
        amp_i : 1

        # --- External current for optogenetic injection ---
        I : amp
        I_ext = I/amp * {1000.0 / (area_soma * 1e-8)} : 1
        noise_amplitude : 1

        # --- Total membrane current (mA/cm2, positive outward) ---
        I_memb_s = I_Na + I_K + I_Ka + I_KCa_s + I_CaN_s + I_leak_s : 1
        # No dendrite. The slot is kept so the per-cell recording layout stays
        # (soma, dend) as it is for the motoneuron and in the NEURON backend,
        # where the compartment this cell does not have reads back as zeros.
        I_memb_d = 0.0 * I_memb_s : 1

        # v alias for voltage monitoring (Vs is dimensionless in mV, convert to volt)
        v = Vs * mV : volt
        """

    def _brian2_refractory(self, v_threshold: float) -> str:
        return (
            f"(Vs > {v_threshold}) or (t - lastspike < {self.refractory_period} * ms)"
        )

    def brian2_population_group(
        self, population_name, n, offset, coordinates, prng, rows=None
    ):
        import math as _m

        import brian2 as b2

        if population_name == "EXC":
            p = self.params("BoothRinzelKiehn-MN")

            # Compute ic_constant dynamically from equilibrium condition
            v_rest = hold_potential(p)
            e_pas = p["e_pas"]
            celsius = 6.3
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                ef = 1.0 - nu / 2.0 if abs(nu) < 0.0001 else nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            # Gating at rest
            m_inf = 1.0 / (1.0 + _m.exp(-(v_rest + 35.0) / 7.8))
            h_rest = 1.0 / (1.0 + _m.exp((v_rest + 55.0) / 7.0))
            n_rest = 1.0 / (1.0 + _m.exp(-(v_rest + 28.0) / 15.0))
            mnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            hnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))

            ghk_s = _ghk(v_rest, cai0, cao, fN)

            # Soma currents at rest
            I_Na = p["soma_gmax_Na"] * m_inf**3 * h_rest * (v_rest - 50.0)
            I_K = p["soma_gmax_K"] * n_rest**4 * (v_rest - (-77.0))
            I_KCa = p["soma_gmax_KCa"] * (cai0 / (cai0 + 0.0005)) * (v_rest - (-77.0))
            I_CaN = p["soma_gmax_CaN"] * mnS_rest**2 * hnS_rest * ghk_s
            I_leak = p["soma_g_pas"] * (v_rest - e_pas)
            # ic_constant balances soma at rest (coupling=0, input=0)
            p = dict(p)  # copy so we can override
            p["ic_constant"] = I_Na + I_K + I_KCa + I_CaN + I_leak

            # Dendrite currents at rest
            ghk_d_N = _ghk(v_rest, cai0, cao, fN)
            ghk_d_L = _ghk(v_rest, cai0, cao, fL)
            mnD_rest = mnS_rest
            hnD_rest = hnS_rest
            ml_rest = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            I_KCa_d = p["dend_gmax_KCa"] * (cai0 / (cai0 + 0.0005)) * (v_rest - (-77.0))
            I_CaN_d = p["dend_gmax_CaN"] * mnD_rest**2 * hnD_rest * ghk_d_N
            I_CaL_d = p["dend_gmax_CaL"] * ml_rest * ghk_d_L
            I_leak_d = p["dend_g_pas"] * (v_rest - e_pas)
            p["ic_constant_d"] = I_KCa_d + I_CaN_d + I_CaL_d + I_leak_d

            equations = self._brk_equations(p)

            _use_gsl = os.environ.get("LIVN_USE_LIBGSL", "0") == "1"
            _method = "gsl_rkf45" if _use_gsl else "euler"
            _dt = 0.025 if _use_gsl else 0.005

            population = b2.NeuronGroup(
                n,
                equations,
                threshold="Vs > {:f}".format(p["V_threshold"]),
                reset="",  # no artificial reset for biophysical model
                refractory=self._brian2_refractory(p["V_threshold"]),
                method=_method,
                name=population_name,
                dt=_dt * b2.ms,
            )

            # Initial conditions
            v_rest = hold_potential(p)
            population.Vs = v_rest
            population.Vd = v_rest

            population.h_s = 1.0 / (1.0 + _m.exp((v_rest + 55.0) / 7.0))
            population.n_s = 1.0 / (1.0 + _m.exp(-(v_rest + 28.0) / 15.0))
            population.mnS = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            population.hnS = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            population.mnD = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            population.hnD = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            population.ml_d = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            population.Ca_s = 1e-5
            population.Ca_d = 1e-5

            # Compute resting Ca currents for Ca_conc dynamics
            celsius = 6.3
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                ef = 1.0 - nu / 2.0 if abs(nu) < 0.0001 else nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            ghk_s = _ghk(v_rest, cai0, cao, fN)
            ghk_d_N = _ghk(v_rest, cai0, cao, fN)
            ghk_d_L = _ghk(v_rest, cai0, cao, fL)

            mnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            hnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            I_CaN_s_rest = p["soma_gmax_CaN"] * mnS_rest**2 * hnS_rest * ghk_s
            population.I_CaN_s_rest = I_CaN_s_rest

            mnD_rest = mnS_rest
            hnD_rest = hnS_rest
            ml_rest = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            I_Ca_d_rest = (
                p["dend_gmax_CaN"] * mnD_rest**2 * hnD_rest * ghk_d_N
                + p["dend_gmax_CaL"] * ml_rest * ghk_d_L
            )
            population.I_Ca_d_rest = I_Ca_d_rest

            # Reversal potentials
            population.E_Na = 50.0
            population.E_K = -77.0

            _diam = p["global_diam"]
            _Ltot = p["Ltotal"]
            _pp = p["pp"]
            population.add_attribute("area_soma_cm2")
            population.add_attribute("area_dend_cm2")
            population.area_soma_cm2 = _m.pi * _diam * (_pp * _Ltot) * 1e-8
            population.area_dend_cm2 = _m.pi * _diam * ((1 - _pp) * _Ltot) * 1e-8

        else:
            p = self.params(self._inh_params_name())

            v_rest = hold_potential(p)
            e_pas = p["e_pas"]
            celsius = 6.3
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                ef = 1.0 - nu / 2.0 if abs(nu) < 0.0001 else nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            # Gating at rest
            m_inf = 1.0 / (1.0 + _m.exp(-(v_rest + 35.0) / 7.8))
            h_rest = 1.0 / (1.0 + _m.exp((v_rest + 55.0) / 7.0))
            n_rest = 1.0 / (1.0 + _m.exp(-(v_rest + 28.0) / 15.0))
            a_rest = 1.0 / (1.0 + _m.exp(-(v_rest + 36.0) / 8.0))
            b_rest = 1.0 / (1.0 + _m.exp((v_rest + 66.0) / 8.0))
            mn_rest = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            hn_rest = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))

            ghk_rest = _ghk(v_rest, cai0, cao, fN)

            # Soma currents at rest
            I_Na = p["soma_gmax_Na"] * m_inf**3 * h_rest * (v_rest - 50.0)
            I_K = p["soma_gmax_K"] * n_rest**4 * (v_rest - (-77.0))
            I_Ka = p["soma_gmax_Ka"] * a_rest * b_rest * (v_rest - (-77.0))
            I_KCa = p["soma_gmax_KCa"] * (cai0 / (cai0 + 0.0005)) * (v_rest - (-77.0))
            I_CaN = p["soma_gmax_CaN"] * mn_rest**2 * hn_rest * ghk_rest
            I_leak = p["soma_g_pas"] * (v_rest - e_pas)
            # what V1In.init_ic pins into `constant.mod` in NEURON
            p = dict(p)  # copy so we can override
            p["ic_constant"] = I_Na + I_K + I_Ka + I_KCa + I_CaN + I_leak

            equations = self._v1in_equations(p)

            _use_gsl = os.environ.get("LIVN_USE_LIBGSL", "0") == "1"
            _method = "gsl_rkf45" if _use_gsl else "euler"
            _dt = 0.025 if _use_gsl else 0.005

            population = b2.NeuronGroup(
                n,
                equations,
                threshold="Vs > {:f}".format(p["V_threshold"]),
                reset="",  # no artificial reset for biophysical model
                refractory=self._brian2_refractory(p["V_threshold"]),
                method=_method,
                name=population_name,
                dt=_dt * b2.ms,
            )

            # Initial conditions
            population.Vs = v_rest
            population.h_s = h_rest
            population.n_s = n_rest
            population.a_k = a_rest
            population.b_k = b_rest
            population.mnS = mn_rest
            population.hnS = hn_rest
            population.Ca_s = cai0
            population.I_CaN_s_rest = I_CaN

            population.E_Na = 50.0
            population.E_K = -77.0

            _diam = p["global_diam"]
            population.add_attribute("area_soma_cm2")
            population.add_attribute("area_dend_cm2")
            # sphere-equivalent soma, L = diam (V1In.geometry)
            population.area_soma_cm2 = _m.pi * _diam * _diam * 1e-8
            # No dendrite: the slot only keeps the recording layout uniform,
            # and I_memb_d is identically zero, so the area is never load
            # bearing -- it is non-zero because the backend requires it.
            population.area_dend_cm2 = population.area_soma_cm2

        # Common: noise init
        population.g_noise_e = 0.0
        population.g_noise_i = 0.0
        population.g_e0 = 0.0
        population.g_i0 = 0.0
        population.tau_e = 2.728
        population.tau_i = 10.49
        population.amp_e = 0.0
        population.amp_i = 0.0
        population.I = 0 * b2.amp
        population.noise_amplitude = 0.0

        # Synaptic and STDP state, per compartment the cell actually has
        for suffix in ("", "_d"):
            if f"A_ampa{suffix}" not in population.variables:
                continue

            # Conductances
            for mech in ("ampa", "nmda", "gaba_a", "gaba_b"):
                setattr(population, f"A_{mech}{suffix}", 0.0)
                setattr(population, f"B_{mech}{suffix}", 0.0)

            # Default time constants and reversal potentials, overridden per
            # projection by the backend out of the graph config
            setattr(population, f"tau_rise_ampa{suffix}", 0.5)
            setattr(population, f"tau_decay_ampa{suffix}", 3.0)
            setattr(population, f"tau_rise_nmda{suffix}", 10.0)
            setattr(population, f"tau_decay_nmda{suffix}", 35.0)
            setattr(population, f"tau_rise_gaba_a{suffix}", 0.3)
            setattr(population, f"tau_decay_gaba_a{suffix}", 6.0)
            setattr(population, f"tau_rise_gaba_b{suffix}", 1.0)
            setattr(population, f"tau_decay_gaba_b{suffix}", 5.0)
            setattr(population, f"e_ampa{suffix}", 0.0)
            setattr(population, f"e_nmda{suffix}", 0.0)
            setattr(population, f"e_gaba_a{suffix}", -60.0)
            setattr(population, f"e_gaba_b{suffix}", -90.0)

            # NMDA Mg block defaults
            setattr(population, f"nmda_mg{suffix}", 1.0)
            setattr(population, f"nmda_Kd{suffix}", 3.57)
            setattr(population, f"nmda_gamma{suffix}", 0.062)
            setattr(population, f"nmda_vshift{suffix}", 0.0)

            # STDP signal state
            setattr(population, f"learning_w_exc{suffix}", 0.0)
            setattr(population, f"learn_int_exc{suffix}", 0.0)
            setattr(population, f"learning_w_inh{suffix}", 0.0)
            setattr(population, f"learn_int_inh{suffix}", 0.0)

        population.plasticity_on = 0.0

        # STDP parameters (defaults, overridden by enable_plasticity)
        population.theta_ltp_exc = -45.0
        population.theta_ltd_exc = -60.0
        population.half_ltp_exc = -40.0
        population.half_ltd_exc = -55.0
        population.slope_exc = 1.3
        population.A_ltp_exc = 1.0
        population.A_ltd_exc = 1.0
        population.learning_tau_exc = 20.0
        population.theta_ltp_inh = -77.0
        population.theta_ltd_inh = -70.0
        population.half_ltp_inh = -80.0
        population.half_ltd_inh = -73.0
        population.slope_inh = 1.2
        population.A_ltp_inh = 1.0
        population.A_ltd_inh = 1.0
        population.learning_tau_inh = 20.0

        return population

    def brian2_connection_synapse(self, pre_group, post_group):
        """Legacy single-synapse constructor (unused with conductance-based model)."""
        import brian2 as b2

        return b2.Synapses(
            pre_group,
            post_group,
            """
            w : 1
            multiplier : 1
            distance : 1
            prefix : 1
            """,
            on_pre="I += prefix * w * multiplier * pA",
            dt=0.025 * b2.ms,
        )

    def brian2_mechanism_synapse(
        self,
        pre_group,
        post_group,
        mechanism_name,
        mechanism_params,
        synapse_type,
        compartment="soma",
    ):
        """Create a conductance-based synapse for a specific mechanism

        Parameters
        ----------
        pre_group : brian2.NeuronGroup
        post_group : brian2.NeuronGroup
        mechanism_name : str
            One of "AMPA", "NMDA", "GABA_A", "GABA_B"
        mechanism_params : dict
            Must contain: e, g_unit, tau_rise, tau_decay, weight
        synapse_type : str
            "excitatory" or "inhibitory"
        compartment : str
            ``"soma"`` or ``"dend"``; where the conductance is delivered and
            which voltage the synapse (and its STDP signal) sees. A cell with
            no dendrite takes everything on its soma.
        """
        import math

        import brian2 as b2

        suffix = "_d" if compartment == "dend" else ""
        if suffix and f"A_ampa{suffix}" not in post_group.variables:
            suffix = ""  # single-compartment postsynaptic cell

        tau_rise = mechanism_params["tau_rise"]
        tau_decay = mechanism_params["tau_decay"]

        # Compute normalization factor for dual-exponential
        if tau_decay > tau_rise and tau_rise > 0:
            tp = (
                (tau_rise * tau_decay)
                / (tau_decay - tau_rise)
                * math.log(tau_decay / tau_rise)
            )
            factor = 1.0 / (-math.exp(-tp / tau_rise) + math.exp(-tp / tau_decay))
        else:
            factor = 1.0

        g_unit = mechanism_params["g_unit"]
        mech_lower = mechanism_name.lower()

        # STDP-capable mechanisms: AMPA, NMDA, GABA_A
        has_stdp = mechanism_name in ("AMPA", "NMDA", "GABA_A")

        # Choose the learning integral variable based on synapse type
        if synapse_type == "excitatory":
            learn_int_var = f"learn_int_exc{suffix}"
        else:
            learn_int_var = f"learn_int_inh{suffix}"

        model_eqs = """
            w : 1
            multiplier : 1
            distance : 1
        """

        if has_stdp:
            model_eqs += """
            w_plastic : 1
            last_int : 1
            w_min : 1
            w_max : 1
        """

        if has_stdp:
            # STDP weight update happens before conductance delivery
            on_pre_code = f"""
            delta = {learn_int_var}_post - last_int
            last_int = {learn_int_var}_post
            w_plastic = clip(w_plastic + plasticity_on_post * delta * w_plastic, w_min, w_max)
            A_{mech_lower}{suffix}_post += w * w_plastic * multiplier * {g_unit} * {factor}
            B_{mech_lower}{suffix}_post += w * w_plastic * multiplier * {g_unit} * {factor}
            """
        else:
            on_pre_code = f"""
            A_{mech_lower}{suffix}_post += w * multiplier * {g_unit} * {factor}
            B_{mech_lower}{suffix}_post += w * multiplier * {g_unit} * {factor}
            """

        synapse = b2.Synapses(
            pre_group,
            post_group,
            model_eqs,
            on_pre=on_pre_code,
            dt=0.025 * b2.ms,
        )

        synapse._mechanism_name = mechanism_name
        synapse._mechanism_params = mechanism_params
        synapse._has_stdp = has_stdp
        synapse._factor = factor
        synapse._compartment = "dend" if suffix else "soma"

        return synapse

    def brian2_synapse_site(self, population: str, section: str) -> tuple[str, str]:
        if section == "soma":
            return "soma", "soma"
        return ("dend" if population == "EXC" else "soma"), "dend"

    def brian2_noise_op(self, population_group, prng):
        """Ornstein-Uhlenbeck noise via run_regularly

        The OU process is separated from the main ODE system so that the
        deterministic equations can use the GSL adaptive solver.

        The update is the exact one Gfluct3 uses in the NEURON backend,

            g <- g0 + (g - g0) * exp(-h/tau) + std * sqrt(1 - exp(-2h/tau)) * N(0,1)

        rather than an Euler-Maruyama step. Both converge to the same process,
        but only the exact form holds the stationary variance at ``std**2``
        independently of the step, which is what lets a ``std_e``/``tau_e``
        fitted under NEURON mean the same thing here.
        """
        return population_group.run_regularly(
            """
            g_noise_e = g_e0 + (g_noise_e - g_e0) * exp(-(dt/ms) / tau_e) + amp_e * sqrt(1.0 - exp(-2.0 * (dt/ms) / tau_e)) * randn()
            g_noise_i = g_i0 + (g_noise_i - g_i0) * exp(-(dt/ms) / tau_i) + amp_i * sqrt(1.0 - exp(-2.0 * (dt/ms) / tau_i)) * randn()
            """,
            dt=population_group.clock.dt,
        )

    def brian2_noise_configure(
        self,
        population_group,
        std_e=0.003,
        std_i=0.0066,
        g_e0=0.0121,
        g_i0=0.0573,
        tau_e=2.728,
        tau_i=10.49,
        **kwargs,
    ):
        population_group.g_e0 = g_e0
        population_group.g_i0 = g_i0
        population_group.tau_e = tau_e
        population_group.tau_i = tau_i
        population_group.amp_e = std_e
        population_group.amp_i = std_i

        population_group.g_noise_e = g_e0
        population_group.g_noise_i = g_i0

    def brian2_default_noise(self, system: str):
        return self.neuron_default_noise(system)

    def brian2_default_weights(self, system: str):
        """The NEURON weights, unchanged."""
        return self.neuron_default_weights(system)
