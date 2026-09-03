from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_int,
    c_long,
    c_uint,
    c_void_p,
)

import numpy as np

from livn.backend.native import build

LIBRARY_ENV = "LIVN_NATIVE_LIB"

# the C enums, mirrored (kept in sync with csrc/rcsd.h)
M_PAS = 1 << 0
M_CONSTANT = 1 << 1
M_NA_CONC = 1 << 2
M_K_CONC = 1 << 3
M_CA_CONC = 1 << 4
M_NAS = 1 << 5
M_KDR = 1 << 6
M_CAN = 1 << 7
M_CAL = 1 << 8
M_KCA = 1 << 9
M_KA_V1IN = 1 << 10

SEC_SOMA, SEC_DEND, SEC_AXON = 0, 1, 2

(
    P_CM,
    P_G_PAS,
    P_E_PAS,
    P_IC,
    P_GMAX_NAS,
    P_VHALF_NAS,
    P_SLOPE_NAS,
    P_GMAX_KDR,
    P_GMAX_CAN,
    P_GMAX_CAL,
    P_GMAX_KCA,
    P_KD_KCA,
    P_GMAX_KA,
    P_F_CA,
    P_ALPHA_CA,
    P_KCA_CA,
    P_CAI0,
    P_D_NA,
    P_BETA_NA,
    P_NAI0,
    P_NAO0,
    P_D_K,
    P_BETA_K,
    P_KI0,
    P_KO0,
    P_CAO,
    NPARAM,
) = range(27)

(
    S_V,
    S_H,
    S_N,
    S_MN,
    S_HN,
    S_ML,
    S_A,
    S_B,
    S_CAI,
    S_NAI,
    S_NAO,
    S_KI,
    S_KO,
    S_ENA,
    S_EK,
    S_INA,
    S_IK,
    S_ICA,
    S_IPAS,
    S_IREST,
    S_IMEM,
    NSTATE,
) = range(22)

SYN_LINEXP2, SYN_NMDA, SYN_DEP, SYN_STDP, SYN_STDP_NMDA, SYN_STDP_INH = range(6)

SYNAPSE_KINDS = {
    "LinExp2Syn": SYN_LINEXP2,
    "LinExp2SynNMDA": SYN_NMDA,
    "DepLinExp2Syn": SYN_DEP,
    "StdpLinExp2Syn": SYN_STDP,
    "StdpLinExp2SynNMDA": SYN_STDP_NMDA,
    "StdpLinExp2SynInh": SYN_STDP_INH,
}

SP_NAMES = (
    "tau_rise",
    "tau_decay",
    "e",
    "mg",
    "Kd",
    "gamma",
    "vshift",
    "U",
    "tau_rec",
    "plasticity_on",
    "w_init",
    "A_ltp",
    "A_ltd",
    "theta_ltp",
    "theta_ltd",
    "ltp_sigmoid_half",
    "ltd_sigmoid_half",
    "learning_slope",
    "learning_tau",
    "w_max",
    "w_min",
)
SP = {name: i for i, name in enumerate(SP_NAMES)}
SP_N = len(SP_NAMES)

SS_NAMES = ("A", "B", "learning_w", "learn_int", "ltd", "ltp", "w", "factor", "g", "i")
SS = {name: i for i, name in enumerate(SS_NAMES)}
SS_N = len(SS_NAMES)

NWEIGHT = 4

STIM_EXTRACELLULAR, STIM_CURRENT, STIM_CURRENT_DENSITY, STIM_PHOTON_FLUX = range(4)
STIM_MODES = {
    "extracellular": STIM_EXTRACELLULAR,
    "current": STIM_CURRENT,
    "current_density": STIM_CURRENT_DENSITY,
    "photon_flux": STIM_PHOTON_FLUX,
}

OK, NEED_STIMULUS, ERROR = 0, 1, -1

_c_int_p = POINTER(c_int)
_c_double_p = POINTER(c_double)
_c_long_p = POINTER(c_long)
_c_uint_p = POINTER(c_uint)

_SIGNATURES = {
    "rcsd_version": (c_char_p, []),
    "rcsd_last_error": (c_char_p, []),
    "rcsd_create": (c_void_p, [c_double, c_double]),
    "rcsd_destroy": (None, [c_void_p]),
    "rcsd_add_cell": (c_int, [c_void_p, c_int, c_int, c_double, c_double, c_double]),
    "rcsd_add_section": (
        c_int,
        [
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_double,
            c_double,
            c_double,
            c_double,
            c_uint,
            c_int,
            c_double,
        ],
    ),
    "rcsd_section_set": (c_int, [c_void_p, c_int, c_int, c_double]),
    "rcsd_section_get": (c_double, [c_void_p, c_int, c_int]),
    "rcsd_section_geometry": (c_int, [c_void_p, c_int, c_double, c_double, c_double]),
    "rcsd_section_info": (
        c_int,
        [c_void_p, c_int, _c_int_p, _c_double_p, _c_double_p, _c_double_p, _c_uint_p],
    ),
    "rcsd_cell_count": (c_int, [c_void_p]),
    "rcsd_node_count": (c_int, [c_void_p]),
    "rcsd_cell_section_count": (c_int, [c_void_p, c_int]),
    "rcsd_cell_section": (c_int, [c_void_p, c_int, c_int]),
    "rcsd_section_node": (c_int, [c_void_p, c_int, c_double]),
    "rcsd_cell_set": (c_int, [c_void_p, c_int, c_double, c_double, c_double]),
    "rcsd_node_state": (c_double, [c_void_p, c_int, c_int]),
    "rcsd_node_area": (c_double, [c_void_p, c_int]),
    "rcsd_add_synapse": (c_int, [c_void_p, c_int, c_int, c_double, c_int]),
    "rcsd_add_input": (c_int, [c_void_p, c_int]),
    "rcsd_set_input_spikes": (c_int, [c_void_p, c_int, _c_double_p, c_int]),
    "rcsd_add_connections": (
        c_int,
        [c_void_p, c_int, _c_int_p, _c_int_p, _c_double_p, _c_double_p],
    ),
    "rcsd_synapse_count": (c_int, [c_void_p]),
    "rcsd_connection_count": (c_int, [c_void_p]),
    "rcsd_synapse_params": (_c_double_p, [c_void_p]),
    "rcsd_synapse_states": (_c_double_p, [c_void_p]),
    "rcsd_synapse_stride": (c_int, [c_void_p]),
    "rcsd_connection_weights": (_c_double_p, [c_void_p]),
    "rcsd_synapse_node": (c_int, [c_void_p, c_int]),
    "rcsd_synapse_refresh": (c_int, [c_void_p, c_int]),
    "rcsd_set_noise": (
        c_int,
        [
            c_void_p,
            c_int,
            c_int,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_double,
            c_int,
        ],
    ),
    "rcsd_set_noise_stream": (c_int, [c_void_p, c_int, c_int, c_uint, c_uint, c_uint]),
    "rcsd_noise_count": (c_int, [c_void_p]),
    "rcsd_random123_normal": (c_double, [c_uint, c_uint, c_uint, c_uint, c_int]),
    "rcsd_add_opsin": (c_int, [c_void_p, c_int, c_int, c_double]),
    "rcsd_opsin_set": (
        c_int,
        [c_void_p, c_int] + [c_double] * 10,
    ),
    "rcsd_opsin_count": (c_int, [c_void_p]),
    "rcsd_opsin_state": (
        c_int,
        [c_void_p, c_int, _c_double_p, _c_double_p, _c_double_p],
    ),
    "rcsd_set_stimulus_rows": (
        c_int,
        [c_void_p, c_int, c_int, _c_int_p, _c_int_p, c_double],
    ),
    "rcsd_set_stimulus_window": (
        c_int,
        [c_void_p, c_int, c_long, c_int, _c_double_p, c_long],
    ),
    "rcsd_clear_stimulus": (c_int, [c_void_p, c_int]),
    "rcsd_extracellular_junctions": (c_int, [c_void_p, _c_int_p, _c_int_p]),
    "rcsd_stimulus_needed": (c_long, [c_void_p, c_int]),
    "rcsd_record_spikes": (c_int, [c_void_p, c_int]),
    "rcsd_record_voltage": (c_int, [c_void_p, c_int, c_int, c_double]),
    "rcsd_record_current": (c_int, [c_void_p, c_int, c_int, c_double]),
    "rcsd_clear_recordings": (c_int, [c_void_p]),
    "rcsd_spike_count": (c_int, [c_void_p]),
    "rcsd_spike_cells": (_c_int_p, [c_void_p]),
    "rcsd_spike_times": (_c_double_p, [c_void_p]),
    "rcsd_voltage_record_count": (c_int, [c_void_p]),
    "rcsd_voltage_record_length": (c_int, [c_void_p, c_int]),
    "rcsd_voltage_record": (_c_double_p, [c_void_p, c_int]),
    "rcsd_current_record_count": (c_int, [c_void_p]),
    "rcsd_current_record_length": (c_int, [c_void_p, c_int]),
    "rcsd_current_record": (_c_double_p, [c_void_p, c_int]),
    "rcsd_set_dt": (c_int, [c_void_p, c_double]),
    "rcsd_dt": (c_double, [c_void_p]),
    "rcsd_set_v_init": (c_int, [c_void_p, c_double]),
    "rcsd_init": (c_int, [c_void_p]),
    "rcsd_pin_resting": (c_int, [c_void_p]),
    "rcsd_run": (c_int, [c_void_p, c_long, _c_long_p]),
    "rcsd_step": (c_long, [c_void_p]),
    "rcsd_time": (c_double, [c_void_p]),
    "rcsd_initialized": (c_int, [c_void_p]),
}

_loaded: ctypes.CDLL | None = None
_loaded_path: str | None = None


def _module_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def shipped_library() -> str | None:
    """The library a wheel installs next to this module, if any."""
    path = os.path.join(_module_directory(), build.library_name())
    return path if os.path.isfile(path) else None


def library_path(build_if_missing: bool = False) -> str | None:
    """Where the library comes from, in order.

    ``LIVN_NATIVE_LIB`` first. In a source checkout (``csrc/`` next to this
    module) the build matching the current sources wins, compiled on demand
    when allowed, so an edit to the C is never shadowed by an older library;
    the one a wheel ships next to the module is the fallback.
    """
    given = os.environ.get(LIBRARY_ENV)
    if given:
        return given if os.path.isfile(given) else None
    if os.path.isdir(build.SOURCE_DIR):
        found = build.cached_library()
        if found is None and build_if_missing:
            try:
                found = build.compile_library()
            except RuntimeError:
                found = None
        if found is not None:
            return found
    return shipped_library()


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


def load(build_if_missing: bool = True) -> ctypes.CDLL:
    global _loaded, _loaded_path
    if _loaded is not None:
        return _loaded
    path = library_path(build_if_missing=build_if_missing)
    if path is None:
        raise ImportError(
            "the native backend's library (librcsd) is not available: no wheel "
            f"library, no cached build, and {LIBRARY_ENV} is not set"
        )
    lib = _bind(ctypes.CDLL(path))
    version = lib.rcsd_version().decode()
    if version.split(".")[0] != "0":
        raise ImportError(f"librcsd at {path} is version {version}, expected 0.x")
    _loaded, _loaded_path = lib, path
    return lib


def loaded_path() -> str | None:
    return _loaded_path


def available() -> bool:
    """Whether a library can be loaded without building anything.

    Cheap enough for import time: a file check and a dlopen.
    """
    try:
        load(build_if_missing=False)
    except Exception:
        return False
    return True


def version() -> str:
    return load().rcsd_version().decode()


class NativeError(RuntimeError):
    pass


def check(code: int, lib: ctypes.CDLL | None = None) -> int:
    """Raise with the library's message when a call reports an error."""
    if code < 0:
        lib = lib or load()
        message = lib.rcsd_last_error().decode(errors="replace")
        raise NativeError(message or "librcsd reported an error")
    return code


def as_int_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int32).ravel())


def as_double_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())


def int_ptr(array: np.ndarray):
    return array.ctypes.data_as(_c_int_p)


def double_ptr(array: np.ndarray):
    return array.ctypes.data_as(_c_double_p)


def view(pointer, shape: tuple[int, ...]) -> np.ndarray:
    """A numpy view over sim-owned memory (valid until the sim reallocates)."""
    n = int(np.prod(shape)) if shape else 0
    if n == 0 or not pointer:
        return np.zeros(shape, dtype=np.float64)
    buffer = (c_double * n).from_address(ctypes.addressof(pointer.contents))
    return np.frombuffer(buffer, dtype=np.float64).reshape(shape)


def site_view(lib, sim, pointer, columns: int) -> np.ndarray:
    """``[column, site]`` view over one of the column-major site tables."""
    n_sites = lib.rcsd_synapse_count(sim)
    stride = lib.rcsd_synapse_stride(sim)
    if n_sites == 0 or stride == 0:
        return np.zeros((columns, 0), dtype=np.float64)
    return view(pointer, (columns, stride))[:, :n_sites]


def copy_doubles(pointer, n: int) -> np.ndarray:
    if n <= 0 or not pointer:
        return np.zeros(0, dtype=np.float64)
    return np.ctypeslib.as_array(pointer, shape=(n,)).copy()


def copy_ints(pointer, n: int) -> np.ndarray:
    if n <= 0 or not pointer:
        return np.zeros(0, dtype=np.int32)
    return np.ctypeslib.as_array(pointer, shape=(n,)).copy()
