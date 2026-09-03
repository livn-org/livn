import os

import pytest

SYSTEM_ENV = "LIVN_TEST_SYSTEM"
SELECTION_ENV = "LIVN_TEST_SELECTION"


def livn_test_system() -> str:
    system = os.environ.get(SYSTEM_ENV)
    if not system:
        pytest.skip(f"{SYSTEM_ENV} is not set")
    return system


def livn_test_selection() -> str | None:
    return os.environ.get(SELECTION_ENV) or None


def livn_test_env(*args, **kwargs):
    from livn.env import Env
    from testing.capabilities import supports

    env = Env(livn_test_system(), *args, **kwargs)
    selection = livn_test_selection()
    if selection and supports(env, "simulation"):
        env.selection(selection)
    return env


def livn_test_mea(system: str | None = None):
    import numpy as np

    from livn.io import MEA
    from livn.system import System

    xyz = np.asarray(System(system or livn_test_system()).neuron_coordinates)[:, 1:]
    electrode = np.array([xyz[:, 0].min() - 100.0, xyz[:, 1].mean(), xyz[:, 2].mean()])
    reach = float(np.linalg.norm(xyz - electrode, axis=1).max()) + 200.0

    return MEA([[0, *electrode]], input_radius=reach, output_radius=reach)


def safe_stimulus_amplitudes(env, fraction: float = 0.25):
    import numpy as np

    bounds = env.model.stimulus_bounds("extracellular")
    if bounds is None:
        return (125.0, 250.0)
    reach = np.abs(np.asarray(env.channel_reach(env.stimulus_coordinates(False))))
    gain = float(reach.max())  # mV per uA at the most strongly coupled section
    peak = fraction * float(bounds[1]) / gain
    return (peak / 2.0, peak)
