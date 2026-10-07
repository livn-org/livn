import json
import subprocess
import sys

import numpy as np
import pytest

from livn.backend import backend
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.models.rcsd.neuron.extracellular import constants

pytestmark = pytest.mark.skipif(
    backend() != "neuron", reason="the profile is measured with NEURON"
)


def _measured(population: str) -> dict:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from livn.models.rcsd.neuron.extracellular import measure; "
            f"print(json.dumps(measure({population!r})))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_stored_profile_and_amplitudes_are_what_the_cells_produce():
    profile, amplitude = constants(_measured("EXC"), _measured("INH"))
    model = ReducedCalciumSomaDendrite
    np.testing.assert_allclose(
        profile["distance_um"], model.EXTRACELLULAR_PROFILE["distance_um"]
    )
    np.testing.assert_allclose(
        profile["trough_uv"], model.EXTRACELLULAR_PROFILE["trough_uv"], atol=1e-3
    )
    assert amplitude["INH"] == pytest.approx(
        model.EXTRACELLULAR_AMPLITUDE["INH"], rel=1e-6
    )
