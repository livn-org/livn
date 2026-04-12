import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

ON_CI = os.getenv("CI") == "true"


def test_using_the_dataset():
    pytest.importorskip("datasets")
    from examples import using_the_dataset  # noqa: F401


@pytest.mark.skipif(
    not os.getenv("LIVN_BACKEND"), reason="no simulation backend selected"
)
@pytest.mark.skipif(ON_CI, reason="predefined system not available on CI")
def test_run_a_simulation():
    from examples import run_a_simulation

    run_a_simulation.env.close()


@pytest.mark.skipif(
    not os.getenv("LIVN_BACKEND"), reason="no simulation backend selected"
)
@pytest.mark.skipif(ON_CI, reason="predefined system not available on CI")
def test_parallel_simulation():
    pytest.importorskip("matplotlib")
    from examples import parallel_simulation

    parallel_simulation.env.close()


@pytest.mark.skipif(os.getenv("LIVN_BACKEND") != "diffrax", reason="diffrax only")
@pytest.mark.skipif(ON_CI, reason="predefined system not available on CI")
def test_differentiable_simulation():
    from examples import differentiable_simulation  # noqa: F401
