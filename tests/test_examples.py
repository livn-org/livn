import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def test_using_the_dataset():
    from examples import using_the_dataset


def test_run_a_simulation():
    from examples import run_a_simulation

    run_a_simulation.env.close()


def test_parallel_simulation():
    from examples import parallel_simulation

    parallel_simulation.env.close()


@pytest.mark.skipif(os.getenv("LIVN_BACKEND") != "diffrax", reason="diffrax only")
def test_differentiable_simulation():
    from examples import differentiable_simulation
