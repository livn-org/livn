import pytest

from livn.types import Capability
from testing import backend_supports, supports


class _NoDeclaration:
    pass


class _Declares:
    capabilities = frozenset({Capability.SIMULATION})


def test_a_backend_that_declares_nothing_supports_nothing():
    assert not supports(_NoDeclaration, "simulation")
    assert not supports(_NoDeclaration(), "mpi")


def test_asking_for_several_means_all_of_them():
    assert supports(_Declares, "simulation")
    assert not supports(_Declares, "simulation", "mpi")


def test_a_name_no_capability_has_is_refused_rather_than_skipped():
    with pytest.raises(ValueError, match="not a valid Capability"):
        supports(_Declares, "sumilation")


def test_every_capability_is_a_difference_between_backends():
    declared = {}
    for name in ("default", "neuron", "brian2", "diffrax"):
        module = pytest.importorskip(
            f"livn.backend.{name}", reason=f"{name} is not installed"
        )
        declared[name] = module.Env.capabilities

    universal = set(Capability).intersection(*declared.values())
    assert not universal, (
        "every backend declares "
        f"{sorted(c.value for c in universal)}, so asking about it can only "
        "return True. Drop it from Capability and gate those tests on "
        "something that still distinguishes."
    )


def test_the_selected_backend_answers_for_itself():
    from livn.env import Env

    for capability in Capability:
        assert backend_supports(capability.value) == (capability in Env.capabilities)
