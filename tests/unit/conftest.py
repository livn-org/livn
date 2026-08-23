import pytest

from testing import budget


def pytest_collection_modifyitems(config, items):
    budget.refuse_mpiexec_items(
        [item for item in items if "/unit/" in item.nodeid.replace("\\", "/")]
    )


@pytest.fixture(autouse=True)
def _fast_tier_budget(request, monkeypatch):
    allow_tracing = request.node.get_closest_marker("traces") is not None
    budget.install(monkeypatch, allow_tracing=allow_tracing)
    yield
