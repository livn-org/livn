import pytest


@pytest.fixture(scope="module", autouse=True)
def _release_compiled_programs():
    yield
    try:
        import jax
    except ImportError:
        return
    jax.clear_caches()
