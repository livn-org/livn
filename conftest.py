pytest_plugins = ("testing.mpiexec",)


def pytest_configure(config):
    # before anything imports jax and starts compiling
    from testing import jaxcache

    jaxcache.install(config.rootpath)


from testing import (  # noqa: E402,F401  (re-exported for `from conftest import ...`)
    backend_supports,
    livn_test_env,
    livn_test_mea,
    livn_test_selection,
    livn_test_system,
    supports,
)
