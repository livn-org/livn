import os
from pathlib import Path

ENV = "LIVN_TEST_JAX_CACHE"


def cache_directory(rootpath) -> Path | None:
    setting = os.environ.get(ENV)
    if setting in ("0", "off", "no"):
        return None
    if setting:
        return Path(setting)
    return Path(rootpath) / ".pytest_cache" / "jax"


def install(rootpath) -> Path | None:
    directory = cache_directory(rootpath)
    if directory is None:
        return None

    try:
        import jax
    except ImportError:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return directory
