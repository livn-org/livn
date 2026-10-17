import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("diffrax")

from test_trainable_system import needs_diffrax_axon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SCRIPT = textwrap.dedent(
    """
    import sys
    import jax
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", sys.argv[1])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    from test_trainable_system import N, _env, _grad, _parameterization
    from livn.optimization.trainable import TrainableSystem

    grad = _grad(_env(TrainableSystem(_parameterization(), n_neurons=N)))
    leaves = jax.tree_util.tree_leaves(grad.system.parameterization)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves)
    print("OK")
    """
)


@needs_diffrax_axon
def test_the_trainable_gradient_loads_from_the_persistent_cache(tmp_path):
    _twice(tmp_path, SCRIPT)


def _twice(tmp_path, source):
    script = tmp_path / "cached_gradient.py"
    script.write_text(source)
    cache = tmp_path / "cache"
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "LIVN_BACKEND": "diffrax",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), str(HERE)]),
    }
    for attempt in ("writes", "reads"):
        done = subprocess.run(
            [sys.executable, str(script), str(cache)],
            env=env,
            capture_output=True,
            text=True,
            timeout=400,
            cwd=ROOT,
        )
        assert done.returncode == 0 and "OK" in done.stdout, (
            f"the run that {attempt} the cache failed:\n{done.stdout}{done.stderr}"
        )
    assert any(cache.iterdir()), "nothing was cached, so the second run proved nothing"
