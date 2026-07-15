from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess

_loaded: dict[str, str] = {}
_hoc_configured = False


def compile_mechanisms(directory: str, force: bool = False) -> str:
    """Compile the ``.mod`` files under ``directory`` (cached by content hash).

    Returns the directory that ``neuron.load_mechanisms`` should be pointed at.
    Compilation happens at most once per unique set of mechanism sources; the
    result is reused across processes and runs.
    """
    src = os.path.abspath(directory)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Mechanism directory does not exist: {src}")

    output_path = os.path.join(src, "compiled")
    mod_files = [
        f
        for f in glob.glob(os.path.join(src, "**/*.mod"), recursive=True)
        if output_path not in f
    ]

    digest = hashlib.sha256()
    contents: dict[str, str] = {}
    for m in sorted(mod_files, key=lambda x: x.replace(src, "")):
        with open(m) as fh:
            data = fh.read()
        digest.update(data.encode())
        contents[m] = data

    compiled = os.path.join(output_path, digest.hexdigest())

    if force and os.path.isdir(compiled):
        shutil.rmtree(compiled)

    if not os.path.isdir(compiled):
        if not shutil.which("nrnivmodl"):
            raise ModuleNotFoundError("nrnivmodl not found on PATH")
        os.makedirs(compiled)
        for m, data in contents.items():
            with open(os.path.join(compiled, os.path.basename(m)), "w") as f:
                f.write(data)
        subprocess.run(["nrnivmodl"], cwd=compiled, check=True)

    return compiled


def load_mechanisms(directory: str) -> str:
    """Compile (if needed) and load a mechanism directory into NEURON once."""
    compiled = compile_mechanisms(directory)
    if compiled in _loaded:
        return compiled

    from neuron import load_mechanisms as _nrn_load

    _nrn_load(compiled)
    _loaded[compiled] = compiled
    return compiled


def configure(mechanisms_directory: str | None = None):
    """Load mechanisms and initialize the HOC environment + ParallelContext.

    Idempotent so the HOC side runs once per process
    and mechanism loading is cached per directory.
    """
    global _hoc_configured
    from neuron import h

    if mechanisms_directory is not None:
        load_mechanisms(mechanisms_directory)

    if _hoc_configured:
        return h

    h.load_file("stdrun.hoc")
    h.load_file("loadbal.hoc")
    # NB: fast i_membrane_ is enabled lazily by the recorder when membrane
    # current recording is requested so enabling it here would make psolve assert
    # on ranks that own no sections (e.g. more ranks than selected cells).
    h.cvode.cache_efficient(1)
    if not hasattr(h, "pc"):
        h("objref pc")
        h.pc = h.ParallelContext()
    # more accurate integration of synaptic discontinuities
    if hasattr(h, "nrn_netrec_state_adjust"):
        h.nrn_netrec_state_adjust = 1
    if hasattr(h, "nrn_sparse_partrans"):
        h.nrn_sparse_partrans = 1

    _hoc_configured = True
    return h
