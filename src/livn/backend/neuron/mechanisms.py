from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_loaded: dict[str, str] = {}
_hoc_configured = False


def _is_complete(compiled: str) -> bool:
    return os.path.isfile(os.path.join(compiled, ".livn_nrnivmodl_ok"))


def _atomic_compile(compiled: str, contents: dict[str, str]) -> None:
    """Compile ``contents`` into ``compiled`` via a private temp dir + atomic rename.

    nrnivmodl runs in a per-process temp directory and a completion marker
    is written and the finished build is moved into place with a single
    atomic rename. A process that loses the publish race discards.
    """
    if not shutil.which("nrnivmodl"):
        raise ModuleNotFoundError("nrnivmodl not found on PATH")
    parent = os.path.dirname(compiled)
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=os.path.basename(compiled) + ".tmp.", dir=parent)
    try:
        for m, data in contents.items():
            with open(os.path.join(tmp, os.path.basename(m)), "w") as f:
                f.write(data)
        subprocess.run(["nrnivmodl"], cwd=tmp, check=True)
        open(os.path.join(tmp, ".livn_nrnivmodl_ok"), "w").close()
        try:
            os.rename(tmp, compiled)  # atomic publish
            tmp = None
        except OSError:
            pass  # another process published first; fall through to cleanup
    finally:
        if tmp is not None and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def compile_mechanisms(directory: str, force: bool = False) -> str:
    """Compile the ``.mod`` files under ``directory`` (cached by content hash).

    Returns the directory that ``neuron.load_mechanisms`` should be pointed at.
    """
    src = os.path.abspath(directory)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Mechanism directory does not exist: {src}")

    mod_files = [
        f
        for f in glob.glob(os.path.join(src, "**/*.mod"), recursive=True)
        if f"{os.sep}compiled{os.sep}" not in f
    ]

    digest = hashlib.sha256()
    contents: dict[str, str] = {}
    seen: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    # nrnivmodl compiles a flat directory, so files are copied in by basename;
    # sort by basename (then path) for a stable hash and to detect basename
    # collisions (which would silently overwrite each other).
    for m in sorted(mod_files, key=lambda x: (os.path.basename(x), x)):
        with open(m) as fh:
            data = fh.read()
        base = os.path.basename(m)
        digest.update(base.encode())
        digest.update(data.encode())
        contents[m] = data
        if base in seen:
            collisions.setdefault(base, [seen[base]]).append(m)
        seen[base] = m

    if collisions:
        logger.warning(
            "Mechanism directory %s has %d colliding .mod basename(s); flat "
            "compilation keeps only one of each, so conflated variants may be "
            "dropped: %s",
            src,
            len(collisions),
            collisions,
        )

    compiled = os.path.join(src, "compiled", digest.hexdigest())

    if force and os.path.isdir(compiled):
        shutil.rmtree(compiled)

    if os.path.isdir(compiled) and not _is_complete(compiled):
        aside = f"{compiled}.stale.{os.getpid()}"
        try:
            os.rename(compiled, aside)
        except OSError:
            aside = None
        if aside and os.path.isdir(aside):
            shutil.rmtree(aside, ignore_errors=True)

    if not (os.path.isdir(compiled) and _is_complete(compiled)):
        _atomic_compile(compiled, contents)

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
