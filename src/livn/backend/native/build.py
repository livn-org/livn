from __future__ import annotations

import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SOURCES = ("rcsd.c", "synapse.c", "noise.c", "opsin.c", "stimulus.c", "random123.c")
CACHE_DIR_ENV = "LIVN_CACHE_DIR"
# the flags are part of the contract, see csrc/Makefile
POSIX_FLAGS = ("-O2", "-std=c99", "-fno-fast-math", "-fPIC", "-D_USE_MATH_DEFINES")
MSVC_FLAGS = ("/O2", "/fp:precise", "/D_USE_MATH_DEFINES", "/nologo")
# WebAssembly, for Pyodide: a side module keeps every exported symbol
# reachable through dlopen, which is what ctypes goes through there
EMSCRIPTEN_FLAGS = (
    "-O2",
    "-std=c99",
    "-fno-fast-math",
    "-D_USE_MATH_DEFINES",
    "-sSIDE_MODULE=1",
)


def library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "rcsd.dll"
    if system == "Darwin":
        return "librcsd.dylib"
    # Emscripten side modules keep the .so name, which is what dlopen wants
    return "librcsd.so"


def cache_directory() -> str:
    given = os.environ.get(CACHE_DIR_ENV)
    if given:
        return os.path.join(given, "native")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "livn", "native")


def _compiler() -> list[str] | None:
    given = os.environ.get("CC")
    if given:
        parts = given.split()
        # resolve a bare name so the build survives a subprocess with a
        # narrower PATH than the shell that set CC (uv's build backend)
        found = shutil.which(parts[0])
        if found:
            parts[0] = found
        return parts
    if platform.system() == "Windows":
        for name in ("cl", "gcc", "clang"):
            found = shutil.which(name)
            if found:
                return [found]
        return None
    for name in ("cc", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return [found]
    return None


def source_digest(compiler: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(SOURCE_DIR, "*.[ch]"))):
        with open(path, "rb") as fh:
            digest.update(os.path.basename(path).encode())
            digest.update(fh.read())
    digest.update(" ".join(compiler or _compiler() or ["none"]).encode())
    digest.update(platform.machine().encode())
    digest.update(platform.system().encode())
    return digest.hexdigest()[:16]


def cached_library() -> str | None:
    """The cached build for the current sources, if it exists."""
    path = os.path.join(cache_directory(), source_digest(), library_name())
    return path if os.path.isfile(path) else None


def is_emscripten(compiler: list[str] | None = None) -> bool:
    """Whether we are compiling to WebAssembly with Emscripten."""
    if platform.system() == "Emscripten" or sys.platform == "emscripten":
        return True
    compiler = compiler or _compiler() or []
    return bool(compiler) and os.path.basename(compiler[0]).lower().startswith("emcc")


def _command(compiler: list[str], output: str, sources: list[str]) -> list[str]:
    exe = os.path.basename(compiler[0]).lower()
    if exe.startswith("cl"):
        return [*compiler, *MSVC_FLAGS, "/LD", *sources, "/Fe:" + output]
    if is_emscripten(compiler):
        # a side module, which is what Emscripten's dlopen (and so Pyodide's
        # ctypes) can load; -fPIC and -lm are implied and -shared is not it
        return [*compiler, *EMSCRIPTEN_FLAGS, "-o", output, *sources]
    shared = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
    return [*compiler, *POSIX_FLAGS, shared, "-o", output, *sources, "-lm"]


def compile_library(output_dir: str | None = None, force: bool = False) -> str:
    """Build the library into ``output_dir`` (default: the cache) and return its path."""
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError(
            "no C compiler found to build the native backend; install gcc, "
            "clang or MSVC, set CC, or install a livn wheel that ships librcsd"
        )
    if output_dir is None:
        output_dir = os.path.join(cache_directory(), source_digest(compiler))
    target = os.path.join(output_dir, library_name())
    if os.path.isfile(target) and not force:
        return target

    os.makedirs(output_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="librcsd.", dir=output_dir)
    try:
        sources = [os.path.join(SOURCE_DIR, name) for name in SOURCES]
        for header in glob.glob(os.path.join(SOURCE_DIR, "*.h")):
            shutil.copy(header, tmp)
        built = os.path.join(tmp, library_name())
        command = _command(compiler, built, sources)
        result = subprocess.run(
            command, cwd=tmp, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                "building the native backend failed:\n"
                + " ".join(command)
                + "\n"
                + result.stdout
                + result.stderr
            )
        try:
            os.replace(built, target)
        except OSError:
            if not os.path.isfile(target):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    print(compile_library(out, force=True))
