from __future__ import annotations

import pytest

from livn.backend.native import build


def _command(compiler: str) -> list[str]:
    return build._command([compiler], "out.so", ["a.c"])


def test_the_flags_that_keep_a_seeded_run_reproducible():
    for flags in (build.POSIX_FLAGS, build.EMSCRIPTEN_FLAGS, build.MSVC_FLAGS):
        joined = " ".join(flags)
        assert "-ffast-math" not in joined
        assert "-march=native" not in joined
    assert "-fno-fast-math" in build.POSIX_FLAGS
    assert "-fno-fast-math" in build.EMSCRIPTEN_FLAGS
    assert "/fp:precise" in build.MSVC_FLAGS


def test_emscripten_is_recognised_by_the_compiler_name():
    assert build.is_emscripten(["/opt/emsdk/upstream/emscripten/emcc"])
    assert build.is_emscripten(["emcc"])
    assert not build.is_emscripten(["/usr/bin/cc"])
    assert not build.is_emscripten(["gcc"])


def test_the_wasm_build_is_a_side_module():
    command = _command("/opt/emsdk/upstream/emscripten/emcc")
    assert "-sSIDE_MODULE=1" in command
    # -shared is not how Emscripten builds something dlopen can load, and
    # -lm/-fPIC are implied there
    assert "-shared" not in command
    assert "-dynamiclib" not in command


def test_the_native_build_is_a_shared_object():
    command = _command("/usr/bin/cc")
    assert "-shared" in command or "-dynamiclib" in command
    assert "-sSIDE_MODULE=1" not in command
    assert command[-1] == "-lm"


def test_msvc_takes_its_own_spelling():
    command = _command("cl.exe")
    assert "/LD" in command
    assert any(part.startswith("/Fe:") for part in command)


def test_the_library_keeps_the_so_name_under_emscripten(monkeypatch):
    # Pyodide reports platform.system() == "Emscripten"; dlopen wants .so
    monkeypatch.setattr(build.platform, "system", lambda: "Emscripten")
    assert build.library_name() == "librcsd.so"


def test_a_bare_cc_is_resolved_to_an_absolute_path(monkeypatch):
    monkeypatch.setenv("CC", "emcc")
    monkeypatch.setattr(build.shutil, "which", lambda name: "/opt/emsdk/" + name)
    assert build._compiler() == ["/opt/emsdk/emcc"]


def test_an_unresolvable_cc_is_left_alone(monkeypatch):
    monkeypatch.setenv("CC", "emcc --flag")
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    assert build._compiler() == ["emcc", "--flag"]


def test_the_cache_key_separates_targets(monkeypatch):
    native = build.source_digest(["/usr/bin/cc"])
    wasm = build.source_digest(["/opt/emsdk/upstream/emscripten/emcc"])
    assert native != wasm, (
        "a wasm build and a native build would share a cache entry, so one "
        "would be handed out for the other"
    )


@pytest.mark.parametrize(
    ("system", "expected"),
    [("Windows", "rcsd.dll"), ("Darwin", "librcsd.dylib"), ("Linux", "librcsd.so")],
)
def test_the_library_is_named_for_its_platform(monkeypatch, system, expected):
    monkeypatch.setattr(build.platform, "system", lambda: system)
    assert build.library_name() == expected
