from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

NATIVE_DIR = os.path.join("src", "livn", "backend", "native")


def _platform_tag() -> str:
    given = os.environ.get("LIVN_WHEEL_PLATFORM")
    if given:
        return given
    try:
        from packaging import tags

        return next(iter(tags.sys_tags())).platform
    except Exception:
        import sysconfig

        tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        if sys.platform == "linux":
            tag = tag.replace("linux", "manylinux_2_28")
        return tag


class NativeBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or os.environ.get("LIVN_SKIP_NATIVE"):
            return

        # loaded by path: importing the package would pull in livn's runtime
        # dependencies, which the isolated build environment does not have
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "livn_native_build", os.path.join(self.root, NATIVE_DIR, "build.py")
        )
        native_build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native_build)

        tag = _platform_tag()
        wants_wasm = "wasm" in tag or "emscripten" in tag
        if wants_wasm != native_build.is_emscripten():
            raise RuntimeError(
                f"the wheel tag {tag!r} and the compiler disagree: "
                f"{'no' if wants_wasm else 'an'} Emscripten compiler is in use. "
                "Set CC to emcc for a pyemscripten wheel, or leave "
                "LIVN_WHEEL_PLATFORM unset for a native one."
            )

        self._staging = tempfile.mkdtemp(prefix="livn-librcsd.")
        library = native_build.compile_library(output_dir=self._staging, force=True)

        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{_platform_tag()}"
        build_data.setdefault("force_include", {})[library] = (
            "livn/backend/native/" + os.path.basename(library)
        )
        target = (
            "WebAssembly (Emscripten side module)"
            if wants_wasm
            else f"{platform.system()} {platform.machine()}"
        )
        self.app.display_info(f"built {os.path.basename(library)} for {target}")

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        staging = getattr(self, "_staging", None)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
            self._staging = None
