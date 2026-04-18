import subprocess
import os
import shutil
import glob

from machinable import Interface
from pydantic import BaseModel

_WEB_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_WEB_DIR))


class Web(Interface):
    class Config(BaseModel):
        host: str = "localhost"
        port: int = 5173

    def launch(self):
        self.build_wheel()
        self.dev()

    def build_wheel(self):
        repo_root = _REPO_ROOT
        dist_dir = os.path.join(repo_root, "dist")
        static_dir = os.path.join(_WEB_DIR, "static")

        # Build pure-Python wheel
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", dist_dir, repo_root],
            check=True,
        )

        # Copy wheel to static/
        os.makedirs(static_dir, exist_ok=True)
        wheels = sorted(glob.glob(os.path.join(dist_dir, "livn-*.whl")))
        if not wheels:
            raise RuntimeError("No wheel built")

        # Remove old wheels from static/
        for old in glob.glob(os.path.join(static_dir, "livn-*.whl")):
            os.remove(old)

        wheel_name = os.path.basename(wheels[-1])
        target = os.path.join(static_dir, wheel_name)
        shutil.copy2(wheels[-1], target)

        # Write manifest so the frontend knows the wheel filename
        import json

        manifest = os.path.join(static_dir, "wheel.json")
        with open(manifest, "w") as f:
            json.dump({"filename": wheel_name}, f)

        print(f"Wheel copied to {target}")

    def dev(self):
        # Install npm deps if needed
        if not os.path.isdir(os.path.join(_WEB_DIR, "node_modules")):
            subprocess.run(["npm", "install"], cwd=_WEB_DIR, check=True)

        # Start Vite dev server
        subprocess.run(
            [
                "npx",
                "vite",
                "dev",
                "--host",
                self.config.host,
                "--port",
                str(self.config.port),
            ],
            cwd=_WEB_DIR,
        )
