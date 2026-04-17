import subprocess
import os
import shutil
import glob

from machinable import Interface
from pydantic import BaseModel


class Web(Interface):
    class Config(BaseModel):
        host: str = "localhost"
        port: int = 5173

    def launch(self):
        self.build_wheel()
        self.dev()

    def build_wheel(self):
        repo_root = self._find_repo_root()
        dist_dir = os.path.join(repo_root, "dist")
        static_dir = os.path.join(os.path.dirname(__file__), "web", "static")

        # Build pure-Python wheel
        subprocess.run(
            ["python", "-m", "build", "--wheel", "--outdir", dist_dir, repo_root],
            check=True,
        )

        # Copy wheel to static/
        os.makedirs(static_dir, exist_ok=True)
        wheels = sorted(glob.glob(os.path.join(dist_dir, "livn-*.whl")))
        if not wheels:
            raise RuntimeError("No wheel built")

        target = os.path.join(static_dir, "livn.whl")
        shutil.copy2(wheels[-1], target)
        print(f"Wheel copied to {target}")

    def dev(self):
        web_dir = os.path.join(os.path.dirname(__file__), "web")

        # Install npm deps if needed
        if not os.path.isdir(os.path.join(web_dir, "node_modules")):
            subprocess.run(["npm", "install"], cwd=web_dir, check=True)

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
            cwd=web_dir,
        )

    def _find_repo_root(self):
        import os

        d = os.path.dirname(__file__)
        while d != "/":
            if os.path.isfile(os.path.join(d, "pyproject.toml")):
                return d
            d = os.path.dirname(d)
        raise RuntimeError("Could not find repo root")
