import json
import subprocess
import os

from aiohttp import web
import aiohttp_cors

from machinable import Interface
from pydantic import BaseModel

_GLOBAL_ROOTS_FILE = os.path.expanduser("~/.livn/roots.json")


def _default_experiment_root() -> str:
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return str(parent / "livn_experiments")
    return os.path.expanduser("~/livn_experiments")


def _load_experiments() -> list[dict]:
    """Return a list of experiment dicts across all known roots."""
    # Collect roots: global registry + LIVN_EXPERIMENT_ROOT env var
    roots: list[str] = []
    try:
        with open(_GLOBAL_ROOTS_FILE) as f:
            roots = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    env_root = os.environ.get("LIVN_EXPERIMENT_ROOT")
    if env_root:
        abs_env = os.path.abspath(env_root)
        if abs_env not in roots:
            roots.append(abs_env)

    if not roots:
        roots = [_default_experiment_root()]

    experiments: list[dict] = []
    seen: set[str] = set()

    for root in roots:
        registry_path = os.path.join(root, "experiments.json")
        if not os.path.isfile(registry_path):
            continue
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for name, info in registry.items():
            path = info.get("path", os.path.join(root, name))
            if path in seen:
                continue
            seen.add(path)

            n_shards = 0
            if os.path.isdir(path):
                n_shards = sum(
                    1
                    for f in os.listdir(path)
                    if f.startswith("data-") and f.endswith(".arrow")
                )

            exp: dict = {
                "name": name,
                "root": root,
                "path": path,
                "created_at": info.get("created_at"),
                "n_shards": n_shards,
                "metadata": None,
            }

            meta_path = os.path.join(path, "metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        exp["metadata"] = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            experiments.append(exp)

    return experiments


class Server(Interface):
    class Config(BaseModel):
        host: str = "localhost"
        port: int = 5101
        root_dir: str = "./systems/graphs"
        password_file: str | None = None

    def launch(self):
        self.hsds()
        self.file_server()

    def hsds(self):
        #converts relative path to absolute path
        root_dir = os.path.abspath(self.config.root_dir)

        #if cant find absolute path means it cant find file
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # List available systems for the user
        # sort d for d where d is directories within systems folder
        # the join is required as listdir only lists files/folders without absolute path so need to append to root_dir
        # then checks if directory or file 
        systems = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        print(f"Available systems: {', '.join(systems)}")

        # setting environment variables for HSDS subprocess
        env = os.environ.copy()
        env["HSDS_ENDPOINT"] = f"http://{self.config.host}:{self.config.port}"
        env["ROOT_DIR"] = root_dir
        env["BUCKET_NAME"] = root_dir

        # constructing command thjat suplies to process
        cmd = [
            "hsds",
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--root_dir",
            root_dir,
        ]

        # starts hsds server by calling system level command, hsds must be installed system wide
        print(f"Starting HSDS at http://{self.config.host}:{self.config.port}")
        print(f"Serving H5 files from: {root_dir}")
        subprocess.Popen(cmd, env=env)

    def file_server(self, port_offset=1):
        root_dir = os.path.abspath(self.config.root_dir)

        async def list_systems(request):
            systems = sorted(
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
            return web.json_response(systems)

        async def serve_file(request):
            path = request.match_info["path"]
            # Only allow JSON files
            if not path.endswith(".json"):
                return web.Response(status=403)
            filepath = os.path.join(root_dir, path)
            filepath = os.path.realpath(filepath)
            if not filepath.startswith(os.path.realpath(root_dir)):
                return web.Response(status=403)  # path traversal protection
            if not os.path.isfile(filepath):
                return web.Response(status=404)
            return web.FileResponse(filepath)

        async def list_experiments(request):
            return web.json_response(_load_experiments())

        app = web.Application()
        cors = aiohttp_cors.setup(
            app, defaults={"*": aiohttp_cors.ResourceOptions(allow_headers="*")}
        )
        # GET /systems -> ["EI1", "EI2", "S1", ...]
        systems_resource = cors.add(app.router.add_resource("/systems"))
        cors.add(systems_resource.add_route("GET", list_systems))
        # GET /experiments -> list of all experiment dicts across known roots
        experiments_resource = cors.add(app.router.add_resource("/experiments"))
        cors.add(experiments_resource.add_route("GET", list_experiments))
        # GET /files/{system}/{file} -> e.g. /files/EI1/graph.json
        resource = cors.add(app.router.add_resource("/files/{path:.+}"))
        cors.add(resource.add_route("GET", serve_file))

        port = self.config.port + port_offset
        print(f"File server at http://{self.config.host}:{port}/files/")
        web.run_app(app, host=self.config.host, port=port)
