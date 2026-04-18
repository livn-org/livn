import subprocess
import os

from aiohttp import web
import aiohttp_cors

from machinable import Interface
from pydantic import BaseModel


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
        root_dir = os.path.abspath(self.config.root_dir)

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # List available systems for the user
        systems = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        print(f"Available systems: {', '.join(systems)}")

        env = os.environ.copy()
        env["HSDS_ENDPOINT"] = f"http://{self.config.host}:{self.config.port}"
        env["ROOT_DIR"] = root_dir
        env["BUCKET_NAME"] = root_dir

        cmd = [
            "hsds",
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--root_dir",
            root_dir,
        ]

        print(f"Starting HSDS at http://{self.config.host}:{self.config.port}")
        print(f"Serving H5 files from: {root_dir}")
        subprocess.run(cmd, env=env)

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

        app = web.Application()
        cors = aiohttp_cors.setup(
            app, defaults={"*": aiohttp_cors.ResourceOptions(allow_headers="*")}
        )
        # GET /systems -> ["EI1", "EI2", "S1", ...]
        systems_resource = cors.add(app.router.add_resource("/systems"))
        cors.add(systems_resource.add_route("GET", list_systems))
        # GET /files/{system}/{file} -> e.g. /files/EI1/graph.json
        resource = cors.add(app.router.add_resource("/files/{path:.+}"))
        cors.add(resource.add_route("GET", serve_file))

        port = self.config.port + port_offset
        print(f"File server at http://{self.config.host}:{port}/files/")
        web.run_app(app, host=self.config.host, port=port)
