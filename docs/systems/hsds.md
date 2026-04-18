# Using HSDS

livn supports serving system data over HTTP using [HSDS](https://github.com/HDFGroup/hsds) (Highly Scalable Data Service), an object-store-based access layer for HDF5 files. This enables browser-based access to systems via [h5pyd](https://github.com/HDFGroup/h5pyd) and is the backbone of the [WebUI](/ui).

::: tip
HSDS is optional. Without it, livn defaults to the pyfive backend which reads H5 files directly from disk (or the Pyodide in-memory filesystem). HSDS is only needed when you want to serve systems to the browser-based UI or to remote clients.
:::

## Architecture

livn uses a three-tier backend selection for reading HDF5 system files:

| Priority | Backend | When used | Use case |
|----------|---------|-----------|----------|
| 1 | **h5pyd** (HSDS) | `LIVN_HSDS` env var set + h5pyd installed | Browser UI, remote access |
| 2 | **neuroh5** (MPI) | neuroh5 + mpi4py installed | HPC parallel simulations |
| 3 | **pyfive** | Always available | Local development, Pyodide fallback |

The h5pyd and pyfive backends share identical read logic where only the file opener differs. This means all backends produce the same results.

## Starting the HSDS Server

The `livn ui server` command launches an HSDS instance that serves all system files from `systems/graphs/`:

```bash
livn ui server --launch
```

This starts HSDS on `http://localhost:5101` with `systems/graphs/` as its storage root. Systems must be [imported into HSDS](#data-ingestion) before they can be accessed — the server does not read raw `.h5` files directly.

### Custom configuration

```bash
# Custom port and root directory
livn ui server port=5102 root_dir=/data/my_systems --launch
```

### Multi-system support

The server's `root_dir` points at a directory containing one subdirectory per system:

```
systems/graphs/
├── EI1/
│   ├── cells.h5
│   ├── connections.h5
│   └── graph.json
├── EI2/
│   └── ...
└── CA1/
    └── ...
```

HSDS maps each imported file to a domain path — e.g. after importing `EI1/cells.h5`, it becomes the HSDS domain `/EI1/cells.h5`. System selection happens by opening the appropriate domain, not by configuring the server.

### Companion file server

Alongside HSDS, a companion HTTP server runs on port + 1 (default `5102`) to serve non-H5 files like `graph.json`:

| Endpoint | Description |
|----------|-------------|
| `GET /systems` | List available system names |
| `GET /files/{system}/{file}` | Serve a JSON file (e.g. `/files/EI1/graph.json`) |

The file server includes path traversal protection and CORS headers for browser access.

## Data Ingestion

HSDS does not serve raw HDF5 files directly — it uses its own internal chunked storage format. Before a system can be accessed via HSDS, its `.h5` files must be imported using the `hsload` utility that ships with [h5pyd](https://github.com/HDFGroup/h5pyd):

```bash
# Import a single file
hsload systems/graphs/EI1/cells.h5 /EI1/cells.h5 \
  --endpoint http://localhost:5101 -u admin -p admin

# Import all .h5 files for a system
for f in systems/graphs/EI1/*.h5; do
  name=$(basename "$f")
  hsload "$f" "/EI1/$name" --endpoint http://localhost:5101 -u admin -p admin
done
```

To bulk-import every system:

```bash
for sys in systems/graphs/*/; do
  name=$(basename "$sys")
  for f in "$sys"*.h5; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    hsload "$f" "/$name/$fname" --endpoint http://localhost:5101 -u admin -p admin
  done
done
```

See the [HSDS documentation on data loading](https://github.com/HDFGroup/hsds/blob/master/docs/post_install.md) for more details on `hsload` options and alternative ingestion methods.

::: warning
JSON sidecar files like `graph.json` and `mea.json` are **not** stored in HSDS. They are served by the companion file server (port 5102) or the Vite dev server's `/files/` endpoint.
:::

## Configuring the Client

Set the `LIVN_HSDS` environment variable to connect to an HSDS server:

```bash
export LIVN_HSDS='{"endpoint": "http://localhost:5101"}'
```

The value is a JSON string with the following fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `endpoint` | Yes | — | HSDS server URL |
| `bucket` | No | `/data` | Root path / bucket in HSDS |

Once set, all livn system reads will go through HSDS instead of local files:

```python
import os
os.environ["LIVN_HSDS"] = '{"endpoint": "http://localhost:5101"}'

from livn.system import predefined, System

path = predefined("EI1")
system = System(path)
print(system.populations)  # reads from HSDS
```

::: info
`LIVN_HSDS` configures the server connection, not a specific system. The system is selected by the file path passed to `System()` — the same code works for any system served by the HSDS instance.
:::

## Browser Usage (Pyodide + h5pyd)

In the browser, h5pyd's HTTP requests go through the browser's `fetch()` API via [pyodide-http](https://github.com/nicola-rig/pyodide-http) patching. This is set up automatically by the [WebUI](/ui).

To use HSDS manually in a Pyodide environment:

```python
import micropip
await micropip.install(["h5pyd", "pyodide-http"])

import pyodide_http
pyodide_http.patch_all()

import h5pyd

# Connect to HSDS
f = h5pyd.File("/EI1/cells.h5", "r", endpoint="http://localhost:5101")
print(list(f["Populations"].keys()))
```

### CORS

The HSDS server must allow cross-origin requests from the browser. The `livn ui server` command configures CORS automatically. If you run HSDS manually, ensure the appropriate `Access-Control-Allow-Origin` headers are set.

## Verifying the Setup

### Server-side test

With HSDS running, test connectivity from Python:

```python
import h5pyd

f = h5pyd.File("/EI1/cells.h5", "r", endpoint="http://localhost:5101")
print("Populations:", list(f["Populations"].keys()))
print("Ranges:", f["H5Types/Populations"][:])
f.close()
```

