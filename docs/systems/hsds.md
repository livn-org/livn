# Using HSDS

livn supports serving system data over HTTP using [HSDS](https://github.com/HDFGroup/hsds) (Highly Scalable Data Service), an object-store-based access layer for HDF5 files. This enables browser-based access to systems via [h5pyd](https://github.com/HDFGroup/h5pyd).

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

In the browser, h5pyd's HTTP requests go through the browser's `fetch()` API via [pyodide-http](https://github.com/nicola-rig/pyodide-http) patching.

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
