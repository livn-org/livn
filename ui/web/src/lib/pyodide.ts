import { loadPyodide as _loadPyodide, type PyodideInterface } from 'pyodide';
import type { EnvSnapshot } from './types';
import { pyodideReady, hsdsConnected, backendInfo, loading, lastError, lastExecTime, updateStores, snapshotLog } from './stores';

let pyodide: PyodideInterface | null = null;
let idbfsMounted = false;

function logSnap(msg: string) {
    const ts = new Date().toLocaleTimeString();
    snapshotLog.update((log) => [...log.slice(-19), `[${ts}] ${msg}`]);
}

export async function initPyodide(onLog: (msg: string) => void): Promise<void> {
    if (pyodide) {
        onLog('Pyodide already loaded');
        pyodideReady.set(true);
        return;
    }

    onLog('Loading Pyodide runtime…');

    // Register service worker to cache pyodide assets across page loads
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(() => { });
    }

    pyodide = await _loadPyodide({
        indexURL: '/pyodide/'
    });

    await pyodide!.loadPackage('micropip');
    await pyodide!.loadPackage(['scipy', 'pandas', 'pydantic']);

    onLog('Installing livn…');
    const manifest = await fetch('/wheel.json').then(r => r.json());
    // Fetch wheel bytes in JS and write to Pyodide's virtual filesystem
    const wheelBytes = new Uint8Array(await fetch(`/${manifest.filename}`).then(r => r.arrayBuffer()));
    pyodide!.FS.writeFile(`/${manifest.filename}`, wheelBytes);
    await pyodide!.runPythonAsync(`
import os
os.environ['TQDM_DISABLE'] = '1'
import micropip
await micropip.install(['fsspec', 'gymnasium', 'pyfive', 'huggingface_hub', 'httpcore'])
await micropip.install('emfs:///${manifest.filename}', deps=False)
    `);

    // Mount IDBFS at /systems/graphs to persist downloaded systems across page loads
    onLog('Restoring cached systems…');
    await mountIDBFS();

    // Probe HSDS
    if (await tryConfigureHSDS()) {
        onLog('HSDS backend connected');
        hsdsConnected.set(true);
        backendInfo.set('HSDS');
    } else {
        onLog('Using pyfive backend (local)');
        backendInfo.set('pyfive (local)');
    }

    pyodideReady.set(true);
    onLog('Ready');
}

async function tryConfigureHSDS(): Promise<boolean> {
    // HSDS is proxied through Vite at /hsds/* → localhost:5101/*
    // This makes h5pyd requests same-origin, avoiding CORS/Wasm issues.
    const params = new URLSearchParams(window.location.search);
    const directEndpoint = params.get('hsds') ?? 'http://localhost:5101';
    const proxyEndpoint = `${window.location.origin}/hsds`;
    try {
        const resp = await fetch(`${proxyEndpoint}/about`);
        if (resp.ok) {
            // Patch Pyodide's urllib to use browser fetch, then configure h5pyd
            await pyodide!.runPythonAsync(`
import os, json
# Use the same-origin proxy endpoint so h5pyd requests go through Vite
os.environ["LIVN_HSDS"] = json.dumps({
    "endpoint": "${proxyEndpoint}",
    "username": "admin",
    "password": "admin",
    "files_endpoint": "${window.location.origin}/files"
})
`);
            await pyodide!.runPythonAsync(`
import micropip
await micropip.install('h5pyd')
`);
            return true;
        }
    } catch {
        // HSDS not available — pyfive fallback
    }
    return false;
}

/** Mount an IndexedDB-backed FS at the systems cache directory */
async function mountIDBFS(): Promise<void> {
    if (!pyodide || idbfsMounted) return;
    try {
        const FS = pyodide.FS;
        // Pyodide cwd is /home/pyodide; predefined() downloads to ./systems/graphs/
        const base = '/home/pyodide/systems';
        const mount = `${base}/graphs`;
        try { FS.mkdir(base); } catch { /* exists */ }
        try { FS.mkdir(mount); } catch { /* exists */ }
        FS.mount(FS.filesystems.IDBFS, {}, mount);
        // Populate from IndexedDB (true = load from persistent store)
        await new Promise<void>((resolve, reject) => {
            FS.syncfs(true, (err: Error | null) => {
                if (err) reject(err); else resolve();
            });
        });
        idbfsMounted = true;
        // Log what was restored
        try {
            const entries = FS.readdir(mount).filter((e: string) => e !== '.' && e !== '..');
            if (entries.length > 0) {
                logSnap(`IDBFS restored: ${entries.join(', ')}`);
            }
        } catch { /* ignore */ }
    } catch (e) {
        logSnap(`IDBFS mount failed: ${String(e).slice(0, 200)}`);
    }
}

/** Flush in-memory FS changes to IndexedDB */
async function syncFSToDisk(): Promise<void> {
    if (!pyodide || !idbfsMounted) return;
    try {
        await new Promise<void>((resolve, reject) => {
            pyodide!.FS.syncfs(false, (err: Error | null) => {
                if (err) reject(err); else resolve();
            });
        });
    } catch { /* best-effort */ }
}

async function snapshotEnv(): Promise<EnvSnapshot | null> {
    logSnap('snapshotEnv() called');
    try {
        const result = await pyodide!.runPythonAsync(`
import json as _json
import numpy as _np
from js import Object as _JsObject
from pyodide.ffi import to_js

def _snapshot():
    env = globals().get('env')
    if env is None:
        return 'NO_ENV'

    snap = {}
    _diag = []

    # System
    if hasattr(env, 'system') and env.system is not None:
        s = env.system
        coords = {}
        for pop in s.populations:
            arr = _np.ascontiguousarray(s.coordinate_array(pop), dtype=_np.float64).flatten()
            coords[pop] = to_js(arr)

        snap['system'] = to_js({
            'name': s.name,
            'populations': to_js(list(s.populations)),
            'num_neurons': int(s.num_neurons),
            'bounding_box': to_js(_np.ascontiguousarray(s.bounding_box, dtype=_np.float64).flatten()),
            'pop_coords': to_js(coords),
        }, dict_converter=_JsObject.fromEntries)
        _diag.append(f'system={s.name} neurons={s.num_neurons}')
    else:
        _diag.append('system=None')

    # IO
    if hasattr(env, 'io') and env.io is not None:
        io_obj = env.io
        snap['io'] = to_js({
            'type': type(io_obj).__name__,
            'num_channels': int(io_obj.num_channels),
            'electrode_coordinates': to_js(
                _np.ascontiguousarray(io_obj.electrode_coordinates, dtype=_np.float64).flatten()
            ),
        }, dict_converter=_JsObject.fromEntries)
        _diag.append(f'io={type(io_obj).__name__} channels={io_obj.num_channels}')
    else:
        _diag.append('io=None')

    # Model
    if hasattr(env, 'model') and env.model is not None:
        snap['model'] = to_js({
            'type': type(env.model).__name__
        }, dict_converter=_JsObject.fromEntries)

    # Return diagnostic string if snap is empty
    if not snap:
        return 'EMPTY:' + ','.join(_diag)

    return to_js(snap, dict_converter=_JsObject.fromEntries)

_snapshot()
	`);

        if (typeof result === 'string') {
            logSnap(`snapshot returned string: ${result}`);
            return null;
        }

        logSnap(`snapshot OK: keys=[${result ? Object.keys(result).join(',') : 'null'}]`);
        return result as EnvSnapshot | null;
    } catch (e) {
        logSnap(`snapshot ERROR: ${String(e).slice(0, 500)}`);
        throw e;
    }
}

let datasetsInstalled = false;

export async function loadHFDataset(
    name: string,
    expPath: string,
    serverBase: string
): Promise<void> {
    if (!pyodide) throw new Error('Pyodide not initialized');

    // 1. Fetch manifest (size-checked by server)
    const manifestUrl = `${serverBase}/dataset_manifest?path=${encodeURIComponent(expPath)}`;
    const mr = await fetch(manifestUrl);
    if (!mr.ok) {
        const body = await mr.json().catch(() => ({ error: `HTTP ${mr.status}` }));
        throw new Error((body as { error?: string }).error ?? `HTTP ${mr.status}`);
    }
    const { files } = await mr.json() as { files: string[] };

    // 2. Write dataset files to Pyodide FS
    const fsDir = `/datasets/${name}`;
    try { pyodide.FS.mkdir('/datasets'); } catch { /* exists */ }
    try { pyodide.FS.mkdir(fsDir); } catch { /* exists */ }

    for (const file of files) {
        const fileUrl = `${serverBase}/dataset_file?path=${encodeURIComponent(expPath)}&file=${encodeURIComponent(file)}`;
        const bytes = new Uint8Array(await (await fetch(fileUrl)).arrayBuffer());
        pyodide.FS.writeFile(`${fsDir}/${file}`, bytes);
    }

    // 3. Ensure pyarrow and xxhash prebuilt packages are loaded
    await pyodide.loadPackage(['pyarrow', 'xxhash', 'lzma']);

    // 4. Install datasets if not already done (lazy, first-use only)
    if (!datasetsInstalled) {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install('datasets')
`);
        datasetsInstalled = true;
    }

    // 5. Load from disk — patch out ThreadPoolExecutor first (Pyodide has no threads)
    await pyodide.runPythonAsync(`
import tqdm.contrib.concurrent as _tcc
_tcc.thread_map = lambda fn, *iters, **kw: list(map(fn, *iters))

import datasets as _ds
loaded_dataset = _ds.load_from_disk(${JSON.stringify(fsDir)})
print(f"loaded_dataset ready: {loaded_dataset.num_rows} rows, features: {list(loaded_dataset.features)}")
del _ds, _tcc
`);
}

/** Force a manual snapshot refresh and store update */
export async function forceRefresh(): Promise<void> {
    logSnap('forceRefresh() triggered');
    try {
        const snapshot = await snapshotEnv();
        updateStores(snapshot);
        logSnap('forceRefresh() done');
    } catch (e) {
        logSnap(`forceRefresh() error: ${String(e).slice(0, 200)}`);
    }
}

export async function executeCode(code: string): Promise<{
    output: string;
    error: string | null;
    snapshot: EnvSnapshot | null;
}> {
    loading.set(true);
    lastError.set(null);
    const start = performance.now();

    // Redirect stdout/stderr
    await pyodide!.runPythonAsync(`
import sys as _sys, io as _io
_stdout_capture = _io.StringIO()
_stderr_capture = _io.StringIO()
_sys.stdout = _stdout_capture
_sys.stderr = _stderr_capture
	`);

    let error: string | null = null;
    try {
        const result = await pyodide!.runPythonAsync(code);
        // If the code returned a value, capture its repr
        if (result !== undefined && result !== null) {
            const repr = String(result);
            if (repr && repr !== 'None') {
                await pyodide!.runPythonAsync(`_stdout_capture.write(${JSON.stringify(repr)} + "\\n")`);
            }
        }
    } catch (e) {
        error = String(e);
    }

    const output = String(
        await pyodide!.runPythonAsync(`
_sys.stdout = _sys.__stdout__
_sys.stderr = _sys.__stderr__
_stdout_capture.getvalue() + _stderr_capture.getvalue()
		`)
    );

    // Always snapshot after execution
    let snapshot: EnvSnapshot | null = null;
    let snapshotError: string | null = null;
    try {
        snapshot = await snapshotEnv();
    } catch (e) {
        snapshotError = String(e);
    }

    // Persist any newly downloaded system files to IndexedDB
    await syncFSToDisk();

    const elapsed = performance.now() - start;
    lastExecTime.set(elapsed);
    loading.set(false);

    // Surface snapshot errors alongside execution errors
    if (snapshotError) {
        const msg = `\n[snapshot error] ${snapshotError}`;
        error = error ? error + msg : msg;
    }
    if (error) lastError.set(error);

    updateStores(snapshot);

    return { output, error, snapshot };
}
