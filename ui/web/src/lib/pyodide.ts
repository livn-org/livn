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
    const wheelBytes = new Uint8Array(await fetch(`/${manifest.filename}`).then(r => r.arrayBuffer()));
    pyodide!.FS.writeFile(`/${manifest.filename}`, wheelBytes);
    await pyodide!.runPythonAsync(`
import os
os.environ['TQDM_DISABLE'] = '1'
import micropip
await micropip.install(['fsspec', 'gymnasium', 'pyfive', 'huggingface_hub', 'httpcore'])
await micropip.install('emfs:///${manifest.filename}', deps=False)
    `);

    onLog('Restoring cached systems…');
    await mountIDBFS();

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
    // proxied through Vite at /hsds/* so h5pyd requests are same-origin
    const params = new URLSearchParams(window.location.search);
    const directEndpoint = params.get('hsds') ?? 'http://localhost:5101';
    const proxyEndpoint = `${window.location.origin}/hsds`;
    try {
        const resp = await fetch(`${proxyEndpoint}/about`);
        if (resp.ok) {
            await pyodide!.runPythonAsync(`
import os, json
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
        // pyfive fallback
    }
    return false;
}

async function mountIDBFS(): Promise<void> {
    if (!pyodide || idbfsMounted) return;
    try {
        const FS = pyodide.FS;
        // predefined() downloads systems to ./systems/graphs/ relative to cwd
        const base = '/home/pyodide/systems';
        const mount = `${base}/graphs`;
        try { FS.mkdir(base); } catch { /* exists */ }
        try { FS.mkdir(mount); } catch { /* exists */ }
        FS.mount(FS.filesystems.IDBFS, {}, mount);
        await new Promise<void>((resolve, reject) => {
            FS.syncfs(true, (err: Error | null) => {
                if (err) reject(err); else resolve();
            });
        });
        idbfsMounted = true;
        try {
            const entries = FS.readdir(mount).filter((e: string) => e !== '.' && e !== '..');
            if (entries.length > 0) logSnap(`IDBFS restored: ${entries.join(', ')}`);
        } catch { /* ignore */ }
    } catch (e) {
        logSnap(`IDBFS mount failed: ${String(e).slice(0, 200)}`);
    }
}

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

    if hasattr(env, 'model') and env.model is not None:
        snap['model'] = to_js({
            'type': type(env.model).__name__
        }, dict_converter=_JsObject.fromEntries)

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

    const manifestUrl = `${serverBase}/dataset_manifest?path=${encodeURIComponent(expPath)}`;
    const mr = await fetch(manifestUrl);
    if (!mr.ok) {
        const body = await mr.json().catch(() => ({ error: `HTTP ${mr.status}` }));
        throw new Error((body as { error?: string }).error ?? `HTTP ${mr.status}`);
    }
    const { files } = await mr.json() as { files: string[] };

    const fsDir = `/datasets/${name}`;
    try { pyodide.FS.mkdir('/datasets'); } catch { /* exists */ }
    try { pyodide.FS.mkdir(fsDir); } catch { /* exists */ }

    for (const file of files) {
        const fileUrl = `${serverBase}/dataset_file?path=${encodeURIComponent(expPath)}&file=${encodeURIComponent(file)}`;
        const bytes = new Uint8Array(await (await fetch(fileUrl)).arrayBuffer());
        pyodide.FS.writeFile(`${fsDir}/${file}`, bytes);
    }

    await pyodide.loadPackage(['pyarrow', 'xxhash', 'lzma']);

    if (!datasetsInstalled) {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install('datasets')
`);
        datasetsInstalled = true;
    }

    // patch out ThreadPoolExecutor — Pyodide has no threads
    await pyodide.runPythonAsync(`
import tqdm.contrib.concurrent as _tcc
_tcc.thread_map = lambda fn, *iters, **kw: list(map(fn, *iters))

import datasets as _ds
loaded_dataset = _ds.load_from_disk(${JSON.stringify(fsDir)})
print(f"loaded_dataset ready: {loaded_dataset.num_rows} rows, features: {list(loaded_dataset.features)}")
del _ds, _tcc
`);
}

export type RowData = {
    duration: number;
    spikes: Record<number, number[]>;
    voltages: Record<number, number[]>;
};

export async function loadExpSystem(sysName: string): Promise<void> {
    if (!pyodide) throw new Error('Pyodide not initialized');
    await pyodide.runPythonAsync(`
from livn.env import Env
from livn.system import predefined
env = Env(predefined(${JSON.stringify(sysName)}))
`);
}

export async function getExpRowData(rowIdx: number, gids: number[]): Promise<RowData> {
    if (!pyodide) throw new Error('Pyodide not initialized');
    const result = await pyodide.runPythonAsync(`
import json as _json
_row  = loaded_dataset[${rowIdx}]
_gset = set(${JSON.stringify(gids)})
_it   = list(_row.get('it') or [])
_tt   = list(_row.get('tt') or [])
_spk  = {g: [] for g in _gset}
for _n, _t in zip(_it, _tt):
    if _n in _spk:
        _spk[_n].append(float(_t))

_iv   = list(_row.get('iv') or [])
_vv   = list(_row.get('vv') or [])
_volt = {g: [] for g in _gset}
for _idx, _g in enumerate(_iv):
    if _g in _volt and _idx < len(_vv):
        _volt[_g] = [float(v) for v in _vv[_idx]]

_json.dumps({
    'duration': int(_row['duration']),
    'spikes':   {str(k): v for k, v in _spk.items()},
    'voltages': {str(k): v for k, v in _volt.items()},
})
`);
    return JSON.parse(result as string) as RowData;
}

let matplotlibReady = false;

export async function exportChartPng(
    gid: number,
    pop: string,
    chartType: 'spikes' | 'voltage',
    spikes: number[],
    voltages: number[],
    duration: number
): Promise<string> {
    if (!pyodide) throw new Error('Pyodide not initialized');
    if (!matplotlibReady) {
        await pyodide.loadPackage('matplotlib');
        matplotlibReady = true;
    }
    const result = await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
import io as _io
import base64 as _b64

_spikes   = ${JSON.stringify(spikes)}
_voltages = ${JSON.stringify(voltages)}
_duration = ${duration}
_gid      = ${gid}
_pop      = ${JSON.stringify(pop)}
_type     = ${JSON.stringify(chartType)}

_dark = '#0d0d1a'
_fig, _ax = _plt.subplots(figsize=(10, 2.5), facecolor=_dark)
_ax.set_facecolor(_dark)
for _sp in _ax.spines.values(): _sp.set_color('#444')
_ax.tick_params(colors='#888')
_ax.set_xlabel('Time (ms)', color='#888')
_ax.xaxis.label.set_color('#888')
_ax.yaxis.label.set_color('#888')

if _type == 'spikes':
    _col = '#4fc3f7' if _pop == 'EXC' else '#ef5350' if _pop == 'INH' else '#aaaaaa'
    for _t in _spikes:
        _ax.axvline(_t, color=_col, linewidth=0.8, alpha=0.85)
    _ax.set_xlim(0, _duration)
    _ax.set_ylim(0, 1)
    _ax.set_yticks([])
    _ax.set_title(f'GID {_gid} ({_pop}) — Spikes', color='#ccc', fontsize=11)
else:
    import numpy as _np
    _t_ax = _np.linspace(0, _duration, len(_voltages))
    _ax.plot(_t_ax, _voltages, color='#ffb74d', linewidth=0.9)
    _ax.set_xlim(0, _duration)
    _ax.set_ylabel('mV', color='#888')
    _ax.set_title(f'GID {_gid} ({_pop}) — Voltage', color='#ccc', fontsize=11)

_plt.tight_layout()
_buf = _io.BytesIO()
_fig.savefig(_buf, format='png', dpi=150, bbox_inches='tight', facecolor=_dark)
_plt.close(_fig)
_buf.seek(0)
_b64.b64encode(_buf.getvalue()).decode()
`);
    return `data:image/png;base64,${result as string}`;
}

export type ElectrodeData = {
    duration: number;
    hasLfp: boolean;
    lfp: number[];
    spikeTimes: number[];
};

export async function getElectrodeData(rowIdx: number, electrodeId: number): Promise<ElectrodeData> {
    if (!pyodide) throw new Error('Pyodide not initialized');
    const result = await pyodide.runPythonAsync(`
import json as _json
import numpy as _np

_row        = loaded_dataset[${rowIdx}]
_duration   = int(_row['duration'])
_eid        = ${electrodeId}

_it = _np.array(list(_row.get('it') or []))
_tt = _np.array(list(_row.get('tt') or []))
try:
    _cit, _ct = env.channel_recording(_it, _tt)
    _spike_times = [float(t) for t in _ct.get(int(_eid), [])]
except Exception:
    _spike_times = []

_has_mp = bool('mp' in _row and _row['mp'] is not None and len(_row.get('mp') or []) > 0)
_lfp = []
if _has_mp:
    try:
        _mp      = _np.array(_row['mp'])
        _lfp_all = env.potential_recording(_mp)
        _ch_ids  = list(env.io.channel_ids)
        _ch_idx  = _ch_ids.index(int(_eid))
        _trace   = _np.nan_to_num(_lfp_all[_ch_idx], nan=0.0, posinf=0.0, neginf=0.0)
        _lfp     = [float(v) for v in _trace]
    except Exception:
        _has_mp = False

_json.dumps({
    'duration':   _duration,
    'hasLfp':     bool(_has_mp),
    'lfp':        _lfp,
    'spikeTimes': _spike_times,
})
`);
    return JSON.parse(result as string) as ElectrodeData;
}

export async function getAllRowSpikes(rowIdx: number): Promise<{ it: number[]; tt: number[]; duration: number }> {
    if (!pyodide) throw new Error('Pyodide not initialized');
    const result = await pyodide.runPythonAsync(`
import json as _json
_row = loaded_dataset[${rowIdx}]
_json.dumps({
    'duration': int(_row['duration']),
    'it': [int(x) for x in (_row.get('it') or [])],
    'tt': [float(x) for x in (_row.get('tt') or [])],
})
`);
    return JSON.parse(result as string);
}

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

    let snapshot: EnvSnapshot | null = null;
    let snapshotError: string | null = null;
    try {
        snapshot = await snapshotEnv();
    } catch (e) {
        snapshotError = String(e);
    }

    await syncFSToDisk();

    const elapsed = performance.now() - start;
    lastExecTime.set(elapsed);
    loading.set(false);

    if (snapshotError) {
        const msg = `\n[snapshot error] ${snapshotError}`;
        error = error ? error + msg : msg;
    }
    if (error) lastError.set(error);

    updateStores(snapshot);

    return { output, error, snapshot };
}
