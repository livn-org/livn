import { writable } from 'svelte/store';
import type { SystemData, IOData, ModelData, DecodingData, ViewConfig, TooltipData, EnvSnapshot, Experiment } from './types';

export const pyodideReady = writable(false);
export const hsdsConnected = writable(false);

export const envSystem = writable<SystemData | null>(null);
export const envIO = writable<IOData | null>(null);
export const envModel = writable<ModelData | null>(null);
export const envDecoding = writable<DecodingData | null>(null);

export const viewConfig = writable<ViewConfig>({
    popVisibility: {},
    pointSize: 1.0,
    opacity: 0.85,
    showBoundingBox: true,
    showElectrodes: true
});

export const consoleHistory = writable<string[]>([]);
export const loading = writable(false);
export const lastError = writable<string | null>(null);

export const tooltip = writable<TooltipData>({
    visible: false,
    gid: 0,
    population: '',
    x: 0,
    y: 0,
    z: 0,
    nearestElectrode: null
});

export const backendInfo = writable<string>('Initializing…');
export const lastExecTime = writable<number | null>(null);

export const snapshotLog = writable<string[]>([]);
export const pendingCommand = writable<string | null>(null);

export const datasetLoading = writable<boolean>(false);
export const datasetError   = writable<string | null>(null);

export const selectedNeurons  = writable<number[]>([]);
export const activeExperiment = writable<Experiment | null>(null);
export const activeExpRow     = writable<number>(0);
export const selectedElectrode = writable<number | null>(null);
function logSnapshot(msg: string) {
    const ts = new Date().toLocaleTimeString();
    snapshotLog.update((log) => [...log.slice(-19), `[${ts}] ${msg}`]);
}

let prevSnapshot: EnvSnapshot | null = null;

export function updateStores(snapshot: EnvSnapshot | null) {
    if (!snapshot) {
        logSnapshot('updateStores called with null snapshot');
        return;
    }

    const keys = Object.keys(snapshot);
    logSnapshot(`updateStores: keys=[${keys.join(',')}]`);

    if (snapshot.system !== prevSnapshot?.system) {
        logSnapshot(`system changed: ${snapshot.system ? snapshot.system.name + ' (' + snapshot.system.num_neurons + ' neurons)' : 'null'}`);
        envSystem.set(snapshot.system);
        if (snapshot.system) {
            viewConfig.update((vc) => ({
                ...vc,
                popVisibility: Object.fromEntries(
                    snapshot.system!.populations.map((p) => [p, vc.popVisibility[p] ?? true])
                )
            }));
        }
    } else {
        logSnapshot('system unchanged');
    }
    if (snapshot.io !== prevSnapshot?.io) {
        logSnapshot(`io changed: ${snapshot.io ? snapshot.io.type : 'null'}`);
        envIO.set(snapshot.io);
    }
    if (snapshot.model !== prevSnapshot?.model) envModel.set(snapshot.model);
    if (snapshot.decoding !== prevSnapshot?.decoding) envDecoding.set(snapshot.decoding);

    prevSnapshot = snapshot;
}
