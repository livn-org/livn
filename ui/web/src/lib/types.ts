export interface SystemData {
    name: string;
    populations: string[];
    num_neurons: number;
    bounding_box: Float64Array; // [xmin, ymin, zmin, xmax, ymax, zmax]
    pop_coords: Record<string, Float64Array>; // [gid, x, y, z] interleaved per pop
}

export interface IOData {
    type: string;
    num_channels: number;
    electrode_coordinates: Float64Array; // [id, x, y, z] interleaved
}

export interface ModelData {
    type: string;
}

export interface DecodingData {
    spike_ids: Int32Array | null;
    spike_times: Float64Array | null;
    voltage_ids: Int32Array | null;
    voltage_traces: Float64Array | null;
    duration: number;
    dt: number;
}

export interface EnvSnapshot {
    system: SystemData | null;
    io: IOData | null;
    model: ModelData | null;
    decoding: DecodingData | null;
}

export interface ViewConfig {
    popVisibility: Record<string, boolean>;
    pointSize: number;
    opacity: number;
    showBoundingBox: boolean;
    showElectrodes: boolean;
}

export interface TooltipData {
    visible: boolean;
    gid: number;
    population: string;
    x: number;
    y: number;
    z: number;
    nearestElectrode: { id: number; distance: number } | null;
}

export type ExpMeta = {
    duration?: number;
    system?: { uri?: string; populations?: string[]; n_neurons?: number };
    encoding?: Record<string, unknown>;
    model?: string;
    recording?: { spikes?: boolean; voltages?: boolean; membrane_currents?: boolean };
};

export type Experiment = {
    name: string;
    root: string;
    path: string;
    created_at: string | null;
    n_shards: number;
    metadata: ExpMeta | null;
};
