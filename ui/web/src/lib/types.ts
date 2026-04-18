export interface SystemData {
    name: string;
    populations: string[];
    num_neurons: number;
    bounding_box: Float64Array; // [2×3] flattened: [xmin, ymin, zmin, xmax, ymax, zmax]
    pop_coords: Record<string, Float64Array>; // per-pop [gid, x, y, z] interleaved
}

export interface IOData {
    type: string; // "MEA", "LightArray", etc.
    num_channels: number;
    electrode_coordinates: Float64Array; // [n×4] flattened [id, x, y, z]
}

export interface ModelData {
    type: string; // "RCSD", "LIF", etc.
}

export interface DecodingData {
    spike_ids: Int32Array | null;
    spike_times: Float64Array | null;
    voltage_ids: Int32Array | null;
    voltage_traces: Float64Array | null; // [n_neurons × timestep] flattened
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
