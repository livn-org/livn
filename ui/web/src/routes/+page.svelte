<script lang="ts">
    import "../app.css";
    import { Canvas } from "@threlte/core";
    import EnvScene from "$lib/components/EnvScene.svelte";
    import Console from "$lib/components/Console.svelte";
    import StatusBar from "$lib/components/StatusBar.svelte";
    import Tooltip from "$lib/components/Tooltip.svelte";
    import NavBar from "$lib/components/NavBar.svelte";
    import SimSystemList from "$lib/components/SimSystemList.svelte";
    import BioRecordingList from "$lib/components/BioRecordingList.svelte";
    import BioRecordingDetail from "$lib/components/BioRecordingDetail.svelte";
    import { viewConfig, envSystem, pendingCommand, pyodideReady } from "$lib/stores";
    import SystemGenerator from "$lib/components/SystemGenerator.svelte"; 

    // ── Navigation state (flat primitives avoid TS union-narrowing issues) ──
    let navTab       = $state<'bio' | 'sim' | 'build'>('sim');
    let navPage      = $state<'list' | 'detail'>('detail');
    let navSystem    = $state('EI1');
    let navRecording = $state('');

    let buildSubTab  = $state<'system' | 'stim'>('system');

    const activeTab        = $derived(navTab);
    const isSimDetail      = $derived(navTab === 'sim' && navPage === 'detail');
    const currentSystem    = $derived(isSimDetail ? navSystem : null);
    const currentRecording = $derived(navTab === 'bio' && navPage === 'detail' ? navRecording : null);

    // Auto-load the default system once Pyodide is ready
    let autoLoadDone = $state(false);
    $effect(() => {
        if ($pyodideReady && !autoLoadDone && isSimDetail) {
            autoLoadDone = true;
            pendingCommand.set(
                `from livn.env import Env\nfrom livn.system import predefined\nenv = Env(predefined('${navSystem}'))`
            );
        }
    });

    function setTab(tab: 'bio' | 'sim' | 'build') {
        if (tab === navTab) return;
        navTab  = tab;
        navPage = 'list';
    }

    function selectSystem(system: string) {
        navTab    = 'sim';
        navPage   = 'detail';
        navSystem = system;
        pendingCommand.set(
            `from livn.env import Env\nfrom livn.system import predefined\nenv = Env(predefined('${system}'))`
        );
    }

    function selectRecording(recording: string) {
        navTab       = 'bio';
        navPage      = 'detail';
        navRecording = recording;
    }

    // ── Viz controls ──────────────────────────────────────────────────────
    const config = $derived($viewConfig);
    const system = $derived($envSystem);

    function togglePop(pop: string) {
        viewConfig.update((vc) => ({
            ...vc,
            popVisibility: { ...vc.popVisibility, [pop]: !vc.popVisibility[pop] },
        }));
    }
    function setPointSize(e: Event) {
        const val = parseFloat((e.target as HTMLInputElement).value);
        viewConfig.update((vc) => ({ ...vc, pointSize: val }));
    }
    function setOpacity(e: Event) {
        const val = parseFloat((e.target as HTMLInputElement).value);
        viewConfig.update((vc) => ({ ...vc, opacity: val }));
    }
    function toggleBBox() {
        viewConfig.update((vc) => ({ ...vc, showBoundingBox: !vc.showBoundingBox }));
    }
    function toggleElectrodes() {
        viewConfig.update((vc) => ({ ...vc, showElectrodes: !vc.showElectrodes }));
    }

    // ── Info panels (Sim detail) ──────────────────────────────────────────
    let setupOpen      = $state(false);
    let neuronInfoOpen = $state(false);

    const SHAPES: Record<string, string> = {
        EI1: 'Circular', EI2: 'Circular', CA1d: 'Rectangular',
    };

    function bboxDimensions(bb: Float64Array): string {
        return `${(bb[3]-bb[0]).toFixed(0)} × ${(bb[4]-bb[1]).toFixed(0)} × ${(bb[5]-bb[2]).toFixed(0)} µm`;
    }

    function popCounts(popCoords: Record<string, Float64Array>): string {
        return Object.entries(popCoords)
            .map(([pop, arr]) => `${pop}: ${arr.length / 4}`)
            .join(', ');
    }

    function sampleNeurons(popCoords: Record<string, Float64Array>, n = 10) {
        const out: Array<{ pop: string; gid: number; x: string; y: string; z: string }> = [];
        for (const [pop, arr] of Object.entries(popCoords)) {
            for (let i = 0; i < arr.length && out.length < n; i += 4) {
                out.push({
                    pop,
                    gid: Math.round(arr[i]),
                    x: arr[i + 1].toFixed(1),
                    y: arr[i + 2].toFixed(1),
                    z: arr[i + 3].toFixed(1),
                });
            }
        }
        return out;
    }

    const shape      = $derived(currentSystem ? (SHAPES[currentSystem] ?? 'Unknown') : '—');
    const dimensions = $derived(system ? bboxDimensions(system.bounding_box) : '—');
    const counts     = $derived(system ? popCounts(system.pop_coords) : '—');
    const neurons    = $derived(system ? sampleNeurons(system.pop_coords) : []);
</script>

<div class="layout">
    <NavBar {activeTab} onTabChange={setTab} />

    <div class="content">
        {#if isSimDetail}
            <div class="sim-detail">
                <!-- 3D scene panel -->
                <div class="scene-panel">
                    <Canvas><EnvScene /></Canvas>
                    <Tooltip />

                    {#if system}
                        <!-- Viz controls — top-left -->
                        <div class="controls">
                            <div class="control-group">
                                <span class="control-label">Populations</span>
                                {#each system.populations as pop (pop)}
                                    <label class="toggle">
                                        <input
                                            type="checkbox"
                                            checked={config.popVisibility[pop] ?? true}
                                            onchange={() => togglePop(pop)}
                                        />
                                        {pop}
                                    </label>
                                {/each}
                            </div>
                            <div class="control-group">
                                <label class="slider-label">
                                    Size
                                    <input type="range" min="0.2" max="3" step="0.1"
                                        value={config.pointSize} oninput={setPointSize} />
                                </label>
                                <label class="slider-label">
                                    Opacity
                                    <input type="range" min="0.1" max="1" step="0.05"
                                        value={config.opacity} oninput={setOpacity} />
                                </label>
                            </div>
                            <div class="control-group">
                                <label class="toggle">
                                    <input type="checkbox" checked={config.showBoundingBox} onchange={toggleBBox} />
                                    Bounding box
                                </label>
                                <label class="toggle">
                                    <input type="checkbox" checked={config.showElectrodes} onchange={toggleElectrodes} />
                                    Electrodes
                                </label>
                            </div>
                        </div>
                    {/if}

                    <!-- Info panels — top-right -->
                    <div class="info-overlay">
                        <!-- Setup -->
                        <div class="info-section">
                            <button class="info-hdr" onclick={() => (setupOpen = !setupOpen)}>
                                Setup <span class="chevron" class:open={setupOpen}>▶</span>
                            </button>
                            {#if setupOpen}
                                <div class="info-body">
                                    <div class="info-row"><span>Shape</span><span>{shape}</span></div>
                                    <div class="info-row"><span>Dimensions</span><span>{dimensions}</span></div>
                                    <div class="info-row"><span>Populations</span><span>{counts}</span></div>
                                    {#if system}
                                        <div class="info-row"><span>Total neurons</span><span>{system.num_neurons}</span></div>
                                    {/if}
                                </div>
                            {/if}
                        </div>

                        <!-- Neuron Info -->
                        <div class="info-section">
                            <button class="info-hdr" onclick={() => (neuronInfoOpen = !neuronInfoOpen)}>
                                Neuron Info <span class="chevron" class:open={neuronInfoOpen}>▶</span>
                            </button>
                            {#if neuronInfoOpen}
                                <div class="info-body">
                                    {#if neurons.length === 0}
                                        <div class="info-empty">Load a system to see neuron data.</div>
                                    {:else}
                                        <table class="neuron-table">
                                            <thead>
                                                <tr><th>GID</th><th>Pop</th><th>x</th><th>y</th><th>z</th></tr>
                                            </thead>
                                            <tbody>
                                                {#each neurons as n (n.gid)}
                                                    <tr>
                                                        <td>{n.gid}</td><td>{n.pop}</td>
                                                        <td>{n.x}</td><td>{n.y}</td><td>{n.z}</td>
                                                    </tr>
                                                {/each}
                                            </tbody>
                                        </table>
                                        {#if system && system.num_neurons > 10}
                                            <div class="info-note">
                                                Showing first 10 of {system.num_neurons} neurons
                                            </div>
                                        {/if}
                                    {/if}
                                </div>
                            {/if}
                        </div>

                    </div>
                </div>

                <!-- Console panel -->
                <div class="console-panel">
                    <div class="console-header">
                        <span class="console-system">{currentSystem ?? '—'}</span>
                        <button class="list-btn" onclick={() => { navTab = 'sim'; navPage = 'list'; }}>
                            ← Systems
                        </button>
                    </div>
                    <Console />
                </div>
            </div>

        {:else if navTab === 'sim' && navPage === 'list'}
            <SimSystemList onSelect={selectSystem} />

        {:else if navTab === 'bio' && navPage === 'list'}
            <BioRecordingList onSelect={selectRecording} />

        {:else if navTab === 'bio' && navPage === 'detail' && currentRecording}
            <BioRecordingDetail
                recordingId={currentRecording}
                onBack={() => { navTab = 'bio'; navPage = 'list'; }}
            />

        {:else if navTab === 'build'}
            <!-- TODO: Build tab -->
            <div class="build-page">
                <div class="build-tabs">
                    <button
                        class="build-tab"
                        class:active={buildSubTab === 'system'}
                        onclick={() => (buildSubTab = 'system')}
                    >Build your own system</button>
                    <button
                        class="build-tab"
                        class:active={buildSubTab === 'stim'}
                        onclick={() => (buildSubTab = 'stim')}
                    >Build your own stimulation pipeline</button>
                </div>
                <div class="build-body">
                    {#if buildSubTab === 'system'}
                        <SystemGenerator />
                    {:else}
                        <!-- TODO: build your own stimulation pipeline -->
                        <div class="todo-placeholder">Coming soon</div>
                    {/if}
                </div>
            </div>
        {/if}
    </div>

    <div class="footer"><StatusBar /></div>
</div>

<style>
    .layout {
        width: 100vw;
        height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr auto;
        overflow: hidden;
    }

    /* Content area fills the middle row */
    .content {
        min-height: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }

    /* Footer spans full width in the last row */
    .footer {
        grid-column: 1 / -1;
    }

    /* ── Sim detail: 2-column split ── */
    .sim-detail {
        flex: 1;
        display: grid;
        grid-template-columns: 2fr 1fr;
        min-height: 0;
        overflow: hidden;
    }

    .scene-panel {
        position: relative;
        min-height: 0;
        background: #1a1a2e;
    }

    .console-panel {
        border-left: 1px solid #333;
        background: #1a1a2e;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .console-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 10px;
        background: #0d0d1a;
        border-bottom: 1px solid #333;
        font-size: 12px;
        flex-shrink: 0;
    }
    .console-system { font-weight: 700; color: #4fc3f7; }
    .list-btn {
        background: none;
        border: 1px solid #444;
        color: #888;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 3px;
        cursor: pointer;
    }
    .list-btn:hover { color: #ccc; border-color: #666; }

    /* ── Viz controls overlay (top-left of scene) ── */
    .controls {
        position: absolute;
        top: 8px;
        left: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        background: rgba(26, 26, 46, 0.92);
        border: 1px solid #333;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        z-index: 10;
    }
    .control-group { display: flex; align-items: center; gap: 10px; }
    .control-label { color: #888; font-weight: 600; }
    .toggle { display: flex; align-items: center; gap: 4px; cursor: pointer; color: #ccc; }
    .slider-label { display: flex; align-items: center; gap: 6px; color: #ccc; }
    .slider-label input[type="range"] { width: 80px; }

    /* ── Info panels overlay (top-right of scene) ── */
    .info-overlay {
        position: absolute;
        top: 8px;
        right: 8px;
        width: 230px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        z-index: 10;
    }
    .info-section {
        background: rgba(13, 13, 26, 0.93);
        border: 1px solid #333;
        border-radius: 6px;
        overflow: hidden;
    }
    .info-hdr {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 10px;
        background: none;
        border: none;
        color: #ccc;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        text-align: left;
    }
    .info-hdr:hover { background: rgba(255,255,255,0.04); }
    .chevron { font-size: 9px; color: #666; transition: transform 0.15s; }
    .chevron.open { transform: rotate(90deg); }

    .info-body {
        padding: 0 10px 8px;
        max-height: 200px;
        overflow-y: auto;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 11px;
        border-bottom: 1px solid #1a1a2e;
    }
    .info-row:last-child { border-bottom: none; }
    .info-row span:first-child { color: #888; flex-shrink: 0; margin-right: 6px; }
    .info-row span:last-child  { color: #e0e0e0; text-align: right; }
    .info-empty { font-size: 11px; color: #666; font-style: italic; padding: 4px 0; }
    .info-note  { font-size: 10px; color: #666; text-align: center; padding-top: 4px; }

    .neuron-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 10px;
    }
    .neuron-table th {
        color: #666;
        font-weight: 600;
        padding: 2px 4px;
        border-bottom: 1px solid #2a2a4a;
        text-align: left;
    }
    .neuron-table td {
        color: #ccc;
        padding: 2px 4px;
        border-bottom: 1px solid #1a1a2e;
    }
    .neuron-table tr:last-child td { border-bottom: none; }

    /* ── Build tab ── */
    .build-page {
        display: flex;
        flex-direction: column;
        height: 100%;
        background: #0d0d1a;
    }
    .build-tabs {
        display: flex;
        gap: 2px;
        padding: 12px 24px 0;
        border-bottom: 1px solid #2a2a4a;
        flex-shrink: 0;
    }
    .build-tab {
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        color: #666;
        font-size: 13px;
        font-weight: 600;
        padding: 6px 16px 10px;
        cursor: pointer;
        transition: color 0.15s, border-color 0.15s;
        font-family: inherit;
    }
    .build-tab:hover { color: #bbb; }
    .build-tab.active { color: #4fc3f7; border-bottom-color: #4fc3f7; }
    .build-body {
        flex: 1; 
        display: flex;
        min-height: 0;
        overflow: hidden;
    }
    .build-body > :global(*) {   
        flex: 1;
        min-width: 0;
        min-height: 0;
    }
    .todo-placeholder {
        color: #444;
        font-size: 14px;
        font-style: italic;
        margin: auto;
    }

    /* ── Responsive ── */
    @media (max-width: 900px) {
        .sim-detail {
            grid-template-columns: 1fr;
            grid-template-rows: 1fr 280px;
        }
        .console-panel {
            border-left: none;
            border-top: 1px solid #333;
        }
    }
</style>
