<script lang="ts">
    import "../app.css";
    import { Canvas } from "@threlte/core";
    import EnvScene from "$lib/components/EnvScene.svelte";
    import Console from "$lib/components/Console.svelte";
    import SystemLoader from "$lib/components/SystemLoader.svelte";
    import StatusBar from "$lib/components/StatusBar.svelte";
    import Tooltip from "$lib/components/Tooltip.svelte";
    import { viewConfig, envSystem } from "$lib/stores";
    import type { ViewConfig } from "$lib/types";

    const config = $derived($viewConfig);
    const system = $derived($envSystem);

    function togglePop(pop: string) {
        viewConfig.update((vc) => ({
            ...vc,
            popVisibility: {
                ...vc.popVisibility,
                [pop]: !vc.popVisibility[pop],
            },
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
        viewConfig.update((vc) => ({
            ...vc,
            showBoundingBox: !vc.showBoundingBox,
        }));
    }

    function toggleElectrodes() {
        viewConfig.update((vc) => ({
            ...vc,
            showElectrodes: !vc.showElectrodes,
        }));
    }
</script>

<div class="layout">
    <div class="scene-panel">
        <Canvas>
            <EnvScene />
        </Canvas>

        <Tooltip />

        {#if system}
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
                        <input
                            type="range"
                            min="0.2"
                            max="3"
                            step="0.1"
                            value={config.pointSize}
                            oninput={setPointSize}
                        />
                    </label>
                    <label class="slider-label">
                        Opacity
                        <input
                            type="range"
                            min="0.1"
                            max="1"
                            step="0.05"
                            value={config.opacity}
                            oninput={setOpacity}
                        />
                    </label>
                </div>
                <div class="control-group">
                    <label class="toggle">
                        <input
                            type="checkbox"
                            checked={config.showBoundingBox}
                            onchange={toggleBBox}
                        />
                        Bounding box
                    </label>
                    <label class="toggle">
                        <input
                            type="checkbox"
                            checked={config.showElectrodes}
                            onchange={toggleElectrodes}
                        />
                        Electrodes
                    </label>
                </div>
            </div>
        {/if}
    </div>

    <div class="console-panel">
        <SystemLoader />
        <Console />
    </div>

    <StatusBar />
</div>

<style>
    .layout {
        width: 100vw;
        height: 100vh;
        display: grid;
        grid-template-rows: 1fr auto auto;
        grid-template-columns: 2fr 1fr;
    }
    .scene-panel {
        grid-column: 1;
        grid-row: 1;
        position: relative;
        min-height: 0;
        background: #1a1a2e;
    }
    .console-panel {
        grid-column: 2;
        grid-row: 1;
        min-height: 0;
        border-left: 1px solid #333;
        background: #1a1a2e;
        display: flex;
        flex-direction: column;
    }
    :global(.layout > footer),
    :global(StatusBar) {
        grid-column: 1 / -1;
    }
    .controls {
        position: absolute;
        top: 8px;
        left: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        background: rgba(26, 26, 46, 0.9);
        border: 1px solid #333;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        z-index: 10;
    }
    .control-group {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .control-label {
        color: #888;
        font-weight: 600;
    }
    .toggle {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
        color: #ccc;
    }
    .slider-label {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #ccc;
    }
    .slider-label input[type="range"] {
        width: 80px;
    }

    /* StatusBar spans full width */
    .layout > :global(:last-child) {
        grid-column: 1 / -1;
        grid-row: 3;
    }

    /* Responsive: stack on narrow screens */
    @media (max-width: 900px) {
        .layout {
            grid-template-columns: 1fr;
            grid-template-rows: 1fr 300px auto;
        }
        .scene-panel {
            grid-column: 1;
            grid-row: 1;
        }
        .console-panel {
            grid-column: 1;
            grid-row: 2;
            border-left: none;
            border-top: 1px solid #333;
        }
    }
</style>
