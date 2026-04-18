<script lang="ts">
    import { T, useThrelte } from "@threlte/core";
    import { OrbitControls } from "@threlte/extras";
    import { Color, Vector3 } from "three";
    import System from "./System.svelte";
    import IO from "./IO.svelte";
    import { envSystem, envIO, viewConfig } from "$lib/stores";

    const system = $derived($envSystem);
    const io = $derived($envIO);
    const config = $derived($viewConfig);

    // Compute bounding box center for OrbitControls target
    const bboxCenter = $derived.by(() => {
        const bb = system?.bounding_box;
        if (!bb || bb.length < 6) return [0, 0, 0];
        const cx = (bb[0] + bb[3]) / 2;
        const cy = (bb[2] + bb[5]) / 2; // system z → Three.js y
        const cz = (bb[1] + bb[4]) / 2; // system y → Three.js z
        return [cx, cy, cz];
    });

    const { scene } = useThrelte();
    scene.background = new Color(0xffffff);
</script>

<T.PerspectiveCamera makeDefault position={[0, 800, 1200]} fov={50}>
    <OrbitControls enableDamping dampingFactor={0.1} target={bboxCenter} />
</T.PerspectiveCamera>

<T.AmbientLight intensity={0.6} />
<T.DirectionalLight position={[500, 1000, 500]} intensity={0.8} />

{#if system}
    <System data={system} />
{/if}

{#if io && config.showElectrodes}
    <IO data={io} />
{/if}
