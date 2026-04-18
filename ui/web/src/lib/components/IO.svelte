<script lang="ts">
    import { useThrelte } from "@threlte/core";
    import {
        InstancedMesh,
        BoxGeometry,
        MeshBasicMaterial,
        EdgesGeometry,
        LineSegments,
        LineBasicMaterial,
        Matrix4,
        Color,
    } from "three";
    import type { IOData, ViewConfig } from "$lib/types";
    import { viewConfig, envSystem } from "$lib/stores";
    import { onMount, onDestroy } from "svelte";

    interface Props {
        data: IOData;
    }

    let { data }: Props = $props();
    const config = $derived($viewConfig);
    const systemData = $derived($envSystem);

    const { scene, invalidate } = useThrelte();

    let electrodeLines: LineSegments[] = [];

    function getSceneSpan(): number {
        const bb = systemData?.bounding_box;
        if (bb && bb.length >= 6) {
            const dx = bb[3] - bb[0];
            const dy = bb[4] - bb[1];
            const dz = bb[5] - bb[2];
            return Math.max(dx, dy, dz, 1);
        }
        return 100;
    }

    function buildElectrodes() {
        // Clean up old
        for (const ls of electrodeLines) {
            scene.remove(ls);
            ls.geometry.dispose();
            (ls.material as LineBasicMaterial).dispose();
        }
        electrodeLines = [];

        const coords = data.electrode_coordinates;
        if (!coords || coords.length < 4) return;

        const count = Math.floor(coords.length / 4);
        const span = getSceneSpan();
        const size = span * 0.015;
        const boxGeo = new BoxGeometry(size, size, size);
        const edgeGeo = new EdgesGeometry(boxGeo);
        const lineMat = new LineBasicMaterial({
            color: new Color("#b8860b"),
            linewidth: 2,
        });

        for (let i = 0; i < count; i++) {
            const idx = i * 4;
            const x = coords[idx + 1];
            const y = coords[idx + 3]; // system z → Three.js y (up)
            const z = coords[idx + 2]; // system y → Three.js z

            const lines = new LineSegments(edgeGeo.clone(), lineMat.clone());
            lines.position.set(x, y, z);
            lines.userData = { electrodeId: coords[idx] };
            scene.add(lines);
            electrodeLines.push(lines);
        }

        boxGeo.dispose();
        edgeGeo.dispose();
        lineMat.dispose();
    }

    $effect(() => {
        data;
        config;
        systemData;
        buildElectrodes();
        invalidate();
    });

    onDestroy(() => {
        for (const ls of electrodeLines) {
            scene.remove(ls);
            ls.geometry.dispose();
            (ls.material as LineBasicMaterial).dispose();
        }
    });
</script>
