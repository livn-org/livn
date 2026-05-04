<script lang="ts">
    import { useThrelte } from "@threlte/core";
    import {
        InstancedMesh,
        BoxGeometry,
        MeshBasicMaterial,
        EdgesGeometry,
        LineSegments,
        LineBasicMaterial,
        Mesh,
        SphereGeometry,
        Matrix4,
        Color,
        Raycaster,
        type Intersection,
    } from "three";
    import type { IOData } from "$lib/types";
    import { viewConfig, envSystem, activeExperiment, selectedElectrode } from "$lib/stores";
    import { onMount, onDestroy } from "svelte";

    interface Props {
        data: IOData;
    }

    let { data }: Props = $props();
    const config = $derived($viewConfig);
    const systemData = $derived($envSystem);

    const { scene, renderer, camera, invalidate } = useThrelte();

    let electrodeLines: LineSegments[] = [];
    let electrodeMeshes: Mesh[] = [];

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
        for (const ls of electrodeLines) {
            scene.remove(ls);
            ls.geometry.dispose();
            (ls.material as LineBasicMaterial).dispose();
        }
        for (const m of electrodeMeshes) {
            scene.remove(m);
            m.geometry.dispose();
            (m.material as MeshBasicMaterial).dispose();
        }
        electrodeLines = [];
        electrodeMeshes = [];

        const coords = data.electrode_coordinates;
        if (!coords || coords.length < 4) return;

        const count = Math.floor(coords.length / 4);
        const span = getSceneSpan();
        const size = span * 0.015;
        const boxGeo = new BoxGeometry(size, size, size);
        const edgeGeo = new EdgesGeometry(boxGeo);

        const hitGeo = new SphereGeometry(size * 1.2, 8, 8);

        for (let i = 0; i < count; i++) {
            const idx = i * 4;
            const eid = coords[idx];
            const x = coords[idx + 1];
            const y = coords[idx + 3];
            const z = coords[idx + 2];

            const lineMat = new LineBasicMaterial({
                color: new Color("#b8860b"),
                linewidth: 2,
            });
            const lines = new LineSegments(edgeGeo.clone(), lineMat);
            lines.position.set(x, y, z);
            lines.userData = { electrodeId: eid };
            scene.add(lines);
            electrodeLines.push(lines);

            const hitMat = new MeshBasicMaterial({ transparent: true, opacity: 0 });
            const hitMesh = new Mesh(hitGeo.clone(), hitMat);
            hitMesh.position.set(x, y, z);
            hitMesh.userData = { electrodeId: eid };
            scene.add(hitMesh);
            electrodeMeshes.push(hitMesh);
        }

        boxGeo.dispose();
        edgeGeo.dispose();
        hitGeo.dispose();
    }

    function updateElectrodeHighlight(selId: number | null) {
        for (let i = 0; i < electrodeLines.length; i++) {
            const ls = electrodeLines[i];
            const eid = ls.userData.electrodeId;
            const mat = ls.material as LineBasicMaterial;
            mat.color.set(eid === selId ? "#ffd54f" : "#b8860b");
            mat.needsUpdate = true;
        }
        invalidate();
    }

    $effect(() => {
        data;
        config;
        systemData;
        buildElectrodes();
        invalidate();
    });

    $effect(() => {
        updateElectrodeHighlight($selectedElectrode);
    });

    // ── Raycasting for electrode clicks ──────────────────────────────────
    const raycaster = new Raycaster();
    let pointerDownPos = { x: 0, y: 0 };

    function onPointerDown(e: PointerEvent) {
        pointerDownPos.x = e.clientX;
        pointerDownPos.y = e.clientY;
    }

    function onPointerUp(e: PointerEvent) {
        if ($activeExperiment === null) return;

        const canvas = renderer.domElement;
        const rect = canvas.getBoundingClientRect();
        if (
            e.clientX < rect.left || e.clientX > rect.right ||
            e.clientY < rect.top  || e.clientY > rect.bottom
        ) return;

        const dx = e.clientX - pointerDownPos.x;
        const dy = e.clientY - pointerDownPos.y;
        if (dx * dx + dy * dy > 25) return;

        const mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const my = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        const cam = camera.current;
        if (!cam) return;
        raycaster.setFromCamera({ x: mx, y: my } as any, cam);

        const hits: Intersection[] = raycaster.intersectObjects(electrodeMeshes);
        if (hits.length > 0) {
            const eid = hits[0].object.userData.electrodeId as number;
            selectedElectrode.update(cur => cur === eid ? null : eid);
        }
    }

    onMount(() => {
        window.addEventListener("pointerdown", onPointerDown);
        window.addEventListener("pointerup", onPointerUp);
    });

    onDestroy(() => {
        window.removeEventListener("pointerdown", onPointerDown);
        window.removeEventListener("pointerup", onPointerUp);
        for (const ls of electrodeLines) {
            scene.remove(ls);
            ls.geometry.dispose();
            (ls.material as LineBasicMaterial).dispose();
        }
        for (const m of electrodeMeshes) {
            scene.remove(m);
            m.geometry.dispose();
            (m.material as MeshBasicMaterial).dispose();
        }
    });
</script>
