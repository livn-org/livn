<script setup>
import { ref, nextTick, onUnmounted } from "vue";

const state = ref("idle"); // idle | loading | ready | error
const logs = ref([]);
const errorMsg = ref("");
const containerRef = ref(null);
const systemData = ref(null);
const tooltipData = ref(null);

let pyodide = null;
let animationId = null;

function log(msg) {
    logs.value.push(msg);
}

async function run() {
    state.value = "loading";
    logs.value = [];
    errorMsg.value = "";

    try {
        log("Loading Pyodide runtime…");
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js";
        await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = () => reject(new Error("Failed to load Pyodide CDN"));
            document.head.appendChild(script);
        });

        log("Initializing Pyodide…");
        pyodide = await globalThis.loadPyodide();

        log("Installing packages (this may take a moment)…");
        await pyodide.loadPackage("micropip");
        const micropip = pyodide.pyimport("micropip");
        await micropip.install(["livn", "fsspec", "huggingface_hub", "httpcore"]);

        log("Downloading EI1 system…");
        const result = await pyodide.runPythonAsync(`
import json
from livn.env import Env
from livn.system import predefined

env = Env(predefined('EI1'))

coords = env.io.electrode_coordinates.tolist()
neuron_coords = env.system.neuron_coordinates.tolist()
populations = list(env.system.populations)

# Gather per-population coordinates
pop_coords = {}
for pop in populations:
    pop_coords[pop] = env.system.coordinate_array(pop).tolist()

bbox = env.system.bounding_box.tolist()

json.dumps({
    "electrodes": coords,
    "populations": populations,
    "pop_coords": pop_coords,
    "num_neurons": int(env.system.num_neurons),
    "num_electrodes": int(env.io.num_channels),
    "bounding_box": bbox,
})
        `);

        const data = JSON.parse(result);
        systemData.value = data;
        log(
            `Loaded: ${data.num_neurons} neurons, ${data.num_electrodes} electrodes`,
        );
        state.value = "ready";
        await nextTick();

        log("Loading Three.js…");
        const threeUrl = "https://esm.sh/three@0.170.0";
        const controlsUrl = "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";
        const THREE = await import(/* @vite-ignore */ threeUrl);
        const { OrbitControls } = await import(/* @vite-ignore */ controlsUrl);
        drawVisualization(data, THREE, OrbitControls);
    } catch (e) {
        state.value = "error";
        errorMsg.value = e.message || String(e);
    }
}

function drawVisualization(data, THREE, OrbitControls) {
    const container = containerRef.value;
    if (!container) return;

    const width = container.clientWidth;
    const height = 500;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 10000);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Map system coords [x, y, z] -> Three.js [x, z, y] so the thin z-axis becomes Y (up)
    function toScene(x, y, z) {
        return [x, z, y];
    }

    // Bounding box in scene coords
    const bbMin = toScene(...data.bounding_box[0]);
    const bbMax = toScene(...data.bounding_box[1]);
    const cx = (bbMin[0] + bbMax[0]) / 2;
    const cy = (bbMin[1] + bbMax[1]) / 2;
    const cz = (bbMin[2] + bbMax[2]) / 2;
    const span = Math.max(
        bbMax[0] - bbMin[0],
        bbMax[1] - bbMin[1],
        bbMax[2] - bbMin[2],
    ) || 1;

    controls.target.set(cx, cy, cz);
    camera.position.set(cx + span * 0.8, cy + span * 0.6, cz + span * 1.2);

    const popColors = { EXC: 0x4fc3f7, INH: 0xef5350 };
    const defaultColor = 0xaaaaaa;


    const neuronMeshes = [];
    const neuronInfoMap = new Map(); // mesh -> array of neuron info

    const neuronGeo = new THREE.SphereGeometry(span * 0.02, 12, 12);
    for (const pop of data.populations) {
        const color = popColors[pop] ?? defaultColor;
        const mat = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.85,
        });
        const coords = data.pop_coords[pop];
        const mesh = new THREE.InstancedMesh(neuronGeo, mat, coords.length);
        const dummy = new THREE.Object3D();
        const info = coords.map((c, i) => {
            const [sx, sy, sz] = toScene(c[1], c[2], c[3]);
            dummy.position.set(sx, sy, sz);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
            return { id: Math.round(c[0]), x: c[1], y: c[2], z: c[3], pop };
        });
        neuronMeshes.push(mesh);
        neuronInfoMap.set(mesh, info);
        scene.add(mesh);
    }

    // Electrode positions in original coords for distance calculation
    const electrodePositions = data.electrodes.map((e) => [e[1], e[2], e[3]]);

    // Electrodes as cubes
    const elSize = span * 0.03;
    const elGeo = new THREE.BoxGeometry(elSize, elSize, elSize);
    const elMat = new THREE.MeshBasicMaterial({
        color: 0xfdd835,
        wireframe: true,
    });
    const elMesh = new THREE.InstancedMesh(
        elGeo,
        elMat,
        data.electrodes.length,
    );
    const dummy = new THREE.Object3D();
    data.electrodes.forEach((e, i) => {
        const [sx, sy, sz] = toScene(e[1], e[2], e[3]);
        dummy.position.set(sx, sy, sz);
        dummy.updateMatrix();
        elMesh.setMatrixAt(i, dummy.matrix);
    });
    scene.add(elMesh);

    // Bounding box wireframe
    const bbSizeX = bbMax[0] - bbMin[0];
    const bbSizeY = bbMax[1] - bbMin[1];
    const bbSizeZ = bbMax[2] - bbMin[2];
    const bbGeo = new THREE.BoxGeometry(bbSizeX, bbSizeY, bbSizeZ);
    const bbEdges = new THREE.EdgesGeometry(bbGeo);
    const bbLine = new THREE.LineSegments(
        bbEdges,
        new THREE.LineBasicMaterial({ color: 0xbbbbcc, linewidth: 2, transparent: true, opacity: 0.8 }),
    );
    bbLine.position.set(cx, cy, cz);
    scene.add(bbLine);

    // Ambient light
    scene.add(new THREE.AmbientLight(0xffffff, 1));

    // Raycaster for neuron click
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const onClick = (event) => {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);

        const intersects = raycaster.intersectObjects(neuronMeshes);
        if (intersects.length > 0) {
            const hit = intersects[0];
            const info = neuronInfoMap.get(hit.object);
            if (info && hit.instanceId != null) {
                const n = info[hit.instanceId];
                // Find closest electrode
                let minDist = Infinity;
                let closestEl = 0;
                electrodePositions.forEach((ep, idx) => {
                    const d = Math.sqrt(
                        (n.x - ep[0]) ** 2 + (n.y - ep[1]) ** 2 + (n.z - ep[2]) ** 2,
                    );
                    if (d < minDist) { minDist = d; closestEl = idx; }
                });
                tooltipData.value = {
                    id: n.id,
                    pop: n.pop,
                    x: n.x.toFixed(1),
                    y: n.y.toFixed(1),
                    z: n.z.toFixed(1),
                    electrode: Math.round(data.electrodes[closestEl][0]),
                    distance: minDist.toFixed(1),
                };
            }
        } else {
            tooltipData.value = null;
        }
    };
    renderer.domElement.addEventListener("click", onClick);

    // Animate
    function animate() {
        animationId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Resize
    const onResize = () => {
        const w = container.clientWidth;
        camera.aspect = w / height;
        camera.updateProjectionMatrix();
        renderer.setSize(w, height);
    };
    window.addEventListener("resize", onResize);

    // Stash cleanup
    container._cleanup = () => {
        window.removeEventListener("resize", onResize);
        renderer.domElement.removeEventListener("click", onClick);
        if (animationId) cancelAnimationFrame(animationId);
        renderer.dispose();
    };
}

onUnmounted(() => {
    pyodide = null;
    if (animationId) cancelAnimationFrame(animationId);
    const c = containerRef.value;
    if (c && c._cleanup) c._cleanup();
});
</script>

<template>
    <div class="pyodide-widget">
        <div v-if="state === 'idle'" class="widget-start">
            <p class="widget-description">
                Load Pyodide in your browser, download the EI1 system, and
                visualize neuron and electrode positions.
            </p>
            <button class="widget-btn" @click="run">
                Launch Interactive Demo
            </button>
        </div>

        <div v-if="state === 'loading'" class="widget-loading">
            <div class="spinner" />
            <div class="log-output">
                <p v-for="(msg, i) in logs" :key="i">{{ msg }}</p>
            </div>
        </div>

        <div v-if="state === 'error'" class="widget-error">
            <p>Something went wrong:</p>
            <pre>{{ errorMsg }}</pre>
            <button class="widget-btn" @click="run">Retry</button>
        </div>

        <div v-if="state === 'ready'" class="widget-result">
            <div class="viewer-wrapper">
                <div class="system-name">EI1</div>
                <div ref="containerRef" class="three-container" />
                <div class="legend">
                    <span class="legend-item"><span class="dot dot-e" /> Excitatory ({{ systemData?.pop_coords?.EXC?.length ?? systemData?.pop_coords?.E?.length ?? '?' }})</span>
                    <span class="legend-item"><span class="dot dot-i" /> Inhibitory ({{ systemData?.pop_coords?.INH?.length ?? systemData?.pop_coords?.I?.length ?? '?' }})</span>
                    <span class="legend-item"><span class="cube" /> Electrodes ({{ systemData?.num_electrodes ?? '?' }})</span>
                </div>
                <div class="controls-hint">Drag to rotate · Scroll to zoom · Click neuron for details</div>
                <div v-if="tooltipData" class="neuron-tooltip">
                    <button class="tooltip-close" @click="tooltipData = null">&times;</button>
                    <div><strong>{{ tooltipData.pop }}</strong> neuron #{{ tooltipData.id }}</div>
                    <div>Position: ({{ tooltipData.x }}, {{ tooltipData.y }}, {{ tooltipData.z }})</div>
                    <div>Nearest electrode: #{{ tooltipData.electrode }} ({{ tooltipData.distance }} &micro;m)</div>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.pyodide-widget {
    margin: 1.5rem 0;
    padding: 1.25rem;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    background: var(--vp-c-bg-soft);
}

.widget-description {
    margin: 0 0 1rem;
    color: var(--vp-c-text-2);
}

.widget-btn {
    padding: 0.5rem 1.25rem;
    border: none;
    border-radius: 6px;
    background: var(--vp-c-brand-1);
    color: var(--vp-c-white);
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}
.widget-btn:hover {
    background: var(--vp-c-brand-2);
}

.widget-loading {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
}

.spinner {
    width: 24px;
    height: 24px;
    border: 3px solid var(--vp-c-divider);
    border-top-color: var(--vp-c-brand-1);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.log-output {
    font-size: 0.85rem;
    color: var(--vp-c-text-2);
    line-height: 1.6;
}
.log-output p {
    margin: 0;
}
.log-done {
    margin-bottom: 1rem;
}

.widget-error pre {
    color: var(--vp-c-danger-1);
    white-space: pre-wrap;
    font-size: 0.85rem;
}

.widget-result .viewer-wrapper {
    position: relative;
}

.system-name {
    position: absolute;
    top: 12px;
    left: 12px;
    font-size: 1rem;
    font-weight: 700;
    color: #333;
    background: rgba(255, 255, 255, 0.9);
    padding: 4px 10px;
    border-radius: 6px;
    pointer-events: none;
    z-index: 1;
}

.three-container {
    width: 100%;
    height: 500px;
    border-radius: 6px;
    overflow: hidden;
}

.three-container canvas {
    display: block;
}

.legend {
    position: absolute;
    top: 12px;
    right: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(255, 255, 255, 0.9);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #333;
    pointer-events: none;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.dot-e { background: #4fc3f7; }
.dot-i { background: #ef5350; }

.cube {
    width: 10px;
    height: 10px;
    border: 2px solid #fdd835;
    display: inline-block;
}

.controls-hint {
    position: absolute;
    bottom: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.75rem;
    color: rgba(0, 0, 0, 0.4);
    pointer-events: none;
}

.neuron-tooltip {
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #ddd;
    padding: 8px 28px 8px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #333;
    line-height: 1.5;
    z-index: 2;
}

.tooltip-close {
    position: absolute;
    top: 4px;
    right: 6px;
    background: none;
    border: none;
    font-size: 1rem;
    cursor: pointer;
    color: #999;
    line-height: 1;
}
</style>
