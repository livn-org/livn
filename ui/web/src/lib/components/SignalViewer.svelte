<script lang="ts">
    import { onMount, onDestroy } from 'svelte';

    let { rec, totalDuration, totalChannels }: {
        rec: string;          // path relative to bio_data (e.g. 'kimia/recording1')
        totalDuration: number;
        totalChannels: number;
    } = $props();

    const VISIBLE_CH = 32;
    const WIN_DUR    = 5;    // seconds per time window
    const DOWNSAMPLE = 100;  // 30000 / 100 = 300 Hz effective

    let canvasEl: HTMLCanvasElement;
    let chOffset   = $state(0);
    let timeOffset = $state(0);
    let scaleUV    = $state(250);   // µV half-height per channel slot

    let loading = $state(true);
    let error   = $state<string | null>(null);

    let data:     Float32Array | null = null;
    let nSamples  = 0;
    let effHz     = 0;

    let ro: ResizeObserver | undefined;

    onMount(async () => {
        await load();
        ro = new ResizeObserver(() => { if (data) draw(); });
        ro.observe(canvasEl);
    });

    onDestroy(() => ro?.disconnect());

    // ── Data fetching ─────────────────────────────────────────────────────
    async function load() {
        loading = true;
        error   = null;

        const qs = new URLSearchParams({
            rec,
            offset_s:   timeOffset.toString(),
            dur_s:      WIN_DUR.toString(),
            downsample: DOWNSAMPLE.toString(),
            ch_start:   chOffset.toString(),
            ch_end:     (chOffset + VISIBLE_CH).toString(),
        });

        try {
            const resp = await fetch(`/bio-api/chunk?${qs}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);

            const buf = await resp.arrayBuffer();
            data     = new Float32Array(buf);
            nSamples = parseInt(resp.headers.get('X-N-Samples') ?? '0');
            effHz    = parseInt(resp.headers.get('X-Sample-Rate') ?? '300');
            draw();
        } catch (e) {
            error = String(e);
        } finally {
            loading = false;
        }
    }

    // ── Canvas rendering ──────────────────────────────────────────────────
    function draw() {
        if (!canvasEl || !data || nSamples === 0) return;

        const dpr  = window.devicePixelRatio ?? 1;
        const rect = canvasEl.getBoundingClientRect();
        const W    = rect.width;
        const H    = rect.height;

        canvasEl.width  = Math.round(W * dpr);
        canvasEl.height = Math.round(H * dpr);

        const ctx = canvasEl.getContext('2d')!;
        ctx.scale(dpr, dpr);

        const LABEL_W = 50;
        const sigW    = W - LABEL_W;
        const visibleCh = Math.min(VISIBLE_CH, totalChannels - chOffset);
        const chH     = H / visibleCh;

        // Background
        ctx.fillStyle = '#0d0d1a';
        ctx.fillRect(0, 0, W, H);

        // Separator + zero lines
        for (let ci = 0; ci < visibleCh; ci++) {
            const cy = (ci + 0.5) * chH;

            if (ci > 0) {
                ctx.strokeStyle = '#1a1a2e';
                ctx.lineWidth   = 1;
                ctx.beginPath();
                ctx.moveTo(LABEL_W, ci * chH);
                ctx.lineTo(W, ci * chH);
                ctx.stroke();
            }

            ctx.strokeStyle = '#222240';
            ctx.lineWidth   = 0.5;
            ctx.beginPath();
            ctx.moveTo(LABEL_W, cy);
            ctx.lineTo(W, cy);
            ctx.stroke();
        }

        // Waveforms + labels
        const fontSize = Math.max(8, Math.min(11, chH * 0.55));
        ctx.font          = `${fontSize}px monospace`;
        ctx.textAlign     = 'right';
        ctx.textBaseline  = 'middle';

        for (let ci = 0; ci < visibleCh; ci++) {
            const ch = chOffset + ci;
            const cy = (ci + 0.5) * chH;
            const scale = (chH * 0.45) / scaleUV;

            // Channel label
            ctx.fillStyle = '#505070';
            ctx.fillText(`CH${ch + 1}`, LABEL_W - 3, cy);

            // Signal trace
            ctx.strokeStyle = '#4fc3f7';
            ctx.lineWidth   = 0.7;
            ctx.beginPath();

            const base = ci * nSamples;
            for (let s = 0; s < nSamples; s++) {
                const x = LABEL_W + (s / (nSamples - 1)) * sigW;
                const y = cy - data[base + s] * scale;
                s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        // Scale bar — top-right
        const barH  = chH * 0.45;
        const bx    = W - 8;
        const by    = 6;

        ctx.strokeStyle = '#888';
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx, by + barH);
        ctx.stroke();
        // tick marks
        ctx.beginPath();
        ctx.moveTo(bx - 3, by);       ctx.lineTo(bx + 3, by);
        ctx.moveTo(bx - 3, by + barH); ctx.lineTo(bx + 3, by + barH);
        ctx.stroke();

        ctx.textAlign     = 'right';
        ctx.textBaseline  = 'top';
        ctx.fillStyle     = '#777';
        ctx.font          = '9px monospace';
        ctx.fillText(`${scaleUV} µV`, bx - 5, by);

        // Time ticks — bottom of canvas
        const nTicks = 6;
        ctx.textBaseline = 'bottom';
        ctx.fillStyle    = '#555';
        ctx.font         = '9px monospace';
        for (let t = 0; t <= nTicks; t++) {
            const x    = LABEL_W + (t / nTicks) * sigW;
            const tSec = (timeOffset + (t / nTicks) * WIN_DUR).toFixed(1);
            ctx.textAlign = t === 0 ? 'left' : t === nTicks ? 'right' : 'center';
            ctx.fillText(`${tSec}s`, x, H - 2);
        }
    }

    // ── Navigation ────────────────────────────────────────────────────────
    async function prevCh() {
        if (chOffset === 0) return;
        chOffset = Math.max(0, chOffset - VISIBLE_CH);
        await load();
    }
    async function nextCh() {
        if (chOffset + VISIBLE_CH >= totalChannels) return;
        chOffset = Math.min(totalChannels - VISIBLE_CH, chOffset + VISIBLE_CH);
        await load();
    }
    async function prevTime() {
        if (timeOffset === 0) return;
        timeOffset = Math.max(0, timeOffset - WIN_DUR);
        await load();
    }
    async function nextTime() {
        if (timeOffset + WIN_DUR >= totalDuration) return;
        timeOffset = Math.min(totalDuration - WIN_DUR, timeOffset + WIN_DUR);
        await load();
    }
    function onScale(e: Event) {
        scaleUV = parseInt((e.target as HTMLInputElement).value);
        if (data) draw();
    }

    const chLabel   = $derived(`CH${chOffset + 1}–${Math.min(chOffset + VISIBLE_CH, totalChannels)}`);
    const timeLabel = $derived(`${timeOffset.toFixed(0)}s – ${Math.min(timeOffset + WIN_DUR, totalDuration).toFixed(0)}s`);
</script>

<div class="viewer">
    <canvas bind:this={canvasEl} class="canvas"></canvas>

    {#if loading}
        <div class="overlay">Loading…</div>
    {:else if error}
        <div class="overlay err">{error}</div>
    {/if}

    <div class="bar">
        <!-- Channel nav -->
        <div class="group">
            <button onclick={prevCh} disabled={chOffset === 0}>◀</button>
            <span class="lbl">{chLabel}</span>
            <button onclick={nextCh} disabled={chOffset + VISIBLE_CH >= totalChannels}>▶</button>
        </div>

        <!-- Time nav -->
        <div class="group">
            <button onclick={prevTime} disabled={timeOffset === 0}>◀</button>
            <span class="lbl time-lbl">{timeLabel}</span>
            <button onclick={nextTime} disabled={timeOffset + WIN_DUR >= totalDuration}>▶</button>
        </div>

        <!-- Amplitude scale -->
        <div class="group">
            <span class="dim">±</span>
            <input type="range" min="50" max="1000" step="50" value={scaleUV} oninput={onScale} />
            <span class="dim">{scaleUV} µV</span>
        </div>

        <span class="dim right">{effHz} Hz · {totalChannels} ch · {totalDuration.toFixed(0)} s</span>
    </div>
</div>

<style>
    .viewer {
        position: relative;
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 0;
        background: #0d0d1a;
    }

    .canvas {
        flex: 1;
        min-height: 0;
        width: 100%;
        display: block;
    }

    .overlay {
        position: absolute;
        inset: 0 0 36px 0;   /* leave bar at bottom */
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(13, 13, 26, 0.85);
        color: #888;
        font-size: 13px;
        pointer-events: none;
        z-index: 5;
    }
    .overlay.err { color: #ef5350; }

    .bar {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 10px;
        background: #0d0d1a;
        border-top: 1px solid #2a2a4a;
        font-size: 11px;
        flex-shrink: 0;
        flex-wrap: wrap;
        height: 36px;
        box-sizing: border-box;
    }

    .group { display: flex; align-items: center; gap: 5px; }
    .lbl   { color: #bbb; min-width: 90px; text-align: center; }
    .time-lbl { min-width: 70px; }
    .dim   { color: #666; white-space: nowrap; }
    .right { margin-left: auto; }

    button {
        background: #16162a;
        border: 1px solid #333;
        color: #aaa;
        padding: 1px 7px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 10px;
        line-height: 16px;
    }
    button:hover:not(:disabled) { background: #2a2a4a; color: #e0e0e0; }
    button:disabled { opacity: 0.3; cursor: not-allowed; }

    input[type="range"] { width: 70px; }
</style>
