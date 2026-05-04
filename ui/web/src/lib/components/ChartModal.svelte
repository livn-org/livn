<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import type { RowData, ElectrodeData } from '$lib/pyodide';

    interface Props {
        // Neuron chart mode (required when chartType is 'spikes' | 'voltage')
        gid?: number;
        pop?: string;
        rowData?: RowData;
        // Electrode chart mode (required when chartType is 'electrode-spikes' | 'electrode-lfp')
        electrodeData?: ElectrodeData;
        channelId?: number;
        // Shared
        chartType: 'spikes' | 'voltage' | 'electrode-spikes' | 'electrode-lfp';
        initialCursor: number;
        onClose: () => void;
    }

    let { gid, pop, rowData, electrodeData, channelId, chartType, initialCursor, onClose }: Props = $props();

    const totalDuration = $derived(electrodeData?.duration ?? rowData?.duration ?? 1000);

    let canvas: HTMLCanvasElement | undefined = $state();
    let viewStart = $state(0);
    let viewEnd   = $state(totalDuration);

    // Own play state — starts from wherever parent cursor was
    let cursor  = $state(initialCursor);
    let playing = $state(false);
    let animId: number | null = null;
    let lastTs = 0;
    let playbackRate = $state(3);
    const SPEED_OPTIONS = [1, 3, 5, 10];

    let isDragging = false;
    let dragStartX = 0;
    let dragVS = 0, dragVE = 0;

    const spikeTimes = $derived(
        rowData && gid !== undefined
            ? ((rowData.spikes[String(gid) as unknown as number] ?? []) as number[])
            : []
    );
    const voltageData = $derived(
        rowData && gid !== undefined
            ? ((rowData.voltages[String(gid) as unknown as number] ?? []) as number[])
            : []
    );

    let vMin = 0, vMax = 1;
    $effect(() => {
        if (voltageData.length > 0) {
            vMin = voltageData[0]; vMax = voltageData[0];
            for (const v of voltageData) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
        }
    });

    $effect(() => { void viewStart; void viewEnd; void cursor; draw(); });

    // ── Playback ──────────────────────────────────────────────────────────
    function playFrame(ts: number) {
        if (!playing) return;
        if (lastTs > 0) {
            const next = cursor + (ts - lastTs) * playbackRate;
            if (next >= totalDuration) {
                cursor = totalDuration;
                stopPlay();
                return;
            }
            cursor = next;
        }
        lastTs = ts;
        animId = requestAnimationFrame(playFrame);
    }

    function togglePlay() {
        if (playing) {
            stopPlay();
        } else {
            cursor = 0;
            playing = true;
            lastTs  = 0;
            animId  = requestAnimationFrame(playFrame);
        }
    }

    function stopPlay() {
        playing = false;
        if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
    }

    onDestroy(stopPlay);

    // ── Drawing ───────────────────────────────────────────────────────────
    function draw() {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr  = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        canvas.width  = rect.width  * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w    = rect.width;
        const h    = rect.height;
        const span = Math.max(1, viewEnd - viewStart);

        ctx.fillStyle = '#0a0a18';
        ctx.fillRect(0, 0, w, h);

        function tX(t: number) { return ((t - viewStart) / span) * w; }

        // Time grid
        const rawStep   = span / 8;
        const mag       = Math.pow(10, Math.floor(Math.log10(Math.max(rawStep, 0.001))));
        const norm      = rawStep / mag;
        const gridStep  = norm < 1.5 ? mag : norm < 3.5 ? 2 * mag : norm < 7.5 ? 5 * mag : 10 * mag;
        const gridStart = Math.ceil(viewStart / gridStep) * gridStep;

        const axisH = 18;
        for (let t = gridStart; t <= viewEnd + 0.001; t += gridStep) {
            const x = tX(t);
            ctx.strokeStyle = '#1a1a3a';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - axisH); ctx.stroke();
            ctx.fillStyle = '#999';
            ctx.font = '11px monospace';
            ctx.fillText(`${Math.round(t)}ms`, x + 3, h - 4);
        }

        const plotH = h - axisH;

        if (chartType === 'spikes') {
            const margin = plotH * 0.12;
            ctx.strokeStyle = '#4fc3f7';
            ctx.lineWidth = 2;
            for (const t of spikeTimes) {
                if (t < viewStart || t > viewEnd) continue;
                const x = tX(t);
                ctx.beginPath(); ctx.moveTo(x, margin); ctx.lineTo(x, plotH - margin); ctx.stroke();
            }
            const vis = spikeTimes.filter(t => t >= viewStart && t <= viewEnd).length;
            ctx.fillStyle = '#555';
            ctx.font = '10px monospace';
            ctx.fillText(`${vis} spike${vis !== 1 ? 's' : ''} in view`, w - 110, plotH - 6);

        } else if (chartType === 'voltage' && voltageData.length > 0 && rowData) {
            const dt     = rowData.duration / voltageData.length;
            const vRange = vMax - vMin || 1;
            const padT   = 8, padB = 8;
            const traceH = plotH - padT - padB;

            function vY(v: number) { return padT + traceH - ((v - vMin) / vRange) * traceH; }

            if (vMin < 0 && vMax > 0) {
                const y0 = vY(0);
                ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#555'; ctx.font = '10px monospace';
                ctx.fillText('0', 4, y0 - 2);
            }

            ctx.fillStyle = '#777'; ctx.font = '10px monospace';
            ctx.fillText(`${vMax.toFixed(1)} mV`, 4, padT + 10);
            ctx.fillText(`${vMin.toFixed(1)} mV`, 4, plotH - padB - 2);

            const iStart = Math.max(0, Math.floor(viewStart / dt));
            const iEnd   = Math.min(voltageData.length - 1, Math.ceil(viewEnd / dt));
            const step   = Math.max(1, Math.floor((iEnd - iStart) / (w * 2)));

            ctx.strokeStyle = '#ffb74d'; ctx.lineWidth = 1.5;
            ctx.beginPath();
            let first = true;
            for (let i = iStart; i <= iEnd; i += step) {
                const x = tX(i * dt);
                const y = vY(voltageData[i]);
                first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                first = false;
            }
            if (!first) ctx.stroke();

        } else if (chartType === 'electrode-spikes' && electrodeData) {
            const margin = plotH * 0.12;
            ctx.strokeStyle = '#80cbc4';
            ctx.lineWidth = 2;
            for (const t of electrodeData.spikeTimes) {
                if (t < viewStart || t > viewEnd) continue;
                const x = tX(t);
                ctx.beginPath(); ctx.moveTo(x, margin); ctx.lineTo(x, plotH - margin); ctx.stroke();
            }
            const vis = electrodeData.spikeTimes.filter(t => t >= viewStart && t <= viewEnd).length;
            ctx.fillStyle = '#555';
            ctx.font = '10px monospace';
            ctx.fillText(`${vis} spike${vis !== 1 ? 's' : ''} in view`, w - 110, plotH - 6);

        } else if (chartType === 'electrode-lfp' && electrodeData && electrodeData.lfp.length > 1) {
            const vs     = electrodeData.lfp;
            const dt     = electrodeData.duration / vs.length;
            let eMin = vs[0], eMax = vs[0];
            for (const v of vs) { if (v < eMin) eMin = v; if (v > eMax) eMax = v; }
            const eRange = eMax - eMin || 1;
            const padT   = 8, padB = 8;
            const traceH = plotH - padT - padB;

            function eY(v: number) { return padT + traceH - ((v - eMin) / eRange) * traceH; }

            ctx.fillStyle = '#777'; ctx.font = '10px monospace';
            ctx.fillText(`${eMax.toFixed(2)} µV`, 4, padT + 10);
            ctx.fillText(`${eMin.toFixed(2)} µV`, 4, plotH - padB - 2);

            const iStart = Math.max(0, Math.floor(viewStart / dt));
            const iEnd   = Math.min(vs.length - 1, Math.ceil(viewEnd / dt));
            const step   = Math.max(1, Math.floor((iEnd - iStart) / (w * 2)));

            ctx.strokeStyle = '#80cbc4'; ctx.lineWidth = 1.5;
            ctx.beginPath();
            let first = true;
            for (let i = iStart; i <= iEnd; i += step) {
                const x = tX(i * dt);
                const y = eY(vs[i]);
                first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                first = false;
            }
            if (!first) ctx.stroke();
        }

        // Axis separator
        ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, plotH); ctx.lineTo(w, plotH); ctx.stroke();

        // Cursor line (only through plot area, not over axis labels)
        const cx = tX(cursor);
        if (cx >= 0 && cx <= w) {
            ctx.strokeStyle = 'rgba(255, 213, 79, 0.9)'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, plotH); ctx.stroke();
        }

        // ── Fixed-position value box (top-right, always visible) ──────────
        const timeStr = `${cursor.toFixed(0)} ms`;
        let valStr = '';
        if (chartType === 'voltage' && voltageData.length > 0 && rowData) {
            const dt  = rowData.duration / voltageData.length;
            const idx = Math.min(voltageData.length - 1, Math.max(0, Math.round(cursor / dt)));
            valStr = `${voltageData[idx].toFixed(2)} mV`;
        } else if (chartType === 'spikes') {
            const n = spikeTimes.filter(t => t <= cursor).length;
            valStr = `${n} spike${n !== 1 ? 's' : ''}`;
        } else if (chartType === 'electrode-spikes' && electrodeData) {
            const n = electrodeData.spikeTimes.filter(t => t <= cursor).length;
            valStr = `${n} spike${n !== 1 ? 's' : ''}`;
        } else if (chartType === 'electrode-lfp' && electrodeData && electrodeData.lfp.length > 0) {
            const dt  = electrodeData.duration / electrodeData.lfp.length;
            const idx = Math.min(electrodeData.lfp.length - 1, Math.max(0, Math.round(cursor / dt)));
            valStr = `${electrodeData.lfp[idx].toFixed(3)} µV`;
        }

        const pad = 10, margin = 10;
        ctx.font = 'bold 13px monospace';
        const tw1 = ctx.measureText(timeStr).width;
        ctx.font = 'bold 24px monospace';
        const tw2 = valStr ? ctx.measureText(valStr).width : 0;
        const bw  = Math.max(tw1, tw2) + pad * 2;
        const bh  = valStr ? 62 : 30;
        const bx  = w - bw - margin;
        const by  = margin;

        ctx.fillStyle = 'rgba(8, 8, 20, 0.9)';
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeStyle = '#ffd54f';
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = '#999';
        ctx.font = 'bold 13px monospace';
        ctx.fillText(timeStr, bx + pad, by + 20);

        if (valStr) {
            ctx.fillStyle = '#ffd54f';
            ctx.font = 'bold 24px monospace';
            ctx.fillText(valStr, bx + pad, by + 52);
        }
    }

    // ── Zoom / pan ────────────────────────────────────────────────────────
    function onWheel(e: WheelEvent) {
        e.preventDefault();
        if (!canvas) return;
        const rect   = canvas.getBoundingClientRect();
        const mx     = e.clientX - rect.left;
        const mouseT = viewStart + (mx / rect.width) * (viewEnd - viewStart);
        const factor = e.deltaY > 0 ? 1.25 : 0.8;
        const raw    = (viewEnd - viewStart) * factor;
        const span   = Math.max(10, Math.min(totalDuration, raw));
        const frac   = mx / rect.width;
        let ns = mouseT - frac * span;
        let ne = ns + span;
        if (ns < 0) { ns = 0; ne = span; }
        if (ne > totalDuration) { ne = totalDuration; ns = ne - span; }
        viewStart = Math.max(0, ns);
        viewEnd   = Math.min(rowData.duration, ne);
    }

    function onMouseDown(e: MouseEvent) {
        if (e.button !== 0) return;
        isDragging = true;
        dragStartX = e.clientX;
        dragVS = viewStart; dragVE = viewEnd;
    }

    function onMouseMove(e: MouseEvent) {
        if (!isDragging || !canvas) return;
        const rect   = canvas.getBoundingClientRect();
        const dx     = e.clientX - dragStartX;
        const span   = dragVE - dragVS;
        const shift  = -(dx / rect.width) * span;
        let ns = dragVS + shift, ne = ns + span;
        if (ns < 0) { ns = 0; ne = span; }
        if (ne > totalDuration) { ne = totalDuration; ns = ne - span; }
        viewStart = ns; viewEnd = ne;
    }

    function onMouseUp() { isDragging = false; }

    function handleKey(e: KeyboardEvent) {
        if (e.key === 'Escape') onClose();
        if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    }

    onMount(() => {
        window.addEventListener('keydown', handleKey);
        requestAnimationFrame(draw);
    });
    onDestroy(() => window.removeEventListener('keydown', handleKey));
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div class="overlay" onclick={(e) => e.target === e.currentTarget && onClose()}>
    <div class="modal">
        <!-- Header -->
        <div class="modal-header">
            <span class="modal-title">
                {#if chartType === 'electrode-spikes' || chartType === 'electrode-lfp'}
                    <span class="electrode-label">CH {channelId}</span>
                    <span class="pop-badge" style="background:#1a2a1a;color:#80cbc4;">MEA</span>
                    <span class="chart-kind">{chartType === 'electrode-spikes' ? 'Channel Spikes' : 'LFP'}</span>
                {:else}
                    GID {gid}
                    {#if pop}
                        <span class="pop-badge" class:exc={pop === 'EXC'} class:inh={pop === 'INH'}>{pop}</span>
                    {/if}
                    <span class="chart-kind">{chartType === 'spikes' ? 'Spike Raster' : 'Voltage Trace'}</span>
                {/if}
            </span>
            <button class="close-btn" onclick={onClose} title="Close (Esc)">×</button>
        </div>

        <!-- Canvas -->
        <div class="canvas-wrap">
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <canvas
                bind:this={canvas}
                style:cursor={isDragging ? 'grabbing' : 'grab'}
                onwheel={onWheel}
                onmousedown={onMouseDown}
                onmousemove={onMouseMove}
                onmouseup={onMouseUp}
                onmouseleave={onMouseUp}
            ></canvas>
        </div>

        <!-- Play controls -->
        <div class="play-bar">
            <button class="play-btn" onclick={togglePlay} title="Play from start (Space)">
                {playing ? '⏸' : '▶'}
            </button>
            <div class="speed-group">
                {#each SPEED_OPTIONS as spd (spd)}
                    <button
                        class="speed-btn"
                        class:active={playbackRate === spd}
                        onclick={() => { playbackRate = spd; }}
                    >{spd}×</button>
                {/each}
            </div>
            <input
                class="time-slider"
                type="range"
                min="0"
                max={totalDuration}
                step="1"
                value={cursor}
                oninput={(e) => {
                    stopPlay();
                    cursor = parseFloat((e.target as HTMLInputElement).value);
                }}
            />
            <span class="time-label">{cursor.toFixed(0)} / {totalDuration} ms</span>
            <div class="spacer"></div>
            <button class="action-btn" onclick={() => { viewStart = 0; viewEnd = totalDuration; }}>
                Reset zoom
            </button>
            <span class="hint">scroll=zoom · drag=pan</span>
        </div>
    </div>
</div>

<style>
    .overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.55);
        z-index: 200;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .modal {
        background: #13131f;
        border: 1px solid #444;
        border-radius: 8px;
        width: 72vw;
        height: 52vh;
        min-width: 520px;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 6px 32px rgba(0, 0, 0, 0.6);
    }

    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 12px;
        background: #0d0d1a;
        border-bottom: 1px solid #333;
        flex-shrink: 0;
        gap: 10px;
    }

    .modal-title {
        font-size: 13px;
        font-weight: 600;
        color: #ccc;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .pop-badge {
        font-size: 10px; font-weight: 600;
        border-radius: 3px; padding: 1px 6px;
    }
    .pop-badge.exc { background: #1a2a3a; color: #4fc3f7; }
    .pop-badge.inh { background: #2a1a1a; color: #ef5350; }
    .electrode-label { font-size: 13px; font-weight: 700; color: #80cbc4; font-variant-numeric: tabular-nums; }

    .chart-kind { font-size: 11px; color: #666; font-weight: 400; }

    .close-btn {
        background: none; border: 1px solid #444; color: #888;
        font-size: 16px; width: 26px; height: 26px; border-radius: 3px;
        cursor: pointer; display: flex; align-items: center;
        justify-content: center; padding: 0;
    }
    .close-btn:hover { color: #ef5350; border-color: #ef5350; }

    .canvas-wrap {
        flex: 1;
        overflow: hidden;
        min-height: 0;
        position: relative;
    }

    canvas {
        display: block;
        width: 100%;
        height: 100%;
    }

    /* ── Play bar ── */
    .play-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        background: #10101e;
        border-top: 1px solid #2a2a4a;
        flex-shrink: 0;
    }

    .play-btn {
        background: none; border: 1px solid #444; color: #ccc;
        width: 28px; height: 28px; border-radius: 4px;
        cursor: pointer; font-size: 12px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; padding: 0;
    }
    .play-btn:hover { border-color: #66bb6a; color: #66bb6a; }

    .speed-group { display: flex; gap: 2px; flex-shrink: 0; }
    .speed-btn {
        background: none; border: 1px solid #333; color: #666;
        font-size: 10px; padding: 2px 5px; border-radius: 3px; cursor: pointer;
    }
    .speed-btn:hover { color: #bbb; border-color: #555; }
    .speed-btn.active { color: #ffd54f; border-color: #ffd54f; background: rgba(255,213,79,0.08); }

    .time-slider { flex: 1; accent-color: #ffd54f; min-width: 80px; }

    .time-label {
        font-size: 11px; color: #888;
        font-variant-numeric: tabular-nums;
        white-space: nowrap; min-width: 88px;
    }

    .spacer { flex: 0 0 8px; }

    .action-btn {
        background: none; border: 1px solid #333; color: #666;
        font-size: 10px; padding: 3px 8px; border-radius: 3px; cursor: pointer;
        white-space: nowrap;
    }
    .action-btn:hover { color: #bbb; border-color: #555; }

    .hint { font-size: 9px; color: #333; white-space: nowrap; }
</style>
