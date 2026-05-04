<script lang="ts">
    import { onDestroy } from 'svelte';
    import { selectedNeurons, activeExpRow, envSystem, selectedElectrode } from '$lib/stores';
    import { getExpRowData, getElectrodeData, exportChartPng, getAllRowSpikes, type RowData, type ElectrodeData } from '$lib/pyodide';
    import type { Experiment, SystemData } from '$lib/types';
    import ChartModal from './ChartModal.svelte';

    interface Props {
        experiment: Experiment;
        onBack: () => void;
    }
    let { experiment, onBack }: Props = $props();

    let rowData    = $state<RowData | null>(null);
    let loadError  = $state<string | null>(null);
    let timeCursor = $state(0);
    let playing    = $state(false);

    let modalGid          = $state<number | null>(null);
    let modalChartType    = $state<'spikes' | 'voltage'>('spikes');
    let electrodeModalType = $state<'electrode-spikes' | 'electrode-lfp' | null>(null);

    // Playback speed
    let playbackRate = $state(3);
    const SPEED_OPTIONS = [1, 3, 5, 10];

    // Raster
    let showRaster      = $state(false);
    let rasterAllNeurons = $state(false);
    let allSpikes        = $state<{ it: number[]; tt: number[]; duration: number } | null>(null);
    let allSpikesLoading = $state(false);

    // Electrode
    let electrodeData    = $state<ElectrodeData | null>(null);
    let electrodeLoading = $state(false);
    let electrodeError   = $state<string | null>(null);

    const noSystemMetadata = $derived(!experiment.metadata?.system?.uri);
    const duration = $derived(rowData?.duration ?? 1000);

    // ── Playback ───────────────────────────────────────────────────────────
    let animId: number | null = null;
    let lastTs = 0;

    function playFrame(ts: number) {
        if (!playing) return;
        if (lastTs > 0) {
            const next = timeCursor + (ts - lastTs) * playbackRate;
            if (next >= duration) {
                timeCursor = duration;
                stopPlay();
                return;
            }
            timeCursor = next;
        }
        lastTs = ts;
        animId = requestAnimationFrame(playFrame);
    }

    function togglePlay() {
        if (playing) {
            stopPlay();
        } else {
            timeCursor = 0;
            playing = true;
            lastTs  = 0;
            animId  = requestAnimationFrame(playFrame);
        }
    }

    function stopPlay() {
        playing = false;
        if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
    }

    onDestroy(() => {
        stopPlay();
        selectedElectrode.set(null);
    });

    $effect(() => {
        void $activeExpRow;
        stopPlay();
        timeCursor = 0;
    });

    // ── Data loading ───────────────────────────────────────────────────────
    async function fetchRowData() {
        if ($selectedNeurons.length === 0) { rowData = null; return; }
        try {
            rowData = await getExpRowData($activeExpRow, $selectedNeurons);
            loadError = null;
        } catch (e) {
            loadError = (e as Error).message;
        }
    }

    $effect(() => {
        void $activeExpRow;
        void $selectedNeurons.length;
        fetchRowData();
    });

    // ── All-spikes for raster ──────────────────────────────────────────────
    $effect(() => {
        const row = $activeExpRow;
        if (rasterAllNeurons) {
            allSpikesLoading = true;
            getAllRowSpikes(row)
                .then(d => { allSpikes = d; })
                .catch(() => { allSpikes = null; })
                .finally(() => { allSpikesLoading = false; });
        } else {
            allSpikes = null;
        }
    });

    // ── Electrode data ─────────────────────────────────────────────────────
    $effect(() => {
        const row = $activeExpRow;
        const id  = $selectedElectrode;
        if (id !== null) {
            electrodeLoading = true;
            electrodeError   = null;
            getElectrodeData(row, id)
                .then(d => { electrodeData = d; })
                .catch(e => { electrodeError = (e as Error).message; electrodeData = null; })
                .finally(() => { electrodeLoading = false; });
        } else {
            electrodeData  = null;
            electrodeError = null;
        }
    });

    // ── Helpers ────────────────────────────────────────────────────────────
    function systemLabel() {
        const uri = experiment.metadata?.system?.uri;
        return uri ? (uri.split('/').pop() ?? uri) : '—';
    }
    function modelLabel() { return experiment.metadata?.model ?? '—'; }

    function deselectNeuron(gid: number) {
        selectedNeurons.update(ns => ns.filter(n => n !== gid));
    }

    function popForGid(gid: number): string {
        const sys = $envSystem;
        if (!sys) return '';
        for (const [pop, coords] of Object.entries(sys.pop_coords))
            for (let i = 0; i < coords.length; i += 4)
                if (coords[i] === gid) return pop;
        return '';
    }

    function spikesFor(gid: number): number[] {
        if (!rowData) return [];
        return (rowData.spikes[String(gid) as unknown as number] ?? []) as number[];
    }

    function voltageFor(gid: number): number[] {
        if (!rowData) return [];
        return (rowData.voltages[String(gid) as unknown as number] ?? []) as number[];
    }

    function hasVoltage(gid: number): boolean {
        return voltageFor(gid).length > 0;
    }

    function openModal(gid: number, type: 'spikes' | 'voltage') {
        stopPlay();
        modalGid       = gid;
        modalChartType = type;
    }

    function openElectrodeModal(type: 'electrode-spikes' | 'electrode-lfp') {
        stopPlay();
        electrodeModalType = type;
    }

    // ── Export ─────────────────────────────────────────────────────────────
    async function exportChart(gid: number, chartType: 'spikes' | 'voltage') {
        try {
            const dataUrl = await exportChartPng(
                gid, popForGid(gid), chartType,
                spikesFor(gid), voltageFor(gid), duration
            );
            const a = document.createElement('a');
            a.href = dataUrl;
            a.download = `gid${gid}_${chartType}.png`;
            a.click();
        } catch (e) {
            console.error('Export failed:', e);
        }
    }

    // ── Raster helpers ─────────────────────────────────────────────────────
    function buildRasterNeurons(): { gid: number; pop: string }[] {
        if (rasterAllNeurons) {
            const sys = $envSystem;
            if (!sys) return [];
            const result: { gid: number; pop: string }[] = [];
            for (const [pop, coords] of Object.entries(sys.pop_coords)) {
                const count = Math.floor(coords.length / 4);
                for (let i = 0; i < count; i++) {
                    result.push({ gid: coords[i * 4], pop });
                }
            }
            result.sort((a, b) => a.gid - b.gid);
            return result;
        }
        return $selectedNeurons
            .map(gid => ({ gid, pop: popForGid(gid) }))
            .sort((a, b) => a.gid - b.gid);
    }

    const rasterNeurons  = $derived(buildRasterNeurons());
    const rasterCanvasH  = $derived(
        rasterAllNeurons
            ? 280
            : Math.min(280, Math.max(60, rasterNeurons.length * 14 + 20))
    );

    // ── Canvas drawing ─────────────────────────────────────────────────────
    function drawValueBadge(
        ctx: CanvasRenderingContext2D,
        cx: number, w: number, h: number,
        label: string, font: string
    ) {
        ctx.font = font;
        const tw = ctx.measureText(label).width;
        const pad = 4, bw = tw + pad * 2, bh = 16;
        const bx = cx + 3 + bw > w ? cx - bw - 3 : cx + 3;
        const by = Math.round((h - bh) / 2);
        ctx.fillStyle = 'rgba(8,8,20,0.92)';
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = '#ffd54f';
        ctx.fillText(label, bx + pad, by + bh - 4);
    }

    function drawSpike(canvas: HTMLCanvasElement, spikes: number[], dur: number, cursor: number) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, w, h);

        const axisH = 12;
        const plotH = h - axisH;
        ctx.strokeStyle = '#4fc3f7'; ctx.lineWidth = 1.5;
        for (const t of spikes) {
            const x = Math.round((t / dur) * w);
            ctx.beginPath(); ctx.moveTo(x, plotH - 1); ctx.lineTo(x, 2); ctx.stroke();
        }

        const cx = Math.round((cursor / dur) * w);
        ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, plotH); ctx.stroke();

        const n = spikes.filter(t => t <= cursor).length;
        drawValueBadge(ctx, cx, w, plotH, `${n} spk`, 'bold 11px monospace');

        ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, plotH); ctx.lineTo(w, plotH); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '10px monospace';
        ctx.fillText('0', 2, h - 1);
        ctx.fillText(`${dur}ms`, w - 32, h - 1);
    }

    function drawVoltage(canvas: HTMLCanvasElement, voltages: number[], dur: number, cursor: number) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, w, h);
        if (voltages.length < 2) return;

        const axisH = 12;
        const plotH = h - axisH;

        let min = voltages[0], max = voltages[0];
        for (const v of voltages) { if (v < min) min = v; if (v > max) max = v; }
        const range = max - min || 1;
        const step = Math.max(1, Math.floor(voltages.length / (w * 2)));
        ctx.strokeStyle = '#ffb74d'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < voltages.length; i += step) {
            const x = (i / (voltages.length - 1)) * w;
            const y = 2 + ((max - voltages[i]) / range) * (plotH - 4);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        const cx = Math.round((cursor / dur) * w);
        ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, plotH); ctx.stroke();

        const idx = Math.min(voltages.length - 1, Math.max(0, Math.round((cursor / dur) * (voltages.length - 1))));
        drawValueBadge(ctx, cx, w, plotH, `${voltages[idx].toFixed(1)}mV`, 'bold 11px monospace');

        ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, plotH); ctx.lineTo(w, plotH); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '10px monospace';
        ctx.fillText('0', 2, h - 1);
        ctx.fillText(`${dur}ms`, w - 32, h - 1);
    }

    const POP_COLORS: Record<string, string> = { EXC: '#4fc3f7', INH: '#ef5350' };

    function drawRaster(
        canvas: HTMLCanvasElement,
        neurons: { gid: number; pop: string }[],
        spikesMap: Record<number, number[]> | null,
        spikesAll: { it: number[]; tt: number[]; duration: number } | null,
        useAll: boolean,
        selected: number[],
        dur: number,
        cursor: number,
        sys: SystemData | null
    ) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0a0a18'; ctx.fillRect(0, 0, w, h);

        const axisH = 14;
        const innerH = h - axisH;

        if (useAll && spikesAll && spikesAll.it.length > 0) {
            const { it, tt } = spikesAll;
            let minG = it[0], maxG = it[0];
            for (const g of it) { if (g < minG) minG = g; if (g > maxG) maxG = g; }
            const gRange = maxG - minG || 1;

            const gidPop = new Map<number, string>();
            if (sys) {
                for (const [pop, coords] of Object.entries(sys.pop_coords)) {
                    const count = Math.floor(coords.length / 4);
                    for (let i = 0; i < count; i++) gidPop.set(coords[i * 4], pop);
                }
            }
            const selSet = new Set(selected);

            // Draw non-selected first, then selected on top
            for (const pass of [false, true]) {
                for (let k = 0; k < it.length; k++) {
                    const gid = it[k];
                    const isSel = selSet.has(gid);
                    if (isSel !== pass) continue;
                    const x = (tt[k] / dur) * w;
                    const y = innerH - ((gid - minG) / gRange) * innerH;
                    const pop = gidPop.get(gid) ?? '';
                    ctx.fillStyle = POP_COLORS[pop] ?? '#aaaaaa';
                    ctx.globalAlpha = isSel ? 1.0 : 0.35;
                    const r = isSel ? 2 : 1.5;
                    ctx.beginPath();
                    ctx.arc(x, y, r, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.globalAlpha = 1;

            ctx.fillStyle = '#999'; ctx.font = '10px monospace';
            ctx.fillText(String(maxG), 2, 11);
            ctx.fillText(String(minG), 2, innerH - 2);

            const cx = (cursor / dur) * w;
            ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, innerH); ctx.stroke();

            ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, innerH); ctx.lineTo(w, innerH); ctx.stroke();
            ctx.fillStyle = '#999'; ctx.font = '10px monospace';
            ctx.fillText('0', 2, h - 1);
            ctx.fillText(`${dur}ms`, w - 32, h - 1);

        } else if (!useAll) {
            if (neurons.length === 0) {
                ctx.fillStyle = '#555'; ctx.font = '11px monospace';
                ctx.textAlign = 'center';
                ctx.fillText('No neurons selected', w / 2, h / 2);
                ctx.textAlign = 'left';
                return;
            }

            const labelW = 48;
            const plotW  = w - labelW;
            const rowH   = innerH / neurons.length;
            const tickH  = Math.max(3, Math.min(rowH - 2, 12));

            // Row dividers
            ctx.strokeStyle = '#2e2e50'; ctx.lineWidth = 1;
            for (let i = 1; i < neurons.length; i++) {
                const y = Math.round(i * rowH);
                ctx.beginPath(); ctx.moveTo(labelW, y); ctx.lineTo(w, y); ctx.stroke();
            }

            for (let i = 0; i < neurons.length; i++) {
                const { gid, pop } = neurons[i];
                const spikes = spikesMap
                    ? ((spikesMap[String(gid) as any] ?? []) as number[])
                    : [];
                const rowTop = i * rowH;
                const mid    = rowTop + rowH / 2;
                ctx.strokeStyle = POP_COLORS[pop] ?? '#aaaaaa';
                ctx.lineWidth = 1.5;
                for (const t of spikes) {
                    const x = labelW + (t / dur) * plotW;
                    ctx.beginPath(); ctx.moveTo(x, mid - tickH / 2); ctx.lineTo(x, mid + tickH / 2); ctx.stroke();
                }
                ctx.fillStyle = '#aaa'; ctx.font = '10px monospace';
                ctx.textAlign = 'right';
                ctx.fillText(String(gid), labelW - 3, mid + 4);
                ctx.textAlign = 'left';
            }

            const cx2 = labelW + (cursor / dur) * plotW;
            ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx2, 0); ctx.lineTo(cx2, innerH); ctx.stroke();

            ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(labelW, innerH); ctx.lineTo(w, innerH); ctx.stroke();
            ctx.fillStyle = '#999'; ctx.font = '10px monospace';
            ctx.fillText('0', labelW + 2, h - 1);
            ctx.fillText(`${dur}ms`, w - 32, h - 1);
        }
    }

    function drawElectrodeSpikes(canvas: HTMLCanvasElement, data: ElectrodeData, cursor: number) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, w, h);

        const axisH = 12;
        const plotH = h - axisH;
        ctx.strokeStyle = '#80cbc4'; ctx.lineWidth = 1.5;
        for (const t of data.spikeTimes) {
            const x = Math.round((t / data.duration) * w);
            ctx.beginPath(); ctx.moveTo(x, plotH - 1); ctx.lineTo(x, 2); ctx.stroke();
        }
        const n = data.spikeTimes.filter(t => t <= cursor).length;
        const cx = Math.round((cursor / data.duration) * w);
        ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, plotH); ctx.stroke();
        drawValueBadge(ctx, cx, w, plotH, `${n} spk`, 'bold 11px monospace');

        ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, plotH); ctx.lineTo(w, plotH); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '10px monospace';
        ctx.fillText('0', 2, h - 1);
        ctx.fillText(`${data.duration}ms`, w - 32, h - 1);
    }

    function drawElectrodeLfp(canvas: HTMLCanvasElement, data: ElectrodeData, cursor: number) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#0d0d1a'; ctx.fillRect(0, 0, w, h);

        const vs = data.lfp;
        if (vs.length > 1) {
            const axisH = 12;
            const plotH = h - axisH;

            let min = vs[0], max = vs[0];
            for (const v of vs) { if (v < min) min = v; if (v > max) max = v; }
            const range = max - min || 1;
            const step = Math.max(1, Math.floor(vs.length / (w * 2)));
            ctx.strokeStyle = '#80cbc4'; ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < vs.length; i += step) {
                const x = (i / (vs.length - 1)) * w;
                const y = 2 + ((max - vs[i]) / range) * (plotH - 4);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();

            const cx = Math.round((cursor / data.duration) * w);
            ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, plotH); ctx.stroke();
            const idx = Math.min(vs.length - 1, Math.max(0, Math.round((cursor / data.duration) * (vs.length - 1))));
            drawValueBadge(ctx, cx, w, plotH, `${vs[idx].toFixed(2)}µV`, 'bold 11px monospace');

            ctx.strokeStyle = '#4a4a6a'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, plotH); ctx.lineTo(w, plotH); ctx.stroke();
            ctx.fillStyle = '#999'; ctx.font = '10px monospace';
            ctx.fillText('0', 2, h - 1);
            ctx.fillText(`${data.duration}ms`, w - 32, h - 1);
        }
    }

    // ── Svelte use-actions ─────────────────────────────────────────────────
    type SpikeArgs = { gid: number; rowData: RowData | null; cursor: number };
    function spikeAction(canvas: HTMLCanvasElement, args: SpikeArgs) {
        const render = () => { if (args.rowData) drawSpike(canvas, spikesFor(args.gid), args.rowData.duration, args.cursor); };
        render();
        return { update(a: SpikeArgs) { args = a; render(); } };
    }

    type VoltArgs = { gid: number; rowData: RowData | null; cursor: number };
    function voltAction(canvas: HTMLCanvasElement, args: VoltArgs) {
        const render = () => { if (args.rowData) drawVoltage(canvas, voltageFor(args.gid), args.rowData.duration, args.cursor); };
        render();
        return { update(a: VoltArgs) { args = a; render(); } };
    }

    type RasterArgs = {
        neurons: { gid: number; pop: string }[];
        spikesMap: Record<number, number[]> | null;
        allSpikes: typeof allSpikes;
        useAll: boolean;
        selected: number[];
        cursor: number;
        duration: number;
        sys: SystemData | null;
    };
    function rasterAction(canvas: HTMLCanvasElement, args: RasterArgs) {
        const render = () => drawRaster(
            canvas, args.neurons, args.spikesMap, args.allSpikes,
            args.useAll, args.selected, args.duration, args.cursor, args.sys
        );
        render();
        return { update(a: RasterArgs) { args = a; render(); } };
    }

    type ElectrodeArgs = { data: ElectrodeData | null; cursor: number };
    function electrodeSpikesAction(canvas: HTMLCanvasElement, args: ElectrodeArgs) {
        const render = () => { if (args.data) drawElectrodeSpikes(canvas, args.data, args.cursor); };
        render();
        return { update(a: ElectrodeArgs) { args = a; render(); } };
    }
    function electrodeLfpAction(canvas: HTMLCanvasElement, args: ElectrodeArgs) {
        const render = () => { if (args.data) drawElectrodeLfp(canvas, args.data, args.cursor); };
        render();
        return { update(a: ElectrodeArgs) { args = a; render(); } };
    }
</script>

<!-- ── Modal ─────────────────────────────────────────────── -->
{#if modalGid !== null && rowData}
    <ChartModal
        gid={modalGid}
        pop={popForGid(modalGid)}
        chartType={modalChartType}
        {rowData}
        initialCursor={timeCursor}
        onClose={() => { modalGid = null; }}
    />
{/if}
{#if electrodeModalType !== null && electrodeData}
    <ChartModal
        chartType={electrodeModalType}
        electrodeData={electrodeData}
        channelId={$selectedElectrode ?? undefined}
        initialCursor={timeCursor}
        onClose={() => { electrodeModalType = null; }}
    />
{/if}

<div class="panel">
    <!-- Header -->
    <div class="panel-header">
        <button class="back-btn" onclick={onBack}>← Back</button>
        <span class="exp-name">{experiment.name}</span>
        <div class="exp-meta">
            <span class="meta-tag">system: {systemLabel()}</span>
            <span class="meta-tag">model: {modelLabel()}</span>
        </div>
    </div>

    {#if noSystemMetadata}
        <div class="warning-bar">No system metadata — 3D scene unavailable. Charts still work.</div>
    {/if}

    <!-- Controls bar -->
    <div class="controls-bar">
        <div class="trial-picker">
            <button
                class="trial-btn"
                disabled={$activeExpRow === 0}
                onclick={() => activeExpRow.update(r => Math.max(0, r - 1))}
            >‹</button>
            <span class="trial-label">Trial {$activeExpRow + 1}/{experiment.n_shards}</span>
            <button
                class="trial-btn"
                disabled={$activeExpRow >= experiment.n_shards - 1}
                onclick={() => activeExpRow.update(r => Math.min(experiment.n_shards - 1, r + 1))}
            >›</button>
        </div>

        <div class="time-divider"></div>

        <button class="play-btn" onclick={togglePlay} title={playing ? 'Pause' : 'Play from start'}>
            <span class="play-icon">{playing ? '⏸' : '▶'}</span>
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
            max={duration}
            step="1"
            value={timeCursor}
            oninput={(e) => {
                stopPlay();
                timeCursor = parseFloat((e.target as HTMLInputElement).value);
            }}
        />

        <span class="time-label">{timeCursor.toFixed(0)} / {duration} ms</span>

        <div class="time-divider"></div>

        <label class="toggle-label" title="Show spike raster">
            <input type="checkbox" bind:checked={showRaster} />
            <span>Raster</span>
        </label>

        {#if showRaster}
            <label class="toggle-label" title="Show all neurons (scatter)">
                <input type="checkbox" bind:checked={rasterAllNeurons} />
                <span>All</span>
            </label>
        {/if}
    </div>

    <div class="divider"></div>

    <!-- Spike raster -->
    {#if showRaster}
        <div class="raster-wrap">
            <div class="raster-hdr">
                <span class="section-title">Spike Raster</span>
                {#if rasterAllNeurons}
                    <span class="raster-hint">scatter · colored by population · selected neurons highlighted</span>
                {/if}
                {#if allSpikesLoading}
                    <span class="loading-tag">loading…</span>
                {/if}
            </div>
            <canvas
                class="raster-canvas"
                width="800"
                height={rasterCanvasH}
                style:height="{rasterCanvasH}px"
                use:rasterAction={{
                    neurons: rasterNeurons,
                    spikesMap: rowData?.spikes ?? null,
                    allSpikes,
                    useAll: rasterAllNeurons && !!allSpikes,
                    selected: $selectedNeurons,
                    cursor: timeCursor,
                    duration,
                    sys: $envSystem,
                }}
            ></canvas>
        </div>
        <div class="divider"></div>
    {/if}

    <!-- Neuron charts -->
    <div class="neurons-area">
        {#if $selectedNeurons.length === 0 && $selectedElectrode === null}
            <div class="empty-msg">No neurons selected — click a neuron in the 3D view</div>
        {:else}
            {#if loadError}
                <div class="load-error">{loadError}</div>
            {/if}

            {#each $selectedNeurons as gid (gid)}
                {@const pop = popForGid(gid)}
                <div class="neuron-row">
                    <div class="neuron-label">
                        <span class="gid">GID {gid}</span>
                        {#if pop}
                            <span class="pop-tag" class:exc={pop === 'EXC'} class:inh={pop === 'INH'}>{pop}</span>
                        {/if}
                        <button class="desel-btn" onclick={() => deselectNeuron(gid)} title="Deselect">×</button>
                    </div>

                    <div class="charts">
                        <div class="chart-wrap">
                            <div class="chart-header">
                                <span class="chart-title">spikes</span>
                                <div class="chart-actions">
                                    <button class="action-icon-btn" onclick={() => exportChart(gid, 'spikes')} title="Export PNG">↓</button>
                                    <button class="expand-btn" onclick={() => openModal(gid, 'spikes')} title="Expand">⤢</button>
                                </div>
                            </div>
                            <canvas
                                width="400" height="44"
                                use:spikeAction={{ gid, rowData, cursor: timeCursor }}
                            ></canvas>
                        </div>

                        {#if hasVoltage(gid)}
                            <div class="chart-wrap">
                                <div class="chart-header">
                                    <span class="chart-title">voltage</span>
                                    <div class="chart-actions">
                                        <button class="action-icon-btn" onclick={() => exportChart(gid, 'voltage')} title="Export PNG">↓</button>
                                        <button class="expand-btn" onclick={() => openModal(gid, 'voltage')} title="Expand">⤢</button>
                                    </div>
                                </div>
                                <canvas
                                    width="400" height="60"
                                    use:voltAction={{ gid, rowData, cursor: timeCursor }}
                                ></canvas>
                            </div>
                        {/if}
                    </div>
                </div>
            {/each}

            <!-- Electrode panel -->
            {#if $selectedElectrode !== null}
                <div class="electrode-row">
                    <div class="neuron-label">
                        <span class="electrode-id">CH {$selectedElectrode}</span>
                        <span class="pop-tag" style="background:#1a2a1a;color:#80cbc4;">MEA</span>
                        <button class="desel-btn" onclick={() => selectedElectrode.set(null)} title="Close">×</button>
                    </div>
                    <div class="charts">
                        {#if electrodeLoading}
                            <div class="electrode-loading">Loading electrode data…</div>
                        {:else if electrodeError}
                            <div class="load-error">{electrodeError}</div>
                        {:else if electrodeData}
                            <div class="chart-wrap">
                                <div class="chart-header">
                                    <span class="chart-title">channel spikes</span>
                                    <button class="expand-btn" onclick={() => openElectrodeModal('electrode-spikes')} title="Expand">⤢</button>
                                </div>
                                <canvas
                                    width="400" height="44"
                                    use:electrodeSpikesAction={{ data: electrodeData, cursor: timeCursor }}
                                ></canvas>
                            </div>
                            {#if electrodeData.hasLfp}
                                <div class="chart-wrap">
                                    <div class="chart-header">
                                        <span class="chart-title">LFP (µV)</span>
                                        <button class="expand-btn" onclick={() => openElectrodeModal('electrode-lfp')} title="Expand">⤢</button>
                                    </div>
                                    <canvas
                                        width="400" height="60"
                                        use:electrodeLfpAction={{ data: electrodeData, cursor: timeCursor }}
                                    ></canvas>
                                </div>
                            {/if}
                        {:else}
                            <div class="electrode-loading">Select an experiment trial to load electrode data.</div>
                        {/if}
                    </div>
                </div>
            {/if}
        {/if}
    </div>
</div>

<style>
    .panel {
        display: flex;
        flex-direction: column;
        height: 100%;
        overflow: hidden;
        font-size: 12px;
        color: #ccc;
    }

    /* ── Header ── */
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: #0d0d1a;
        border-bottom: 1px solid #333;
        flex-shrink: 0;
        flex-wrap: wrap;
    }
    .back-btn {
        background: none; border: 1px solid #444; color: #888;
        font-size: 11px; padding: 2px 8px; border-radius: 3px; cursor: pointer; flex-shrink: 0;
    }
    .back-btn:hover { color: #ccc; border-color: #666; }
    .exp-name { font-weight: 700; color: #66bb6a; font-size: 13px; flex-shrink: 0; }
    .exp-meta { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
    .meta-tag {
        font-size: 10px; color: #555;
        background: #16162a; border: 1px solid #2a2a4a;
        border-radius: 3px; padding: 1px 6px;
    }

    .warning-bar {
        background: #2a2000; border-bottom: 1px solid #665500;
        color: #ffd54f; font-size: 11px; padding: 6px 12px; flex-shrink: 0;
    }

    /* ── Controls bar ── */
    .controls-bar {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: #10101e;
        border-bottom: 1px solid #2a2a4a;
        flex-shrink: 0;
        flex-wrap: wrap;
    }

    .trial-picker { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
    .trial-btn {
        background: none; border: 1px solid #333; color: #888;
        font-size: 14px; width: 22px; height: 22px; border-radius: 3px;
        cursor: pointer; display: flex; align-items: center; justify-content: center;
        padding: 0; line-height: 1;
    }
    .trial-btn:hover:not(:disabled) { color: #ccc; border-color: #666; }
    .trial-btn:disabled { opacity: 0.3; cursor: default; }
    .trial-label { font-size: 11px; color: #888; white-space: nowrap; }

    .time-divider { width: 1px; height: 18px; background: #2a2a4a; flex-shrink: 0; }

    .play-btn {
        background: none; border: 1px solid #444; color: #ccc;
        width: 28px; height: 28px; border-radius: 4px; cursor: pointer;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0; padding: 0;
    }
    .play-btn:hover { border-color: #66bb6a; color: #66bb6a; }
    .play-icon { font-size: 12px; line-height: 1; }

    .speed-group { display: flex; gap: 2px; flex-shrink: 0; }
    .speed-btn {
        background: none; border: 1px solid #333; color: #666;
        font-size: 10px; padding: 2px 5px; border-radius: 3px; cursor: pointer; white-space: nowrap;
    }
    .speed-btn:hover { color: #bbb; border-color: #555; }
    .speed-btn.active { color: #ffd54f; border-color: #ffd54f; background: rgba(255,213,79,0.08); }

    .time-slider { flex: 1; min-width: 60px; accent-color: #ffd54f; }
    .time-label {
        font-size: 11px; color: #888; white-space: nowrap;
        font-variant-numeric: tabular-nums; min-width: 90px; text-align: right;
    }

    .toggle-label {
        display: flex; align-items: center; gap: 4px;
        color: #888; font-size: 11px; cursor: pointer; white-space: nowrap; flex-shrink: 0;
    }
    .toggle-label input { cursor: pointer; accent-color: #4fc3f7; }

    .divider { height: 1px; background: #2a2a4a; flex-shrink: 0; }

    /* ── Raster ── */
    .raster-wrap {
        flex-shrink: 0;
        background: #0a0a18;
    }
    .raster-hdr {
        display: flex; align-items: center; gap: 8px;
        padding: 3px 12px;
    }
    .section-title { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
    .raster-hint  { font-size: 9px; color: #444; }
    .loading-tag  { font-size: 9px; color: #666; }
    .raster-canvas { width: 100%; display: block; }

    /* ── Neurons area ── */
    .neurons-area {
        flex: 1; overflow-y: auto; padding: 10px 12px;
        display: flex; flex-direction: column; gap: 14px;
    }

    .empty-msg { color: #555; font-style: italic; font-size: 12px; text-align: center; margin-top: 32px; }
    .load-error { color: #ef5350; font-size: 11px; padding: 6px; background: #1a0000; border-radius: 4px; }

    .neuron-row {
        display: flex; gap: 10px; align-items: flex-start;
        border-bottom: 1px solid #1a1a2e; padding-bottom: 10px;
    }

    .neuron-label {
        display: flex; flex-direction: column; align-items: flex-start; gap: 4px;
        min-width: 58px; flex-shrink: 0;
    }
    .gid { font-size: 12px; font-weight: 700; color: #ffd54f; font-variant-numeric: tabular-nums; }
    .pop-tag { font-size: 9px; border-radius: 2px; padding: 1px 5px; font-weight: 600; }
    .pop-tag.exc { background: #1a2a3a; color: #4fc3f7; }
    .pop-tag.inh { background: #2a1a1a; color: #ef5350; }
    .desel-btn {
        background: none; border: 1px solid #333; color: #666;
        font-size: 13px; width: 20px; height: 20px; border-radius: 3px;
        cursor: pointer; display: flex; align-items: center; justify-content: center;
        padding: 0; line-height: 1;
    }
    .desel-btn:hover { color: #ef5350; border-color: #ef5350; }

    .charts { display: flex; flex-direction: column; gap: 5px; flex: 1; min-width: 0; }

    .chart-wrap { display: flex; flex-direction: column; gap: 2px; }
    .chart-header { display: flex; justify-content: space-between; align-items: center; }
    .chart-title { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
    .chart-actions { display: flex; align-items: center; gap: 3px; }

    .action-icon-btn {
        background: none; border: none; color: #444;
        font-size: 13px; cursor: pointer; padding: 0; line-height: 1;
    }
    .action-icon-btn:hover { color: #4fc3f7; }

    .expand-btn {
        background: none; border: none; color: #444;
        font-size: 13px; cursor: pointer; padding: 0; line-height: 1;
        display: flex; align-items: center;
    }
    .expand-btn:hover { color: #aaa; }

    .chart-wrap canvas { width: 100%; height: auto; border-radius: 3px; display: block; }

    /* ── Electrode row ── */
    .electrode-row {
        display: flex; gap: 10px; align-items: flex-start;
        border-top: 1px solid #2a2a4a; padding-top: 10px;
    }
    .electrode-id { font-size: 12px; font-weight: 700; color: #80cbc4; font-variant-numeric: tabular-nums; }
    .electrode-loading { font-size: 11px; color: #666; font-style: italic; padding: 4px 0; }
</style>
