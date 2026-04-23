<script lang="ts">
    import SignalViewer from './SignalViewer.svelte';

    let { recordingId, onBack }: { recordingId: string; onBack: () => void } = $props();

    type PlaceholderInfo = {
        kind: 'placeholder';
        name: string; date: string; dur: string; protocol: string;
        stimulation: string; amplitude: string;
        shape: string; dimensions: string; neuronTypes: string;
    };

    type RealInfo = {
        kind: 'real';
        name: string; date: string; dur: string; durS: number; protocol: string;
        source: string; channels: number; sampleRate: string; bitVolts: string;
        guiVersion: string; format: string; path: string; apiPath: string;
        ttlCount: number; ttlRate: string; ttlISI: string;
    };

    type Info = PlaceholderInfo | RealInfo;

    const DETAILS: Record<string, Info> = {
        // ── Real recordings ──────────────────────────────────────────────
        'demo-neural1': {
            kind: 'real',
            name: 'Demo Recording',
            date: '—', dur: '1 min 0 s (60 s)', durS: 60,
            protocol: 'LFP (θ 8 Hz + γ 40 Hz) + TTL stim @ 1 Hz',
            source: 'Demo data (generated on-the-fly, no file download needed)',
            channels: 512, sampleRate: '30,000 Hz',
            bitVolts: '—',
            guiVersion: '—',
            format: 'Generated in Vite dev server',
            path: '—',
            apiPath: 'demo/neural1',
            ttlCount: 60, ttlRate: '~1 Hz', ttlISI: '1.000 s (mean)',
        },
    };

    const rec = $derived(DETAILS[recordingId] ?? DETAILS['rec-001']);
    let setupOpen = $state(false);
    let dataOpen  = $state(false);
    let stimOpen  = $state(false);

    // SVG placeholder waveform (used for placeholder recordings only)
    const W = 600, H = 100;
    const wavePts = Array.from({ length: W / 2 + 1 }, (_, i) => {
        const x = i * 2;
        const y = H / 2
            + Math.sin(x * 0.04) * 28
            + Math.sin(x * 0.09 + 1.2) * 14
            + Math.sin(x * 0.02 - 0.5) * 10;
        return `${x},${y.toFixed(1)}`;
    }).join(' ');
</script>

<div class="detail">
    <!-- Header -->
    <div class="header">
        <button class="back" onclick={onBack}>← Back</button>
        <span class="title">{rec.name}</span>
        {#if rec.kind === 'real'}
            <span class="real-tag">real data</span>
        {:else}
            <span class="ph-tag">placeholder</span>
        {/if}
    </div>

    <!-- Signal area (fills available height) -->
    <div class="signal-area">
        {#if rec.kind === 'real'}
            <SignalViewer
                rec={rec.apiPath}
                totalDuration={rec.durS}
                totalChannels={rec.channels}
            />
        {:else}
            <div class="wave-card">
                <div class="panel-label">Raw signal (placeholder waveform)</div>
                <svg viewBox="0 0 {W} {H}" class="wave" preserveAspectRatio="none">
                    <polyline points={wavePts} fill="none" stroke="#4fc3f7" stroke-width="1.5" />
                </svg>
                <div class="time-axis">
                    <span>0 s</span><span>2.5 s</span><span>5 s</span>
                    <span>7.5 s</span><span>10 s</span>
                </div>
            </div>
        {/if}
    </div>

    <!-- Accordion panels -->
    <div class="panels">
        <div class="section">
            <button class="sec-hdr" onclick={() => (setupOpen = !setupOpen)}>
                <span>Setup</span>
                <span class="chevron" class:open={setupOpen}>▶</span>
            </button>
            {#if setupOpen}
                <div class="sec-body">
                    {#if rec.kind === 'real'}
                        <div class="row"><span>Source</span><span>{rec.source}</span></div>
                        <div class="row"><span>Channels</span><span>{rec.channels}</span></div>
                        <div class="row"><span>Sample rate</span><span>{rec.sampleRate}</span></div>
                        <div class="row"><span>Resolution</span><span>{rec.bitVolts}</span></div>
                        <div class="row"><span>Format</span><span>{rec.format}</span></div>
                        <div class="row"><span>GUI version</span><span>{rec.guiVersion}</span></div>
                    {:else}
                        <div class="row"><span>Culture shape</span><span>{rec.shape}</span></div>
                        <div class="row"><span>Dimensions</span><span>{rec.dimensions}</span></div>
                        <div class="row"><span>Neuron types</span><span>{rec.neuronTypes}</span></div>
                    {/if}
                </div>
            {/if}
        </div>

        <div class="section">
            <button class="sec-hdr" onclick={() => (dataOpen = !dataOpen)}>
                <span>Data Info</span>
                <span class="chevron" class:open={dataOpen}>▶</span>
            </button>
            {#if dataOpen}
                <div class="sec-body">
                    <div class="row"><span>Recording date</span><span>{rec.date}</span></div>
                    <div class="row"><span>Duration</span><span>{rec.dur}</span></div>
                    {#if rec.kind === 'real'}
                        <div class="row"><span>Path</span><span class="mono">{rec.path}</span></div>
                    {:else}
                        <div class="row"><span>Protocol</span><span>{rec.protocol}</span></div>
                        <div class="row"><span>Stimulation</span><span>{rec.stimulation}</span></div>
                        <div class="row"><span>Amplitude</span><span>{rec.amplitude}</span></div>
                    {/if}
                </div>
            {/if}
        </div>
        {#if rec.kind === 'real'}
        <div class="section">
            <button class="sec-hdr" onclick={() => (stimOpen = !stimOpen)}>
                <span>Stimulation Protocol</span>
                <span class="chevron" class:open={stimOpen}>▶</span>
            </button>
            {#if stimOpen}
                <div class="sec-body">
                    <div class="row"><span>TTL pulses</span><span>{rec.ttlCount} ON events</span></div>
                    <div class="row"><span>Rate</span><span>{rec.ttlRate}</span></div>
                    <div class="row"><span>ISI</span><span>{rec.ttlISI}</span></div>
                </div>
            {/if}
        </div>
        {/if}
    </div>
</div>

<style>
    .detail { height: 100%; display: flex; flex-direction: column; overflow: hidden; }

    /* Header */
    .header {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 24px; border-bottom: 1px solid #333;
        background: #0d0d1a; flex-shrink: 0;
    }
    .back {
        background: none; border: 1px solid #444; color: #888;
        border-radius: 4px; padding: 3px 10px; font-size: 12px; cursor: pointer;
    }
    .back:hover { color: #ccc; border-color: #666; }
    .title { font-size: 15px; font-weight: 600; color: #e0e0e0; }
    .real-tag {
        font-size: 9px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        background: #1a3a1a; border: 1px solid #66bb6a; color: #66bb6a;
        border-radius: 3px; padding: 2px 6px;
    }
    .ph-tag {
        font-size: 9px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        background: #1a1a2e; border: 1px solid #555; color: #666;
        border-radius: 3px; padding: 2px 6px;
    }

    /* Signal area: fills remaining height */
    .signal-area {
        flex: 1;
        min-height: 0;
        overflow: hidden;
    }

    /* Placeholder waveform */
    .wave-card {
        height: 100%;
        box-sizing: border-box;
        padding: 16px 24px;
        display: flex;
        flex-direction: column;
    }
    .panel-label { font-size: 11px; color: #666; margin-bottom: 8px; flex-shrink: 0; }
    .wave {
        flex: 1;
        min-height: 0;
        display: block;
        background: #0d0d1a;
        border-radius: 4px;
        border: 1px solid #2a2a4a;
    }
    .time-axis {
        display: flex; justify-content: space-between;
        font-size: 10px; color: #555; margin-top: 4px; flex-shrink: 0;
    }

    /* Accordion panels — fixed at bottom, scroll internally */
    .panels {
        flex-shrink: 0;
        max-height: 220px;
        overflow-y: auto;
        border-top: 1px solid #2a2a4a;
    }
    .section { background: #0d0d1a; border-bottom: 1px solid #1a1a2e; }
    .sec-hdr {
        width: 100%; display: flex; justify-content: space-between; align-items: center;
        padding: 9px 16px; background: none; border: none;
        color: #e0e0e0; font-size: 12px; font-weight: 600; cursor: pointer; text-align: left;
    }
    .sec-hdr:hover { background: rgba(255,255,255,0.03); }
    .chevron { font-size: 9px; color: #555; transition: transform 0.15s; }
    .chevron.open { transform: rotate(90deg); }
    .sec-body { padding: 0 16px 10px; }
    .row {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 4px 0; font-size: 12px; border-bottom: 1px solid #1a1a2e;
    }
    .row:last-child { border-bottom: none; }
    .row span:first-child { color: #888; flex-shrink: 0; margin-right: 8px; }
    .row span:last-child  { color: #e0e0e0; text-align: right; }
    .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 11px; }
</style>
