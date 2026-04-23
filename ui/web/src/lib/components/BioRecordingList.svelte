<script lang="ts">
    let { onSelect }: { onSelect: (id: string) => void } = $props();

    type Recording = {
        id: string;
        name: string;
        date: string;
        dur: string;
        protocol: string;
        real?: true;
    };

    const RECORDINGS: Recording[] = [
        {
            id: 'demo-neural1',
            name: 'Demo Recording',
            date: '—',
            dur: '1 min 0 s',
            protocol: 'LFP (θ 8 Hz + γ 40 Hz) + TTL stim @ 1 Hz',
            real: true,
        },
        {
            id: 'kimia-rec1',
            name: 'Kimia – Recording 1',
            date: '2025-08-22',
            dur: '5 min 7 s',
            protocol: 'TTL stimulation @ 1 Hz',
            real: true,
        },
    ];
</script>

<div class="list">
    <h2>Biological Recordings</h2>
    <div class="grid">
        {#each RECORDINGS as rec (rec.id)}
            <button class="card" class:real={rec.real} onclick={() => onSelect(rec.id)}>
                <div class="card-header">
                    <div class="card-name">{rec.name}</div>
                    {#if rec.real}
                        <span class="badge real-badge">real data</span>
                    {:else}
                        <span class="badge placeholder-badge">placeholder</span>
                    {/if}
                </div>
                <div class="card-meta">{rec.date} · {rec.dur}</div>
                <div class="card-proto">{rec.protocol}</div>
            </button>
        {/each}
    </div>
</div>

<style>
    .list { padding: 32px; overflow-y: auto; height: 100%; }
    h2 { font-size: 18px; font-weight: 600; color: #e0e0e0; margin-bottom: 24px; }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 16px;
    }
    .card {
        background: #16162a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px 16px;
        text-align: left;
        cursor: pointer;
        color: inherit;
        font-family: inherit;
        transition: border-color 0.15s, background 0.15s;
    }
    .card:hover        { border-color: #4fc3f7; background: #1e1e3a; }
    .card.real         { border-color: #2a4a2a; }
    .card.real:hover   { border-color: #66bb6a; background: #162216; }

    .card-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px; }
    .card-name   { font-size: 15px; font-weight: 600; color: #e0e0e0; }
    .card-meta   { font-size: 11px; color: #888; margin-bottom: 6px; }
    .card-proto  { font-size: 12px; color: #aaa; }

    .badge {
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 3px;
        padding: 2px 6px;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .real-badge        { background: #1a3a1a; border: 1px solid #66bb6a; color: #66bb6a; }
    .placeholder-badge { background: #1a1a2e; border: 1px solid #555; color: #666; }
</style>
