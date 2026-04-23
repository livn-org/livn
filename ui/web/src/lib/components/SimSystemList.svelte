<script lang="ts">
    let { onSelect }: { onSelect: (system: string) => void } = $props();

    const SYSTEMS = [
        { id: 'EI1',  name: 'EI1',  desc: 'Excitatory-Inhibitory network, type 1', pops: 'EXC, INH' },
        { id: 'EI2',  name: 'EI2',  desc: 'Excitatory-Inhibitory network, type 2', pops: 'EXC, INH' },
        { id: 'CA1d', name: 'CA1d', desc: 'CA1 pyramidal layer, dorsal',            pops: 'PYR, INT' },
    ];
</script>

<div class="list">
    <h2>Neural Systems</h2>
    <div class="grid">
        {#each SYSTEMS as sys (sys.id)}
            <button class="card" onclick={() => onSelect(sys.id)}>
                <div class="card-name">{sys.name}</div>
                <div class="card-desc">{sys.desc}</div>
                <div class="card-tag">{sys.pops}</div>
            </button>
        {/each}
    </div>
</div>

<style>
    .list { padding: 32px; overflow-y: auto; height: 100%; }
    h2 { font-size: 18px; font-weight: 600; color: #e0e0e0; margin-bottom: 24px; }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
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
    .card:hover { border-color: #4fc3f7; background: #1e1e3a; }
    .card-name { font-size: 22px; font-weight: 700; color: #4fc3f7; margin-bottom: 8px; }
    .card-desc { font-size: 12px; color: #aaa; margin-bottom: 14px; }
    .card-tag {
        display: inline-block;
        background: #112238;
        border: 1px solid #4fc3f7;
        color: #4fc3f7;
        font-size: 10px;
        border-radius: 3px;
        padding: 2px 7px;
    }
</style>
