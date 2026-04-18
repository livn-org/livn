<script lang="ts">
    import { tooltip as tooltipStore } from "$lib/stores";

    const data = $derived($tooltipStore);

    function dismiss() {
        tooltipStore.update((t) => ({ ...t, visible: false }));
    }
</script>

{#if data.visible}
    <div class="tooltip" role="dialog" aria-label="Neuron info">
        <button class="close" onclick={dismiss} aria-label="Close"
            >&times;</button
        >
        <div class="row"><span class="label">GID:</span> {data.gid}</div>
        <div class="row">
            <span class="label">Population:</span>
            {data.population}
        </div>
        <div class="row">
            <span class="label">Position:</span>
            ({data.x.toFixed(1)}, {data.y.toFixed(1)}, {data.z.toFixed(1)})
        </div>
        {#if data.nearestElectrode}
            <div class="row">
                <span class="label">Nearest electrode:</span>
                #{data.nearestElectrode.id} ({data.nearestElectrode.distance.toFixed(
                    1,
                )} µm)
            </div>
        {/if}
    </div>
{/if}

<style>
    .tooltip {
        position: absolute;
        bottom: 12px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(30, 30, 50, 0.95);
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px 20px;
        color: #e0e0e0;
        font-size: 13px;
        z-index: 100;
        display: flex;
        gap: 16px;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
    .label {
        color: #888;
        margin-right: 4px;
    }
    .close {
        background: none;
        border: none;
        color: #888;
        font-size: 18px;
        cursor: pointer;
        padding: 0 4px;
    }
    .close:hover {
        color: #fff;
    }
</style>
