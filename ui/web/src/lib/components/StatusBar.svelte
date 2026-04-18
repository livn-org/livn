<script lang="ts">
    import {
        backendInfo,
        envSystem,
        envIO,
        envModel,
        lastExecTime,
        loading,
        snapshotLog,
    } from "$lib/stores";
    import { forceRefresh } from "$lib/pyodide";

    const backend = $derived($backendInfo);
    const system = $derived($envSystem);
    const io = $derived($envIO);
    const model = $derived($envModel);
    const execTime = $derived($lastExecTime);
    const isLoading = $derived($loading);
    const logs = $derived($snapshotLog);

    let showLog = $state(false);
</script>

<div class="statusbar">
    <span class="segment">
        <span class="label">Backend:</span>
        {backend}
    </span>

    <span class="segment">
        <span class="indicator" class:active={!!system}>&#9679;</span> system
        <span class="indicator" class:active={!!io}>&#9679;</span> io
        <span class="indicator" class:active={!!model}>&#9679;</span> model
    </span>

    {#if isLoading}
        <span class="segment loading">Running…</span>
    {:else if execTime !== null}
        <span class="segment">{execTime.toFixed(0)} ms</span>
    {/if}

    <span class="segment right">
        <button
            class="refresh-btn"
            onclick={() => forceRefresh()}
            title="Force snapshot refresh">⟳ Refresh</button
        >
        <button
            class="log-btn"
            onclick={() => (showLog = !showLog)}
            title="Toggle snapshot log"
        >
            {showLog ? "▾" : "▸"} Log ({logs.length})
        </button>
    </span>
</div>

{#if showLog}
    <div class="log-panel">
        {#if logs.length === 0}
            <span class="log-empty">No snapshot events yet</span>
        {:else}
            {#each logs as entry}
                <div class="log-entry">{entry}</div>
            {/each}
        {/if}
    </div>
{/if}

<style>
    .statusbar {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 4px 12px;
        background: #16162a;
        border-top: 1px solid #333;
        font-size: 12px;
        color: #888;
        flex-shrink: 0;
    }
    .segment {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .right {
        margin-left: auto;
    }
    .label {
        color: #666;
    }
    .indicator {
        color: #555;
        font-size: 8px;
    }
    .indicator.active {
        color: #4caf50;
    }
    .loading {
        color: #fdd835;
    }
    .refresh-btn,
    .log-btn {
        background: none;
        border: 1px solid #444;
        color: #aaa;
        font-size: 11px;
        padding: 1px 8px;
        cursor: pointer;
        border-radius: 3px;
    }
    .refresh-btn:hover,
    .log-btn:hover {
        background: #2a2a4a;
        color: #ddd;
    }
    .log-panel {
        background: #111;
        border-top: 1px solid #333;
        padding: 6px 12px;
        font-size: 11px;
        font-family: monospace;
        color: #999;
        max-height: 150px;
        overflow-y: auto;
    }
    .log-entry {
        padding: 1px 0;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .log-empty {
        color: #555;
        font-style: italic;
    }
</style>
