<script lang="ts">
    import { hsdsConnected, pyodideReady, pendingCommand } from "$lib/stores";

    const PREDEFINED = ["EI1", "EI2", "CA1d"];

    let selected = $state("predefined:EI1");
    let hsdsSystems = $state<string[]>([]);
    let loading = $state(false);

    const isReady = $derived($pyodideReady);
    const hasHsds = $derived($hsdsConnected);

    // Probe HSDS for available systems when connected
    $effect(() => {
        if (hasHsds) {
            fetchHsdsSystems();
        }
    });

    async function fetchHsdsSystems() {
        try {
            const resp = await fetch("/hsds/domains?domain=/");
            if (!resp.ok) return;
            const data = await resp.json();
            const domains: string[] = [];
            if (data.domains) {
                for (const d of data.domains) {
                    const name = (d.name || "")
                        .replace(/^\//, "")
                        .replace(/\/$/, "");
                    if (name && d.class === "folder") domains.push(name);
                }
            }
            hsdsSystems = domains;
        } catch {
            // HSDS not reachable
        }
    }

    function create() {
        if (!selected || !isReady) return;
        loading = true;

        let code: string;
        if (selected.startsWith("hsds:")) {
            const name = selected.slice(5);
            code = [
                `import importlib, livn.system`,
                `importlib.reload(livn.system)`,
                `from livn.system import System`,
                `from livn.env import Env`,
                `env = Env(System('${name}'))`,
            ].join("\n");
        } else {
            const name = selected.startsWith("predefined:")
                ? selected.slice(11)
                : selected;
            code = `from livn.env import Env\nfrom livn.system import predefined\nenv = Env(predefined('${name}'))`;
        }

        pendingCommand.set(code);
        loading = false;
    }
</script>

<div class="system-loader">
    <span class="loader-label">System</span>
    <select bind:value={selected} disabled={!isReady}>
        <optgroup label="Predefined (HuggingFace)">
            {#each PREDEFINED as name}
                <option value="predefined:{name}">{name}</option>
            {/each}
        </optgroup>
        {#if hsdsSystems.length > 0}
            <optgroup label="HSDS">
                {#each hsdsSystems as name}
                    <option value="hsds:{name}">{name}</option>
                {/each}
            </optgroup>
        {/if}
    </select>
    <button onclick={create} disabled={!isReady || loading}>
        {loading ? "Loading…" : "Create"}
    </button>
</div>

<style>
    .system-loader {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        background: #0d0d1a;
        border-bottom: 1px solid #333;
        font-size: 12px;
    }
    .loader-label {
        color: #888;
        font-weight: 600;
        white-space: nowrap;
        user-select: none;
    }
    select {
        background: #1a1a2e;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 3px 6px;
        font-size: 12px;
        flex: 1;
        min-width: 0;
    }
    select:disabled {
        opacity: 0.5;
    }
    button {
        background: #4fc3f7;
        color: #0d0d1a;
        border: none;
        border-radius: 4px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
    }
    button:hover:not(:disabled) {
        background: #81d4fa;
    }
    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
</style>
