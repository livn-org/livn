<script lang="ts">
  let rows = 8;
  let cols = 8;
  let spacing = 100;
  let electrodeSize = 20;

  type Electrode = {
    id: number;
    x: number;
    y: number;
    z: number;
    size: number;
  };

  $: electrodes = generateGrid(rows, cols, spacing, electrodeSize);

  function generateGrid(rows: number, cols: number, spacing: number, size: number): Electrode[] {
    const out: Electrode[] = [];
    const x0 = -((cols - 1) * spacing) / 2;
    const y0 = -((rows - 1) * spacing) / 2;

    let id = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out.push({
          id,
          x: x0 + c * spacing,
          y: y0 + r * spacing,
          z: 0,
          size
        });
        id++;
      }
    }
    return out;
  }

  $: meaJson = {
    type: "MEA",
    rows,
    cols,
    spacing,
    electrode_size: electrodeSize,
    electrode_coordinates: electrodes.map((e) => [e.id, e.x, e.y, e.z])
  };

  function downloadJson() {
    const blob = new Blob([JSON.stringify(meaJson, null, 2)], {
      type: "application/json"
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mea_config.json";
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="mea-card">
  <h2>MEA Generation</h2>
  <p>Create a grid MEA layout and export the electrode coordinates as JSON.</p>

  <div class="controls">
    <label>
      Rows
      <input type="number" min="1" bind:value={rows} />
    </label>

    <label>
      Columns
      <input type="number" min="1" bind:value={cols} />
    </label>

    <label>
      Spacing
      <input type="number" min="1" bind:value={spacing} />
    </label>

    <label>
      Electrode size
      <input type="number" min="1" bind:value={electrodeSize} />
    </label>
  </div>

  <div class="preview">
    <svg viewBox="-500 -500 1000 1000">
      {#each electrodes as e}
        <circle cx={e.x} cy={e.y} r={electrodeSize / 2}>
          <title>Electrode {e.id}: ({e.x}, {e.y}, {e.z})</title>
        </circle>
      {/each}
    </svg>
  </div>

  <button on:click={downloadJson}>Download MEA JSON</button>

  <pre>{JSON.stringify(meaJson, null, 2)}</pre>
</div>

<style>
  .mea-card {
    padding: 1rem;
    border-radius: 12px;
    background: #181818;
    color: white;
    border: 1px solid #333;
  }

  .controls {
    display: grid;
    grid-template-columns: repeat(4, minmax(100px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.85rem;
  }

  input {
    padding: 0.4rem;
    border-radius: 6px;
    border: 1px solid #555;
    background: #111;
    color: white;
  }

  .preview {
    height: 360px;
    border: 1px solid #444;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  svg {
    width: 100%;
    height: 100%;
    background: #0f0f0f;
  }

  circle {
    fill: #fdd835;
    stroke: #fff8;
    stroke-width: 2;
  }

  button {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
  }

  pre {
    max-height: 260px;
    overflow: auto;
    background: #111;
    padding: 1rem;
    border-radius: 8px;
    font-size: 0.8rem;
  }
</style>