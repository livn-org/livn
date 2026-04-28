<script lang="ts">
    import { onMount } from 'svelte';

    // ── Preset MEA: 512-electrode lab chip (512_long_mea_6x_V2) ───────────────
    // Positions are [x, y] in µm, extracted from the lab's YAML file.
    const PRESET_MEA_COORDS: [number, number][] = [
        [0,11000],[600,11000],[0,10800],[600,10800],[0,10600],[600,10600],
        [0,10400],[600,10400],[0,10200],[600,10200],[0,10000],[600,10000],
        [0,9800],[600,9800],[0,9600],[600,9600],[600,12600],[200,12600],
        [200,12400],[400,12400],[200,12200],[400,12200],[200,12000],[400,12000],
        [200,11800],[400,11800],[200,11600],[400,11600],[200,11400],[400,11400],
        [200,11200],[400,11200],[200,11000],[400,11000],[200,10800],[400,10800],
        [200,10600],[400,10600],[200,10400],[400,10400],[200,10200],[400,10200],
        [200,10000],[400,10000],[200,9800],[400,9800],[200,9600],[400,9600],
        [400,12600],[0,12600],[0,12400],[600,12400],[0,12200],[600,12200],
        [0,12000],[600,12000],[0,11800],[600,11800],[0,11600],[600,11600],
        [0,11400],[600,11400],[0,11200],[600,11200],[1000,11000],[1200,11400],
        [1000,11400],[1200,11200],[1000,11600],[1200,11600],[1000,11800],[1200,11800],
        [1000,12000],[1200,12000],[1000,12200],[1200,12200],[1000,12400],[1200,12400],
        [1200,12600],[800,12600],[1000,6400],[1200,6400],[1000,6600],[1200,6600],
        [1000,6800],[1200,6800],[1000,7000],[1200,7000],[1000,7200],[1200,7200],
        [1000,7400],[1200,7400],[1000,7600],[1200,7600],[1000,7800],[1200,7800],
        [1000,8000],[1200,8000],[1000,8200],[1200,8200],[1000,8400],[1200,8400],
        [1000,8600],[1200,8600],[1000,8800],[1200,8800],[1000,9000],[1200,9000],
        [1000,9200],[1200,9200],[1000,9400],[1200,9400],[1000,9600],[1200,9600],
        [1000,10000],[1200,9800],[1000,9800],[1200,10200],[1000,10200],[1200,10000],
        [1000,10600],[1200,10400],[1000,10400],[1200,10800],[1000,10800],[1200,10600],
        [1000,11200],[1200,11000],[800,1400],[1400,1600],[800,1600],[1400,1800],
        [800,1800],[1400,2000],[800,2000],[1400,2200],[800,2200],[1400,2400],
        [800,2400],[1400,2600],[800,2600],[1400,2800],[1000,3000],[1200,3000],
        [1200,3400],[1000,3200],[1200,3200],[1400,3600],[800,3400],[1400,3400],
        [800,3800],[1400,3800],[800,3600],[1400,4200],[800,4000],[1400,4000],
        [800,4400],[1400,4400],[800,4200],[1400,4800],[800,4600],[1400,4600],
        [800,5000],[1400,5000],[800,4800],[1400,5400],[800,5200],[1400,5200],
        [800,5600],[1400,5600],[800,5400],[1400,6000],[800,5800],[1400,5800],
        [1000,6200],[1200,6200],[1200,200],[1200,0],[800,0],[1400,400],
        [800,200],[1400,200],[800,600],[1400,600],[800,400],[1400,1000],
        [800,800],[1400,800],[800,1000],[1400,1200],[800,1200],[1400,1400],
        [200,1600],[400,1600],[200,1200],[400,1200],[200,1400],[400,800],
        [200,1000],[400,1000],[200,600],[400,600],[200,800],[400,200],
        [200,400],[400,400],[400,0],[0,0],[200,3000],[400,3000],
        [0,2600],[600,2600],[0,2800],[600,2200],[0,2400],[600,2400],
        [0,2000],[600,2000],[0,2200],[600,1600],[0,1800],[600,1800],
        [0,1400],[600,1400],[0,1600],[600,1000],[0,1200],[600,1200],
        [0,800],[600,800],[0,1000],[600,400],[0,600],[600,600],
        [0,200],[600,200],[0,400],[600,0],[200,0],[200,200],
        [600,2800],[0,3000],[600,3000],[400,2600],[200,2800],[400,2800],
        [200,2400],[400,2400],[200,2600],[400,2000],[200,2200],[400,2200],
        [200,1800],[400,1800],[200,2000],[400,1400],[1400,11000],[800,11000],
        [1400,10800],[800,10800],[1400,10600],[800,10600],[1400,10400],[800,10400],
        [1400,10200],[800,10200],[1400,10000],[800,10000],[1400,9800],[800,9800],
        [1400,9600],[800,9600],[800,7000],[1400,7000],[800,7200],[1400,7200],
        [800,7400],[1400,7400],[800,7600],[1400,7600],[800,7800],[1400,7800],
        [800,8000],[1400,8000],[800,8200],[1400,8200],[800,8400],[1400,8400],
        [800,8600],[1400,8600],[800,8800],[1400,8800],[800,9000],[1400,9000],
        [800,9200],[1400,9200],[800,9400],[1400,9400],[800,6400],[1400,6400],
        [800,6600],[1400,6600],[800,6800],[1400,6800],[1000,12600],[1400,12600],
        [1400,12400],[800,12400],[1400,12200],[800,12200],[1400,12000],[800,12000],
        [1400,11800],[800,11800],[1400,11600],[800,11600],[1400,11400],[800,11400],
        [1400,11200],[800,11200],[1200,1600],[1000,1600],[1200,1200],[1000,1200],
        [1200,1400],[1000,800],[1200,1000],[1000,1000],[1200,600],[1000,600],
        [1200,800],[1000,200],[1200,400],[1000,400],[1000,0],[1400,0],
        [800,6000],[1400,6200],[800,6200],[1000,5800],[1200,6000],[1000,6000],
        [1200,5600],[1000,5600],[1200,5800],[1000,5200],[1200,5400],[1000,5400],
        [1200,5000],[1000,5000],[1200,5200],[1000,4600],[1200,4800],[1000,4800],
        [1200,4400],[1000,4400],[1200,4600],[1000,4000],[1200,4200],[1000,4200],
        [1200,3800],[1000,3800],[1200,4000],[1000,3400],[1200,3600],[1000,3600],
        [1400,3200],[800,3200],[800,2800],[1400,3000],[800,3000],[1000,2600],
        [1200,2800],[1000,2800],[1200,2400],[1000,2400],[1200,2600],[1000,2000],
        [1200,2200],[1000,2200],[1200,1800],[1000,1800],[1200,2000],[1000,1400],
        [400,8800],[600,8800],[200,8800],[0,8800],[400,9000],[600,9000],
        [200,9000],[0,9000],[400,9200],[600,9200],[200,9200],[0,9200],
        [400,9400],[600,9400],[200,9400],[0,9400],[400,6400],[600,6400],
        [200,6400],[0,6400],[400,6600],[600,6600],[200,6600],[0,6600],
        [400,6800],[600,6800],[200,6800],[0,6800],[400,7000],[600,7000],
        [200,7000],[0,7000],[400,7200],[600,7200],[200,7200],[0,7200],
        [400,7400],[600,7400],[200,7400],[0,7400],[400,7600],[600,7600],
        [200,7600],[0,7600],[400,7800],[600,7800],[200,7800],[0,7800],
        [400,8000],[600,8000],[200,8000],[0,8000],[400,8200],[600,8200],
        [200,8200],[0,8200],[400,8400],[600,8400],[200,8400],[0,8400],
        [400,8600],[600,8600],[200,8600],[0,8600],[600,3600],[400,4200],
        [0,4200],[200,4200],[600,4000],[400,4600],[0,4000],[200,4600],
        [600,4400],[400,4400],[0,4400],[200,4400],[600,4200],[400,4800],
        [0,4800],[200,4800],[600,4600],[400,4000],[0,4600],[200,5200],
        [600,5000],[400,5000],[0,5000],[200,5000],[600,4800],[400,5400],
        [0,5400],[200,5400],[600,5200],[400,5200],[0,5200],[200,5800],
        [600,5600],[400,5600],[0,5600],[200,5600],[600,5400],[400,6000],
        [0,6000],[200,6000],[600,5800],[400,5800],[0,5800],[600,6200],
        [400,6200],[0,6200],[200,6200],[600,6000],[200,3400],[600,3200],
        [400,3200],[0,3200],[200,3200],[400,3600],[0,3600],[200,3600],
        [600,3400],[400,3400],[0,3400],[200,4000],[600,3800],[400,3800],
        [0,3800],[200,3800],
    ];

    // ── Form state ──────────────────────────────────────────────────────────
    let name         = $state('my_culture');
    let shape        = $state<'rectangle' | 'disk'>('rectangle');
    let rectX        = $state(4000);
    let rectY        = $state(4000);
    let diskRadius   = $state(2000);
    let totalNeurons = $state(10000);
    let excRatio     = $state(0.8);
    let meaPitch     = $state(200);
    let meaMode      = $state<'custom' | 'preset'>('custom');

    // ── Derived counts ──────────────────────────────────────────────────────
    const excCount = $derived(Math.round(totalNeurons * excRatio));
    const inhCount = $derived(totalNeurons - excCount);

    function computeMEA(
        sh: string,
        rx: number, ry: number, dr: number,
        pitch: number
    ): [number, number][] {
        if (pitch <= 0) return [];
        const coords: [number, number][] = [];
        const xmin = sh === 'rectangle' ? 0 : -dr;
        const xmax = sh === 'rectangle' ? rx : dr;
        const ymin = sh === 'rectangle' ? 0 : -dr;
        const ymax = sh === 'rectangle' ? ry : dr;
        const sx = Math.ceil(xmin / pitch) * pitch;
        const sy = Math.ceil(ymin / pitch) * pitch;
        for (let x = sx; x <= xmax + 1e-6; x += pitch) {
            for (let y = sy; y <= ymax + 1e-6; y += pitch) {
                if (sh === 'disk' && Math.hypot(x, y) > dr) continue;
                coords.push([x, y]);
            }
        }
        return coords;
    }

    const meaCoords = $derived(
        meaMode === 'preset'
            ? PRESET_MEA_COORDS
            : computeMEA(shape, rectX, rectY, diskRadius, meaPitch)
    );
    const meaCount  = $derived(meaCoords.length);

    // ── Canvas preview ──────────────────────────────────────────────────────
    let canvas:    HTMLCanvasElement | null = $state(null);
    let container: HTMLDivElement    | null = $state(null);

    const PREVIEW_SAMPLE = 5000;
    // How many neurons are shown in preview (for legend)
    const excSampled = $derived(Math.min(excCount, Math.round(PREVIEW_SAMPLE * excRatio)));
    const inhSampled = $derived(Math.min(inhCount, PREVIEW_SAMPLE - excSampled));

    function makeRng(seed: number) {
        let s = (seed >>> 0) || 1;
        return () => {
            s = (Math.imul(s, 1664525) + 1013904223) | 0;
            return (s >>> 0) / 4294967296;
        };
    }

    // Box-Muller: returns one Gaussian sample (mean=0, std=1)
    function gaussian(rand: () => number): number {
        return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(rand() * 2 * Math.PI);
    }

    function drawPreview() {
        const cv = canvas;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        if (!ctx) return;

        const W = cv.width;
        const H = cv.height;
        if (W === 0 || H === 0) return;

        // Background
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#080812';
        ctx.fillRect(0, 0, W, H);

        // ── Coordinate transform ──────────────────────────────────────────
        let xmin: number, xmax: number, ymin: number, ymax: number;
        if (shape === 'rectangle') {
            const px = rectX * 0.07, py = rectY * 0.07;
            xmin = -px; xmax = rectX + px;
            ymin = -py; ymax = rectY + py;
        } else {
            const r = diskRadius * 1.13;
            xmin = -r; xmax = r; ymin = -r; ymax = r;
        }

        // MEA centering offset — always center MEA on culture center
        // Preset MEA center is (700, 6300); custom grid is already culture-aligned
        const cultureCx = shape === 'rectangle' ? rectX / 2 : 0;
        const cultureCy = shape === 'rectangle' ? rectY / 2 : 0;
        const meaOx = meaMode === 'preset' ? cultureCx - 700 : 0;
        const meaOy = meaMode === 'preset' ? cultureCy - 6300 : 0;

        // Asymmetric padding: extra space on the right for the legend column
        const padL = 55, padR = 210, padT = 20, padB = 50;
        const avW = W - padL - padR;
        const avH = H - padT - padB;
        const scale = Math.min(avW / (xmax - xmin), avH / (ymax - ymin));
        const drawW = (xmax - xmin) * scale;
        const drawH = (ymax - ymin) * scale;
        const ox = padL + (avW - drawW) / 2;
        const oy = padT + (avH - drawH) / 2;

        const tx = (x: number) => ox + (x - xmin) * scale;
        const ty = (y: number) => oy + drawH - (y - ymin) * scale; // flip Y so +y is up

        // ── Bounding shape ────────────────────────────────────────────────
        ctx.strokeStyle = '#2a3055';
        ctx.lineWidth = 1.5;
        if (shape === 'rectangle') {
            ctx.strokeRect(tx(0), ty(rectY), rectX * scale, rectY * scale);
        } else {
            ctx.beginPath();
            ctx.arc(tx(0), ty(0), diskRadius * scale, 0, Math.PI * 2);
            ctx.stroke();
        }

        // ── Sample neuron positions (store arrays so we can draw connections first) ──
        const rand = makeRng(42);

        function samplePt(): [number, number] {
            if (shape === 'rectangle') {
                return [rand() * rectX, rand() * rectY];
            }
            const u = rand(), theta = rand() * 2 * Math.PI;
            const r = Math.sqrt(u) * diskRadius;
            return [r * Math.cos(theta), r * Math.sin(theta)];
        }

        const excPts: [number, number][] = [];
        const inhPts: [number, number][] = [];
        for (let i = 0; i < excSampled; i++) excPts.push(samplePt());
        for (let i = 0; i < inhSampled; i++) inhPts.push(samplePt());

        // ── Approximate connection lines (Gaussian local connectivity) ────
        // Mirrors generate_2d default: sigma=200µm, up to ~2500 lines
        // We sample target offsets from N(0, sigma) directly in µm-space
        // (equivalent to sampling from the distance-dependent probability)
        const sigmaPx = 200 * scale; // 200 µm default in pixel space
        const CONN_SOURCES = Math.min(excPts.length, 350);
        const CONN_PER    = Math.ceil(2000 / Math.max(1, CONN_SOURCES));

        ctx.strokeStyle = 'rgba(79, 195, 247, 0.07)'; // same hue as EXC, very faint
        ctx.lineWidth = 0.35;
        ctx.beginPath();
        for (let i = 0; i < CONN_SOURCES; i++) {
            const [sx, sy] = excPts[i];
            const spx = tx(sx), spy = ty(sy);
            for (let j = 0; j < CONN_PER; j++) {
                // Gaussian offset in px (Box-Muller)
                const dx = gaussian(rand) * sigmaPx;
                const dy = gaussian(rand) * sigmaPx;
                ctx.moveTo(spx, spy);
                ctx.lineTo(spx + dx, spy + dy); // dy keeps same sign (both flipped)
            }
        }
        ctx.stroke();

        // ── MEA electrodes: dot + output-radius ring + ID label ──────────
        if (meaCoords.length > 0 && meaCoords.length <= 4096) {
            const outputRadiusPx = 50 * scale;

            // Recording radius rings
            ctx.strokeStyle = 'rgba(253, 216, 53, 0.25)';
            ctx.lineWidth = 0.4;
            ctx.beginPath();
            for (const [ex, ey] of meaCoords) {
                const cx = tx(ex + meaOx), cy = ty(ey + meaOy);
                ctx.moveTo(cx + outputRadiusPx, cy);
                ctx.arc(cx, cy, outputRadiusPx, 0, Math.PI * 2);
            }
            ctx.stroke();

            // Electrode dots
            ctx.fillStyle = 'rgba(253, 216, 53, 0.85)';
            ctx.strokeStyle = 'rgba(0,0,0,0.6)';
            ctx.lineWidth = 0.4;
            ctx.beginPath();
            for (const [ex, ey] of meaCoords) {
                const cx = tx(ex + meaOx), cy = ty(ey + meaOy);
                ctx.moveTo(cx + 2, cy);
                ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            }
            ctx.fill();
            ctx.stroke();

            // Electrode ID numbers (only if spaced enough to read, custom mode only)
            if (meaMode === 'custom' && meaPitch * scale > 22 && meaCoords.length <= 256) {
                ctx.fillStyle = 'rgba(150, 180, 220, 0.7)';
                ctx.font = `${Math.max(6, Math.min(9, meaPitch * scale * 0.18))}px monospace`;
                ctx.textAlign = 'left';
                meaCoords.forEach(([ex, ey], idx) => {
                    ctx.fillText(String(idx), tx(ex + meaOx) + 3, ty(ey + meaOy) - 3);
                });
            }
        }

        // ── Neuron circles (batched arc — matches system.png scatter s=5, alpha=0.7) ──
        const dotR = Math.max(1, Math.min(2.5, drawW / 280));

        // EXC
        ctx.fillStyle = 'rgba(79, 195, 247, 0.68)';
        ctx.beginPath();
        for (const [x, y] of excPts) {
            const cx = tx(x), cy = ty(y);
            ctx.moveTo(cx + dotR, cy);
            ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
        }
        ctx.fill();

        // INH
        ctx.fillStyle = 'rgba(239, 83, 80, 0.75)';
        ctx.beginPath();
        for (const [x, y] of inhPts) {
            const cx = tx(x), cy = ty(y);
            ctx.moveTo(cx + dotR, cy);
            ctx.arc(cx, cy, dotR, 0, Math.PI * 2);
        }
        ctx.fill();

        // ── Legend (right column — outside culture) ───────────────────────
        // Left edge sits 8px right of the culture (half of the 15px axis gap)
        const legX    = ox + drawW + 8;
        const legTopY = oy + 60;
        const legFont = 15;
        const legLineH = 26;
        const legDotR  = 8;

        const legendItems = [
            { color: 'rgba(79, 195, 247, 0.9)',  label: 'EXC',  sub: `${excCount.toLocaleString()} neurons` },
            { color: 'rgba(239,  83,  80, 0.9)',  label: 'INH',  sub: `${inhCount.toLocaleString()} neurons` },
            { color: 'rgba(253, 216, 53,  0.9)',  label: 'MEA',  sub: `${meaCount} electrodes` },
        ];

        ctx.font = `bold ${legFont}px sans-serif`;
        legendItems.forEach(({ color, label, sub }, i) => {
            const cy = legTopY + i * (legLineH * 2 + 8) + legDotR + 4;

            // Colour dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(legX + legDotR, cy, legDotR, 0, Math.PI * 2);
            ctx.fill();

            // Bold label
            ctx.fillStyle = '#ddd';
            ctx.textAlign = 'left';
            ctx.fillText(label, legX + legDotR * 2 + 6, cy + 4.5);

            // Smaller sub-label on the line below
            ctx.font = `${legFont - 1}px sans-serif`;
            ctx.fillStyle = '#777';
            ctx.fillText(sub, legX + legDotR * 2 + 6, cy + legLineH);
            ctx.font = `bold ${legFont}px sans-serif`;
        });

        // ── Scale bar (bottom of legend column) ──────────────────────────
        const spanUm = xmax - xmin;
        const targetUm = Math.pow(10, Math.floor(Math.log10(spanUm / 4)));
        const niceUm = [1, 2, 5, 10].map(m => m * targetUm).find(v => v * scale > 50) ?? targetUm * 10;
        const barPx = niceUm * scale;
        if (barPx > 30 && barPx < (padR - 20)) {
            const bx1 = legX, bx2 = legX + barPx;
            const by = oy + drawH - 10;
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(bx1, by); ctx.lineTo(bx2, by);
            ctx.moveTo(bx1, by - 3); ctx.lineTo(bx1, by + 3);
            ctx.moveTo(bx2, by - 3); ctx.lineTo(bx2, by + 3);
            ctx.stroke();
            ctx.fillStyle = '#777';
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            const barLabel = niceUm >= 1000 ? `${niceUm / 1000} mm` : `${niceUm} µm`;
            ctx.fillText(barLabel, bx1, by - 6);
        }

        // ── Axis labels ───────────────────────────────────────────────────
        ctx.fillStyle = '#888';
        ctx.font = '15px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('X coordinate (µm)', ox + drawW / 2, oy + drawH + 15);
        ctx.save();
        ctx.translate(ox - 19, oy + drawH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Y coordinate (µm)', 0, 0);
        ctx.restore();
    }

    // When preset MEA is selected, auto-size culture to match the MEA footprint
    // (1400 × 12600 µm) so the MEA is never larger than the culture by default
    $effect(() => {
        if (meaMode === 'preset') {
            shape  = 'rectangle';
            rectX  = 1400;
            rectY  = 12600;
        }
    });

    // Redraw whenever any reactive dep changes
    $effect(() => {
        void shape; void rectX; void rectY; void diskRadius;
        void totalNeurons; void excRatio; void meaPitch; void meaMode;
        void meaCoords; void canvas;
        drawPreview();
    });

    onMount(() => {
        const cont = container;
        if (!cont) return;
        const observer = new ResizeObserver(() => {
            const cv = canvas;
            if (cv && cont) {
                cv.width  = cont.clientWidth;
                cv.height = cont.clientHeight;
                drawPreview();
            }
        });
        observer.observe(cont);
        return () => observer.disconnect();
    });

    // ── CLI command ─────────────────────────────────────────────────────────
    const cliCommand = $derived((() => {
        const n  = name || 'my_culture';
        const sh = shape;
        const rx = rectX, ry = rectY, dr = diskRadius;
        const tn = totalNeurons, mp = meaPitch;
        const lines = [
            `python systems/generate_cultures.py`,
            `# ── inputs ─────────────────────────`,
            `# Folder name  : ${n}`,
            `# Shape        : ${sh === 'rectangle' ? '1  (rectangle)' : '2  (disk)'}`,
        ];
        if (sh === 'rectangle') {
            lines.push(`# X dimension  : ${rx} µm`);
            lines.push(`# Y dimension  : ${ry} µm`);
        } else {
            lines.push(`# Radius       : ${dr} µm`);
        }
        lines.push(`# Total neurons: ${tn}`);
        lines.push(`# MEA pitch    : ${mp} µm`);
        return lines.join('\n');
    })());

    let copied = $state(false);
    function copyCommand() {
        navigator.clipboard.writeText(cliCommand).then(() => {
            copied = true;
            setTimeout(() => (copied = false), 1800);
        });
    }
</script>

<div class="generator">

    <!-- ── Left: form ─────────────────────────────────────────────────── -->
    <div class="form-panel">
        <h2>System Generation</h2>

        <div class="field">
            <label class="field-label" for="sg-name">Name</label>
            <input id="sg-name" class="text-input" type="text" bind:value={name} placeholder="my_culture" />
        </div>

        <div class="field">
            <div class="field-label">Shape</div>
            <div class="radio-group">
                <label class="radio">
                    <input type="radio" bind:group={shape} value="rectangle" />
                    Rectangle
                </label>
                <label class="radio">
                    <input type="radio" bind:group={shape} value="disk" />
                    Disk
                </label>
            </div>
        </div>

        {#if shape === 'rectangle'}
            <div class="field-row">
                <div class="field">
                    <label class="field-label" for="sg-rect-x">Width (µm)</label>
                    <input id="sg-rect-x" class="num-input" type="number" bind:value={rectX}
                        min="100" max="20000" step="100" />
                </div>
                <div class="field">
                    <label class="field-label" for="sg-rect-y">Height (µm)</label>
                    <input id="sg-rect-y" class="num-input" type="number" bind:value={rectY}
                        min="100" max="20000" step="100" />
                </div>
            </div>
        {:else}
            <div class="field">
                <label class="field-label" for="sg-radius">Radius (µm)</label>
                <input id="sg-radius" class="num-input" type="number" bind:value={diskRadius}
                    min="100" max="10000" step="100" />
            </div>
        {/if}

        <div class="field">
            <label class="field-label" for="sg-neurons">
                Total neurons
                <span class="field-val">{totalNeurons.toLocaleString()}</span>
            </label>
            <input id="sg-neurons" type="range" bind:value={totalNeurons}
                min="100" max="50000" step="100" />
        </div>

        <div class="field">
            <label class="field-label" for="sg-exc-ratio">
                EXC / INH split
                <span class="field-val">
                    {(excRatio * 100).toFixed(0)}% / {((1 - excRatio) * 100).toFixed(0)}%
                </span>
            </label>
            <input id="sg-exc-ratio" type="range" bind:value={excRatio}
                min="0.5" max="0.95" step="0.05" />
        </div>

        <div class="field">
            <div class="field-label">Electrode array (MEA)</div>
            <div class="radio-group">
                <label class="radio">
                    <input type="radio" bind:group={meaMode} value="custom" />
                    Custom grid
                </label>
                <label class="radio">
                    <input type="radio" bind:group={meaMode} value="preset" />
                    Lab preset (512 electrodes)
                </label>
            </div>
        </div>

        {#if meaMode === 'custom'}
            <div class="field">
                <label class="field-label" for="sg-pitch">Electrode spacing (MEA pitch) (µm)</label>
                <input id="sg-pitch" class="num-input" type="number" bind:value={meaPitch}
                    min="50" max="1000" step="50" />
            </div>
        {:else}
            <div class="preset-info">
                512_long_mea_6x_V2 &nbsp;·&nbsp; 8 × 64 grid &nbsp;·&nbsp; 200 µm pitch
            </div>
        {/if}

        <!-- Stats -->
        <div class="stats">
            <div class="stat-row">
                <span class="stat-label">EXC neurons</span>
                <span class="stat-exc">{excCount.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">INH neurons</span>
                <span class="stat-inh">{inhCount.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">MEA electrodes</span>
                <span class="stat-mea">{meaCount}</span>
            </div>
        </div>

        <!-- Generated command -->
        <div class="cmd-section">
            <div class="cmd-header">
                <span class="cmd-title">Run locally</span>
                <button class="copy-btn" onclick={copyCommand}>
                    {copied ? '✓ Copied' : 'Copy'}
                </button>
            </div>
            <pre class="cmd-pre">{cliCommand}</pre>
        </div>
    </div>

    <!-- ── Right: live preview ─────────────────────────────────────────── -->
    <div class="preview-panel" bind:this={container}>
        <canvas bind:this={canvas}></canvas>
    </div>

</div>

<style>
    .generator {
        display: grid;
        grid-template-columns: 320px 1fr;
        height: 100%;
        overflow: hidden;
        background: #0d0d1a;
    }

    /* ── Form panel ──────────────────────────────────────────────────── */
    .form-panel {
        border-right: 1px solid #252540;
        overflow-y: auto;
        padding: 20px 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }

    h2 {
        font-size: 15px;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 4px;
    }

    .field {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .field-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .field-label {
        font-size: 11px;
        font-weight: 600;
        color: #888;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .field-val {
        color: #4fc3f7;
        font-weight: 700;
    }

    .text-input,
    .num-input {
        background: #111126;
        border: 1px solid #2a2a4a;
        border-radius: 4px;
        color: #e0e0e0;
        font-size: 13px;
        font-family: inherit;
        padding: 5px 8px;
        outline: none;
        width: 100%;
    }
    .text-input:focus,
    .num-input:focus {
        border-color: #4fc3f7;
    }

    input[type="range"] {
        width: 100%;
        accent-color: #4fc3f7;
    }

    .radio-group {
        display: flex;
        gap: 16px;
    }

    .radio {
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 13px;
        color: #ccc;
        cursor: pointer;
    }

    /* ── Stats ───────────────────────────────────────────────────────── */
    .stats {
        background: #111126;
        border: 1px solid #222244;
        border-radius: 6px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
    }

    .stat-label { color: #666; }
    .stat-exc   { color: #4fc3f7; font-weight: 700; }
    .stat-inh   { color: #ef5350; font-weight: 700; }
    .stat-mea   { color: #fdd835; font-weight: 700; }

    /* ── CLI command ─────────────────────────────────────────────────── */
    .cmd-section {
        background: #0a0a18;
        border: 1px solid #1e1e38;
        border-radius: 6px;
        overflow: hidden;
    }

    .cmd-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 10px;
        border-bottom: 1px solid #1e1e38;
    }

    .cmd-title {
        font-size: 11px;
        font-weight: 600;
        color: #666;
    }

    .copy-btn {
        background: none;
        border: 1px solid #333;
        border-radius: 3px;
        color: #888;
        font-size: 10px;
        padding: 2px 8px;
        cursor: pointer;
        font-family: inherit;
        transition: color 0.15s, border-color 0.15s;
    }
    .copy-btn:hover { color: #ccc; border-color: #555; }

    .cmd-pre {
        font-family: 'Courier New', Courier, monospace;
        font-size: 10px;
        color: #7ecfef;
        padding: 10px 12px;
        white-space: pre;
        overflow-x: auto;
        line-height: 1.55;
    }

    .preset-info {
        font-size: 11px;
        color: #fdd835;
        background: rgba(253, 216, 53, 0.07);
        border: 1px solid rgba(253, 216, 53, 0.2);
        border-radius: 4px;
        padding: 6px 10px;
    }

    /* ── Preview panel ───────────────────────────────────────────────── */
    .preview-panel {
        position: relative;
        min-width: 0;
        min-height: 0;
        background: #080812;
    }

    .preview-panel canvas {
        display: block;
        width: 100%;
        height: 100%;
    }

</style>
