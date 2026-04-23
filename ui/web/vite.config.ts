import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import { copyPyodidePlugin } from './vite-plugin-pyodide';
import { readFileSync, existsSync, openSync, readSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';

// Serve graph.json (and other metadata) from systems/graphs/ for Pyodide
const graphsDir = resolve(__dirname, '../../systems/graphs');

// Serve binary signal chunks from bio_data recordings
const bioDataRoot = resolve(__dirname, '../../bio_data');

function serveBioData(): Plugin {
    return {
        name: 'serve-bio-data',
        configureServer(server) {
            server.middlewares.use((req: any, res: any, next: () => void) => {
                if (!req.url?.startsWith('/bio-api/chunk')) return next();

                const url      = new URL(req.url, 'http://localhost');
                const rec      = url.searchParams.get('rec')        ?? '';
                const offsetS  = parseFloat(url.searchParams.get('offset_s')  ?? '0');
                const durS     = parseFloat(url.searchParams.get('dur_s')      ?? '5');
                const dsF      = Math.max(1, parseInt(url.searchParams.get('downsample') ?? '100'));
                const chStart  = Math.max(0,   parseInt(url.searchParams.get('ch_start') ?? '0'));
                const chEnd    = Math.min(512, parseInt(url.searchParams.get('ch_end')   ?? '32'));

                if (chEnd <= chStart) { res.statusCode = 400; res.end('Bad ch range'); return; }

                // Synthetic demo recordings — generated on-the-fly, no file needed
                if (rec.startsWith('demo/')) {
                    const SR    = 30000;
                    const effSR = Math.round(SR / dsF);
                    const nOut  = Math.floor(durS * SR / dsF);
                    const chCount = chEnd - chStart;
                    const out   = new Float32Array(chCount * nOut);

                    // Deterministic 32-bit hash for reproducible "noise"
                    function h32(n: number): number {
                        n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
                        n = Math.imul(n ^ (n >>> 16), 0x45d9f3b);
                        return (n ^ (n >>> 16)) >>> 0;
                    }

                    for (let ci = 0; ci < chCount; ci++) {
                        const ch = chStart + ci;
                        for (let si = 0; si < nOut; si++) {
                            const sIdx = Math.round(offsetS * SR) + si * dsF;
                            const t    = sIdx / SR;

                            // LFP: theta (8 Hz) + gamma (40 Hz), phase-shifted per channel
                            const lfp   = 38 * Math.sin(2 * Math.PI * 8  * t + ch * 0.37)
                                        + 14 * Math.sin(2 * Math.PI * 40 * t + ch * 0.59);

                            // Deterministic Gaussian noise via Box-Muller
                            const seed  = (ch * 1800007 + sIdx) | 0;
                            const u1    = Math.max((h32(seed)     >>> 8) / 16777216, 1e-10);
                            const u2    =          (h32(seed + 1) >>> 8) / 16777216;
                            const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 13;

                            // TTL stim artifact at 1 Hz (t = 0.5, 1.5, 2.5, …)
                            const tMod  = (t + 0.5) % 1.0;
                            const stim  = tMod < 0.03
                                ? (280 + 180 * Math.sin(ch * 0.5)) * Math.exp(-tMod * 260)
                                : 0;

                            out[ci * nOut + si] = lfp + noise + stim;
                        }
                    }

                    res.setHeader('Content-Type', 'application/octet-stream');
                    res.setHeader('Access-Control-Expose-Headers',
                        'X-N-Channels,X-N-Samples,X-Sample-Rate,X-Ch-Start');
                    res.setHeader('X-N-Channels',  String(chCount));
                    res.setHeader('X-N-Samples',   String(nOut));
                    res.setHeader('X-Sample-Rate', String(effSR));
                    res.setHeader('X-Ch-Start',    String(chStart));
                    res.end(Buffer.from(out.buffer));
                    return;
                }

                // Security: path must stay under bio_data root
                const datPath = resolve(bioDataRoot, rec,
                    'continuous', 'Acquisition_Board-100.Rhythm Data', 'continuous.dat');
                if (!datPath.startsWith(bioDataRoot) || !existsSync(datPath)) {
                    res.statusCode = 404; res.end('Not found'); return;
                }

                const N_CH       = 512;
                const SR         = 30000;
                const BIT_VOLTS  = 0.195;
                const N_SAMPLES  = 9212672; // total samples in this recording
                const ROW_BYTES  = N_CH * 2;

                const offsetSample = Math.max(0, Math.floor(offsetS * SR));
                const nSamples     = Math.min(Math.floor(durS * SR), N_SAMPLES - offsetSample);
                const chCount      = chEnd - chStart;
                const nOut         = Math.floor(nSamples / dsF);

                // Read every dsF-th row (stride read — avoids loading the full window)
                const rowBuf = Buffer.allocUnsafe(ROW_BYTES);
                const out    = new Float32Array(chCount * nOut);

                let fd = -1;
                try {
                    fd = openSync(datPath, 'r');
                    for (let si = 0; si < nOut; si++) {
                        readSync(fd, rowBuf, 0, ROW_BYTES,
                            (offsetSample + si * dsF) * ROW_BYTES);
                        for (let ci = 0; ci < chCount; ci++) {
                            out[ci * nOut + si] =
                                rowBuf.readInt16LE((chStart + ci) * 2) * BIT_VOLTS;
                        }
                    }
                    closeSync(fd);
                } catch (e) {
                    if (fd >= 0) try { closeSync(fd); } catch {}
                    res.statusCode = 500; res.end(String(e)); return;
                }

                res.setHeader('Content-Type', 'application/octet-stream');
                res.setHeader('Access-Control-Expose-Headers',
                    'X-N-Channels,X-N-Samples,X-Sample-Rate,X-Ch-Start');
                res.setHeader('X-N-Channels',  String(chCount));
                res.setHeader('X-N-Samples',   String(nOut));
                res.setHeader('X-Sample-Rate', String(Math.round(SR / dsF)));
                res.setHeader('X-Ch-Start',    String(chStart));
                res.end(Buffer.from(out.buffer));
            });
        }
    };
}

function serveGraphFiles(): Plugin {
    return {
        name: 'serve-graph-files',
        configureServer(server) {
            server.middlewares.use((req, res, next) => {
                if (!req.url?.startsWith('/files/')) return next();
                // /files/EI1/graph.json -> systems/graphs/EI1/graph.json
                const relPath = req.url.slice('/files/'.length);
                const filePath = join(graphsDir, relPath);
                // Prevent path traversal
                if (!filePath.startsWith(graphsDir)) {
                    res.statusCode = 403;
                    res.end('Forbidden');
                    return;
                }
                if (!existsSync(filePath)) {
                    res.statusCode = 404;
                    res.setHeader('Content-Type', 'application/json');
                    res.end(JSON.stringify({ error: 'Not found' }));
                    return;
                }
                const data = readFileSync(filePath);
                res.setHeader('Content-Type', 'application/json');
                res.end(data);
            });
        }
    };
}

export default defineConfig({
    plugins: [serveGraphFiles(), serveBioData(), copyPyodidePlugin(), sveltekit()],
    server: {
        headers: {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp'
        },
        proxy: {
            '/hsds': {
                target: 'http://localhost:5101',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/hsds/, '')
            }
        }
    },
    optimizeDeps: {
        exclude: ['pyodide']
    }
});
