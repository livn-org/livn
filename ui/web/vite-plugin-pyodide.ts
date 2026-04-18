import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { readFileSync, copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { get as httpsGet } from 'node:https';
import type { Plugin, Connect } from 'vite';

const LOCAL_ASSETS = ['pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'];

const CDN_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';

const MIME: Record<string, string> = {
    '.js': 'application/javascript',
    '.wasm': 'application/wasm',
    '.zip': 'application/zip',
    '.json': 'application/json',
    '.whl': 'application/zip'
};

function setHeaders(res: Connect.ServerResponse) {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
}

export function copyPyodidePlugin(): Plugin {
    const require = createRequire(import.meta.url);
    const pyodideDir = dirname(require.resolve('pyodide/package.json'));

    return {
        name: 'copy-pyodide-assets',

        // Dev: serve core assets locally, proxy package wheels to CDN
        configureServer(server) {
            server.middlewares.use('/pyodide', (req, res, next) => {
                const file = (req.url ?? '/').replace(/^\//, '').split('?')[0];
                if (!file) { next(); return; }

                const ext = '.' + file.split('.').pop()!;
                const contentType = MIME[ext] ?? 'application/octet-stream';

                // Serve any file that exists in node_modules/pyodide/
                const filePath = join(pyodideDir, file);
                if (existsSync(filePath)) {
                    res.setHeader('Content-Type', contentType);
                    setHeaders(res);
                    res.end(readFileSync(filePath));
                    return;
                }

                // Proxy to CDN using Node https (non-async, callback-based)
                const cdnUrl = CDN_BASE + file;
                const followRedirects = (url: string, depth = 0) => {
                    if (depth > 5) { res.statusCode = 502; res.end('Too many redirects'); return; }
                    httpsGet(url, (upstream) => {
                        if ((upstream.statusCode === 301 || upstream.statusCode === 302) && upstream.headers.location) {
                            followRedirects(upstream.headers.location, depth + 1);
                            return;
                        }
                        if (upstream.statusCode !== 200) {
                            res.statusCode = upstream.statusCode ?? 404;
                            res.end();
                            return;
                        }
                        res.setHeader('Content-Type', contentType);
                        setHeaders(res);
                        upstream.pipe(res);
                    }).on('error', (err) => {
                        res.statusCode = 502;
                        res.end('CDN proxy error');
                    });
                };
                followRedirects(cdnUrl);
            });
        },

        // Build: copy assets into the output directory
        writeBundle(options) {
            const outDir = options.dir ?? 'build';
            const dest = join(outDir, 'pyodide');
            mkdirSync(dest, { recursive: true });
            for (const name of LOCAL_ASSETS) {
                copyFileSync(join(pyodideDir, name), join(dest, name));
            }
        }
    };
}
