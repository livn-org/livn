import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type Plugin } from 'vite';
import { copyPyodidePlugin } from './vite-plugin-pyodide';
import { readFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

// Serve graph.json (and other metadata) from systems/graphs/ for Pyodide
const graphsDir = resolve(__dirname, '../../systems/graphs');

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
    plugins: [serveGraphFiles(), copyPyodidePlugin(), sveltekit()],
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
