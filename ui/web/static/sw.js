/// <reference lib="webworker" />
const sw = self as unknown as ServiceWorkerGlobalScope;

const CACHE_NAME = 'pyodide-v0.29.3';
const PYODIDE_ORIGIN = '/pyodide/';

sw.addEventListener('install', () => {
    sw.skipWaiting();
});

sw.addEventListener('activate', (event) => {
    // Purge old caches
    event.waitUntil(
        caches.keys().then((names) =>
            Promise.all(
                names
                    .filter((n) => n.startsWith('pyodide-') && n !== CACHE_NAME)
                    .map((n) => caches.delete(n))
            )
        ).then(() => sw.clients.claim())
    );
});

sw.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Only cache pyodide assets (not the livn wheel, which changes during dev)
    if (!url.pathname.startsWith(PYODIDE_ORIGIN)) {
        return;
    }

    event.respondWith(
        caches.open(CACHE_NAME).then(async (cache) => {
            const cached = await cache.match(event.request);
            if (cached) return cached;

            const response = await fetch(event.request);
            if (response.ok) {
                cache.put(event.request, response.clone());
            }
            return response;
        })
    );
});
