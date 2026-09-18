#include "thread.h"

#include <stdlib.h>

#include "internal.h"

/* --- the serial fallback ---------------------------------------------------- */

#if !RCSD_THREADS_AVAILABLE

int rcsd_set_num_threads(int n) {
    (void) n;
    return 1;
}

int rcsd_num_threads(void) {
    return 1;
}

void rcsd_parallel_for(int n, int grain, RCSDKernel fn, void* ctx) {
    (void) grain;
    if (n > 0) {
        fn(ctx, 0, n);
    }
}

void rcsd_threads_shutdown(void) {}

#else /* RCSD_THREADS_AVAILABLE */

#include <pthread.h>

/* Only gcc and clang reach this file -- the Windows and Emscripten builds take
 * the fallback above -- so the __atomic builtins are available, and are what
 * makes the spin loops below defined rather than a data race. */

/* A timestep posts six parallel loops, so at culture scale a run goes through
 * tens of thousands of them a second. Sleeping and waking on each one costs far
 * more than the loop itself, so a worker spins first and only then blocks. The
 * budget is about a few tens of microseconds: long enough to cover the gap
 * between one loop and the next inside a run, short enough that the pool is
 * asleep rather than burning a core once the run ends. */
#define RCSD_SPIN_BUDGET 20000

#if defined(__x86_64__) || defined(__i386__)
#define CPU_RELAX() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define CPU_RELAX() ((void) 0)
#endif

typedef struct Pool Pool;

typedef struct {
    Pool* pool;
    int id;
} Worker;

struct Pool {
    pthread_mutex_t mutex;
    pthread_cond_t work; /* master -> workers: a job is posted */
    pthread_cond_t done; /* workers -> master: the last slice finished */
    pthread_t* threads;
    Worker* workers;
    int n_threads; /* counting the master */
    int n_workers; /* spawned; n_threads - 1 */
    int started;

    /* the posted job. `generation` is the handshake: the master bumps it last,
     * with a release, and a worker that sees a new value sees the job with it */
    RCSDKernel fn;
    void* ctx;
    int n_items;
    unsigned long generation;
    int n_running; /* workers yet to finish this job */
    int shutdown;
};

static Pool g_pool = {PTHREAD_MUTEX_INITIALIZER,
                      PTHREAD_COND_INITIALIZER,
                      PTHREAD_COND_INITIALIZER,
                      NULL,
                      NULL,
                      1,
                      0,
                      0,
                      NULL,
                      NULL,
                      0,
                      0,
                      0,
                      0};

/* Set while this thread is inside a kernel, so a kernel that itself calls
 * rcsd_parallel_for runs its loop inline instead of waiting on workers that are
 * already busy with the outer one. */
static __thread int t_in_parallel = 0;

/* Contiguous split: thread `id` of `nt` takes [*begin, *end) of [0, n), with
 * the first n % nt threads taking one extra item. It depends only on its
 * arguments, never on timing, so a given thread count always splits the same
 * way -- which is what lets a threaded run reproduce a serial one exactly. */
static void slice(int n, int nt, int id, int* begin, int* end) {
    int base = n / nt;
    int rem = n % nt;
    int b = id * base + (id < rem ? id : rem);
    *begin = b;
    *end = b + base + (id < rem ? 1 : 0);
}

static void run_slice(const Pool* p, RCSDKernel fn, void* ctx, int n, int id) {
    int begin, end;
    slice(n, p->n_threads, id, &begin, &end);
    if (end > begin) {
        t_in_parallel = 1;
        fn(ctx, begin, end);
        t_in_parallel = 0;
    }
}

/* Wait for `generation` to move past `seen`, or for shutdown. Returns the
 * generation now current. */
static unsigned long await_job(Pool* p, unsigned long seen) {
    int spins;
    unsigned long gen;

    for (spins = 0; spins < RCSD_SPIN_BUDGET; ++spins) {
        gen = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE);
        if (gen != seen || __atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) {
            return gen;
        }
        CPU_RELAX();
    }

    /* The spin budget is gone, so block. The master posts under this same mutex,
     * so between the test and the wait there is no window for it to slip a job
     * past a worker that is about to sleep. */
    pthread_mutex_lock(&p->mutex);
    while (!p->shutdown && p->generation == seen) {
        pthread_cond_wait(&p->work, &p->mutex);
    }
    gen = p->generation;
    pthread_mutex_unlock(&p->mutex);
    return gen;
}

static void* worker_main(void* arg) {
    Worker* self = (Worker*) arg;
    Pool* p = self->pool;
    unsigned long seen = 0;

    for (;;) {
        unsigned long gen = await_job(p, seen);
        if (__atomic_load_n(&p->shutdown, __ATOMIC_ACQUIRE)) {
            return NULL;
        }
        seen = gen;

        run_slice(p, p->fn, p->ctx, p->n_items, self->id);

        if (__atomic_sub_fetch(&p->n_running, 1, __ATOMIC_ACQ_REL) == 0) {
            /* the master may have given up spinning and blocked */
            pthread_mutex_lock(&p->mutex);
            pthread_cond_signal(&p->done);
            pthread_mutex_unlock(&p->mutex);
        }
    }
}

static void stop_workers(Pool* p) {
    int i;
    if (!p->started) {
        return;
    }
    pthread_mutex_lock(&p->mutex);
    __atomic_store_n(&p->shutdown, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->mutex);

    for (i = 0; i < p->n_workers; ++i) {
        pthread_join(p->threads[i], NULL);
    }
    free(p->threads);
    free(p->workers);
    p->threads = NULL;
    p->workers = NULL;
    p->n_workers = 0;
    p->n_threads = 1;
    p->started = 0;
    p->shutdown = 0;
    p->generation = 0;
}

int rcsd_set_num_threads(int n) {
    Pool* p = &g_pool;
    int want, i;

    if (t_in_parallel) {
        rcsd_set_error("rcsd_set_num_threads cannot be called from inside a parallel loop");
        return -1;
    }

    want = n < 1 ? 1 : n;
    if (p->started && p->n_threads == want) {
        return want;
    }

    stop_workers(p);
    if (want == 1) {
        return 1;
    }

    p->threads = (pthread_t*) calloc((size_t) (want - 1), sizeof(pthread_t));
    p->workers = (Worker*) calloc((size_t) (want - 1), sizeof(Worker));
    if (!p->threads || !p->workers) {
        free(p->threads);
        free(p->workers);
        p->threads = NULL;
        p->workers = NULL;
        rcsd_set_error("out of memory starting the worker pool");
        return -1;
    }

    p->n_threads = want;
    p->n_workers = 0;
    p->shutdown = 0;
    p->generation = 0;
    p->n_running = 0;

    for (i = 0; i < want - 1; ++i) {
        p->workers[i].pool = p;
        p->workers[i].id = i + 1; /* the master is id 0 */
        if (pthread_create(&p->threads[i], NULL, worker_main, &p->workers[i]) != 0) {
            /* keep whatever did start; the pool is consistent at that size */
            p->n_workers = i;
            p->n_threads = i + 1;
            p->started = i > 0;
            if (i == 0) {
                free(p->threads);
                free(p->workers);
                p->threads = NULL;
                p->workers = NULL;
                p->n_threads = 1;
            }
            rcsd_set_error("could not start every worker thread");
            return -1;
        }
        p->n_workers = i + 1;
    }

    p->started = 1;
    return p->n_threads;
}

int rcsd_num_threads(void) {
    return g_pool.n_threads;
}

void rcsd_parallel_for(int n, int grain, RCSDKernel fn, void* ctx) {
    Pool* p = &g_pool;
    int spins;

    if (n <= 0) {
        return;
    }
    /* Serial pool, a loop too short to be worth splitting, or a nested call.
     * Nested is the one that matters: the workers are inside the outer loop, so
     * a job posted to them would never be picked up. */
    if (!p->started || p->n_threads <= 1 || n < grain || t_in_parallel) {
        t_in_parallel = 1;
        fn(ctx, 0, n);
        t_in_parallel = 0;
        return;
    }

    /* Posting holds the mutex so that a worker on its way into cond_wait cannot
     * miss this job. Spinning workers never take the mutex, so in the middle of
     * a run this is an uncontended lock and a broadcast with nobody waiting --
     * nanoseconds, against loops that take microseconds. */
    pthread_mutex_lock(&p->mutex);
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n;
    __atomic_store_n(&p->n_running, p->n_workers, __ATOMIC_RELAXED);
    /* release: a worker that sees this generation sees the job fields above */
    __atomic_store_n(&p->generation, p->generation + 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->mutex);

    run_slice(p, fn, ctx, n, 0); /* the master takes the first slice */

    for (spins = 0; spins < RCSD_SPIN_BUDGET; ++spins) {
        if (__atomic_load_n(&p->n_running, __ATOMIC_ACQUIRE) == 0) {
            return;
        }
        CPU_RELAX();
    }
    pthread_mutex_lock(&p->mutex);
    while (__atomic_load_n(&p->n_running, __ATOMIC_ACQUIRE) > 0) {
        pthread_cond_wait(&p->done, &p->mutex);
    }
    pthread_mutex_unlock(&p->mutex);
}

void rcsd_threads_shutdown(void) {
    stop_workers(&g_pool);
}

#endif /* RCSD_THREADS_AVAILABLE */
