/* A persistent worker pool for the per-node and per-site loops.
 *
 * The loops this drives all write only at their own iteration index, so
 * splitting them changes no arithmetic and no summation order: a threaded run
 * is bit-identical to a serial one, which is the whole point -- the backend's
 * contract is that it reproduces NEURON exactly.
 *
 * Partitioning is contiguous and depends only on (n_items, n_threads, id),
 * never on timing, so the split a given thread count produces is deterministic.
 *
 * POSIX threads only. Windows and Emscripten fall back to running the kernel
 * inline on the calling thread, which is what the backend did before this
 * existed; RCSD_NO_THREADS forces that fallback anywhere.
 */
#ifndef RCSD_THREAD_H
#define RCSD_THREAD_H

#if defined(RCSD_NO_THREADS) || defined(_WIN32) || defined(__EMSCRIPTEN__)
#define RCSD_THREADS_AVAILABLE 0
#else
#define RCSD_THREADS_AVAILABLE 1
#endif

/* Below this many iterations a loop runs inline: the wake-and-join round trip
 * costs more than the work. Culture-scale runs are far above it; a single-cell
 * run is far below, and pays nothing. */
#define RCSD_PAR_GRAIN 512

/* A slice of a parallel loop: [begin, end) of the iteration space. */
typedef void (*RCSDKernel)(void* ctx, int begin, int end);

/* Grow or shrink the pool. `n` is the total number of threads that run a
 * parallel loop, counting the calling thread, so 1 means "no workers, run
 * inline". Values below 1 clamp to 1. Returns the count actually in effect
 * (always 1 where threads are unavailable), or -1 if the workers could not be
 * started, in which case the pool is left serial and rcsd_last_error is set.
 *
 * Not safe to call while a parallel loop is running. Each process has one pool.
 */
int rcsd_set_num_threads(int n);

/* The count currently in effect. */
int rcsd_num_threads(void);

/* Run `fn` over [0, n) split across the pool, and return once every slice is
 * done. Runs inline when the pool is serial, when `n < grain`, or when called
 * from inside another parallel loop -- so a caller never has to check.
 */
void rcsd_parallel_for(int n, int grain, RCSDKernel fn, void* ctx);

/* Release the workers. Idempotent; the pool restarts on the next
 * rcsd_set_num_threads. */
void rcsd_threads_shutdown(void);

#endif /* RCSD_THREAD_H */
