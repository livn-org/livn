# Distributed Environment

The `DistributedEnv` fans out simulation calls to MPI worker processes, allowing multiple independent simulations to run in parallel across a cluster or multi-core machine. Each worker maintains its own copy of the simulation environment while the controller process coordinates task submission and result collection:

- Controller (highest MPI rank): submits tasks, collects results, and runs user code.
- Workers (all other ranks): each initialises an independent `Env` and executes simulation calls on demand.

Workers can themselves be split into *subworlds* of multiple MPI processes to exploit NEURON's parallelization of the each simulation. The `subworld_size` parameter controls how many MPI ranks each worker uses internally. 
Thus, the total number of MPI processes required is `N = subworld_size x num_workers + 1` where the `+1` accounts for the controller process.

## Creating a distributed environment

```python
from livn.env.distributed import DistributedEnv

env = DistributedEnv(
    "./systems/graphs/EI1",
    subworld_size=3,  # MPI ranks per worker
)

env.init()
```

::: warning
The `system` argument must be a directory path, not a loaded system object as each worker re-initializes the system from disk independently
:::

## Recording and configuration

`DistributedEnv` mirrors the same configuration API as the standard [`Env`](/guide/concepts/env) where calls are automatically broadcast to all workers:

```python
env.record_membrane_current()
env.record_spikes()
env.apply_model_defaults()
```

## Running simulations

Use the `__call__` interface with an [Encoding](/guide/concepts/encoding) and [Decoding](/guide/concepts/decoding). Each element in `inputs` is dispatched to a separate worker, and the results are collected in order:

```python
from livn.decoding import ChannelRecording

if env.is_root():
    responses = env(
        ChannelRecording(duration=100),
        inputs=[10, 20],  # each input is handled by a different worker
        encoding=my_encoding,
    )
```

Only the controller process (`env.is_root() is True`) should issue simulation calls and process results.

## Async submission

For finer-grained control, you can submit tasks individually and collect results later:

```python
# Submit without blocking
task_id = env.submit_call(decoding, inputs=42, encoding=my_encoding)

# Block until the result is ready
response = env.receive_response()

# Or poll without blocking
result = env.probe_response(task_id)
if result is not None:
    print("Done:", result)
```

## Shutdown

Always shut down the environment when finished to cleanly terminate worker processes:

```python
env.shutdown()
```

## Full example

The following example launches a distributed simulation with two inputs processed in parallel across MPI workers:

<<< @/../examples/distributed_workers.py

To run with 2 workers, each using 3 processes:

```bash
mpirun -n 7 python examples/distributed_workers.py
```

## Pipeline caching

`DistributedEnv` automatically caches encoding and decoding pipelines on workers. On the first `submit_call`, the full pipeline is serialized and sent to a worker; subsequent calls with the same (structurally identical) pipeline send only the action input and a lightweight state patch.

This means that worker-side state persists, e.g. Gymnasium episode state lives on the worker across steps. No manual state shuttling is needed. Furthermore, fitted decodings auto-update, e.g. calling `fit()` on a trainable decoding changes its internal weights, which changes the pipeline's content hash. The next `submit_call` detects the cache miss and automatically re-installs the updated pipeline on the worker. Finally, after a pipeline is installed on a worker, subsequent calls prefer dispatching to the same worker to avoid re-installation. This is fully automatic and requires no changes to user code.
