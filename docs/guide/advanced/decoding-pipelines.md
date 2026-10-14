# Constructing decoding pipelines

The `Pipe` decoding composes multiple stages into a single processing pipeline. Beyond simple chaining it provides two shared-state mechanisms to share **per-call context** and **persistent state**. This allows independent stages to communicate without coupling their implementations (maybe typing).

## Stage interface

Every stage is a callable with the signature:

```python
def __call__(self, signal, env=None) -> Run | value | None:
    ...
```

There are two return patterns used by stages:

| Return value | Meaning |
|---|---|
| Any value | Passed as the input to the next stage; a `Run` for a stage that transforms recordings, anything else for one that reduces |
| `None` | Pass-through where current data flows unchanged to the next stage |

The `None` pass-through pattern enables feature-extractor stages that read the neural data. 

## Two state scopes

`Pipe` maintains two dicts with different lifetimes:

```python
pipe = Pipe(stages=[...], duration=1000)

pipe.context # cleared at the top of every __call__
pipe.state   # persists across __call__ invocations; cleared by pipe.clear()
```

A stage that wants them accepts the pipe as a keyword; `Pipe.__call__` sees the `pipe` parameter in the stage's signature and passes itself (any other stage is called as `stage(signal, env)`, as before):

```python
class MyStage(Decoding):
    def __call__(self, signal, env=None, pipe=None):
        ctx = pipe.context
        st = pipe.state
```

The env does not remember the decoding it was called with (there is no `env.decoding`): the pipe reaches its stages through this argument, and an encoding that needs the pipe's `state` accepts `decoding=` in its `__call__`, and `Env.__call__` passes the decoding it was called with.

### within-call communication via `context` 

Use `context` when an early stage computes a value that a later stage in the same call needs:

```python
class FeatureExtractor(Decoding):
    duration: int = 100
    context_key: str = "my_features"
    def __call__(self, signal, env=None, pipe=None):
        features = _compute(signal.spike_ids, signal.spike_times)  # over the raw data
        pipe.context[self.context_key] = features
        return None  # pass-through: raw data continues to the next stage


class AugmentOutput(Decoding):
    duration: int = 1
    def __call__(self, result, env=None, pipe=None):
        feats = pipe.context.get("my_features")
        # combine result and feats...
        return combined_result
```

Because `context` is cleared at the start of every `Pipe.__call__`, there is no risk of a previous step's stale values leaking through.

### cross-call persistence via `state`

Use `state` when a stage needs to carry information across multiple simulation steps, for example a running estimate, a hidden model state, or a value written by one call and read by the next:

```python
class RunningMean(Decoding):
    duration: int = 100
    n: int = 0
    def __call__(self, signal, env=None, pipe=None):
        current = _firing_rate(signal.spike_times)

        st = pipe.state

        prev_mean = st.get("running_mean", 0.0)
        self.n += 1
        new_mean  = prev_mean + (current - prev_mean) / self.n
        st["running_mean"] = new_mean

        return signal  # pass the recordings through unchanged
```

`state` is also the right place for values that an external caller wants to inject into the pipeline between steps:

```python
pipe.state["threshold"] = 0.5   # caller writes
# ... later inside a stage ...
thr = pipe.state.get("threshold", 1.0)
```

Calling `pipe.clear()` resets `state` entirely (analogous to `env.clear()` at episode boundaries).

## Complete example

```python
import numpy as np
from livn.decoding import Pipe
from livn.types import Decoding
from livn.utils import P


class SpikeCountExtractor(Decoding):
    duration: int = 500
    def __call__(self, signal, env=None, pipe=None):
        local_counts = {}
        if signal.spike_ids is not None:
            for nid in signal.spike_ids:
                local_counts[int(nid)] = local_counts.get(int(nid), 0) + 1

        all_counts = P.gather(local_counts, comm=env.comm)
        if P.is_root(comm=env.comm):
            merged = {}
            for d in (all_counts or [local_counts]):
                for k, v in d.items():
                    merged[k] = merged.get(k, 0) + v

            pipe.context["spike_counts"] = merged

        return None # the signal passes unchanged to the next stage


class PopulationSummary(Decoding):
    duration: int = 500
    top_k: int = 5
    def __call__(self, signal, env=None, pipe=None):
        counts = pipe.context.get("spike_counts", {})

        top = sorted(
            counts.items(), 
            key=lambda x: x[1], reverse=True
        )[: self.top_k]

        return {
            "total_spikes": sum(counts.values()),
            "active": len(counts),
            "top_units": top,
        }


decoding = Pipe(
    duration=500,
    stages=[
        SpikeCountExtractor(duration=500),
        PopulationSummary(duration=500, top_k=3),
    ],
)

result = env(decoding)
# result = {"total_spikes": 412, "active": 67, "top_units": [(14, 28), ...]}
```

## `get_stage`

`Pipe.get_stage(stage_type)` returns the first stage whose class name matches `stage_type`, or `None`. The argument can be either a class or a plain string name but both match by class name rather than `isinstance`, so the lookup is robust when the same module is imported under two different paths to allow for 'duck-typed overrides'.

```python
# Pass a class
extractor = decoding.get_stage(SpikeCountExtractor)
if extractor is not None:
    extractor.duration = 1000

# Or pass a string name
gym_step = decoding.get_stage("GymStep")
```

## Resetting state

Call `pipe.clear()` to wipe `state` when starting a new episode or experiment:

```python
env.clear()
pipe.clear()
```

`context` is always cleared automatically, you never need to reset it manually.

## Naming convention

The dictionary keys form a contract for interaction between decoding stages. To ensure interoperability, we recommend a module naming convention that uses the `obj.__module__` import path as a prefix, for example `livn.decoding.<key>`

## Custom stages and hashing

When using `DistributedEnv`, pipelines are cached on workers using content-based hashing. Pydantic-based stages (subclasses of `Encoding` or `Decoding`) work out of the box since their hash is derived from their pickle representation.

For plain Python classes used as Pipe stages (e.g. `GymStep`, `ObsAugmentation` subclasses), you must implement `__hash__` and `__eq__` based on structural configuration, not ephemeral state:

```python
class MyStage:
    def __init__(self, n_features: int):
        self.n_features = n_features

    def __hash__(self):
        return hash((type(self).__name__, tuple(sorted(self.__dict__.items()))))

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __call__(self, signal, env=None):
        ...
```

If a stage holds ephemeral state that changes every step (and shouldn't invalidate the cache), override `__getstate__` to exclude it (see `GymStep` and `Pipe` for examples).

For Pydantic `Decoding` subclasses with a `PrivateAttr` that changes every step, put it in the `_state` dict (automatically excluded) or override `__getstate__` to strip it before pickling.
