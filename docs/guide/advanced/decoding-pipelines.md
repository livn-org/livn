# Constructing decoding pipelines

The `Pipe` decoding composes multiple stages into a single processing pipeline. Beyond simple chaining it provides two shared-state mechanisms to share **per-call context** and **persistent state**. This allows independent stages to communicate without coupling their implementations (maybe typing).

## Stage interface

Every stage is a callable with the signature:

```python
def __call__(self, env, *data) -> tuple | scalar | array | None:
    ...
```

There are three return patterns used by stages:

| Return value | Meaning |
|---|---|
| `tuple` | Transformed data passed as the input to the next stage |
| Scalar / array | Wrapped to `(value,)` and passed on |
| `None` | Pass-through where current data flows unchanged to the next stage |

The `None` pass-through pattern enables feature-extractor stages that read the neural data. 

## Two state scopes

`Pipe` maintains two dicts with different lifetimes:

```python
pipe = Pipe(stages=[...], duration=1000)

pipe.context # cleared at the top of every __call__
pipe.state   # persists across __call__ invocations; cleared by pipe.clear()
```

Both are accessible inside any stage as:

```python
pipe = getattr(env, "decoding", None)  # set by Env.__call__
ctx  = getattr(pipe, "context", {})
st   = getattr(pipe, "state",   {})
```

### within-call communication via `context` 

Use `context` when an early stage computes a value that a later stage in the same call needs:

```python
class FeatureExtractor(Decoding):
    duration: int = 100
    context_key: str = "my_features"

    def __call__(self, env, it, tt, iv, vv, im, mp):
        features = _compute(it, tt)  # some computation over raw data
        pipe = getattr(env, "decoding", None)
        if hasattr(pipe, "context"):
            pipe.context[self.context_key] = features
        return None  # pass-through: raw data continues to the next stage


class AugmentOutput(Decoding):
    duration: int = 1

    def __call__(self, env, result):
        pipe  = getattr(env, "decoding", None)
        ctx   = getattr(pipe, "context", {})
        feats = ctx.get("my_features")
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

    def __call__(self, env, it, tt, iv, vv, im, mp):
        current = _firing_rate(tt)

        pipe = getattr(env, "decoding", None)
        st   = getattr(pipe, "state", {})

        prev_mean = st.get("running_mean", 0.0)
        self.n += 1
        new_mean  = prev_mean + (current - prev_mean) / self.n
        st["running_mean"] = new_mean

        return it, tt, iv, vv, im, mp  # pass data through unchanged
```

`state` is also the right place for values that an external caller wants to inject into the pipeline between steps:

```python
pipe.state["threshold"] = 0.5   # caller writes
# ... later inside a stage ...
thr = getattr(getattr(env, "decoding", None), "state", {}).get("threshold", 1.0)
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

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_counts = {}
        if it is not None:
            for nid in it:
                local_counts[int(nid)] = local_counts.get(int(nid), 0) + 1

        all_counts = P.gather(local_counts, comm=env.comm)
        if P.is_root(comm=env.comm):
            merged = {}
            for d in (all_counts or [local_counts]):
                for k, v in d.items():
                    merged[k] = merged.get(k, 0) + v

            pipe = getattr(env, "decoding", None)
            if hasattr(pipe, "context"):
                pipe.context["spike_counts"] = merged

        return None # raw 6-tuple pass unchanged to the next stage


class PopulationSummary(Decoding):
    duration: int = 500
    top_k: int = 5

    def __call__(self, env, it, tt, iv, vv, im, mp):
        pipe   = getattr(env, "decoding", None)
        counts = getattr(pipe, "context", {}).get("spike_counts", {})

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

`Pipe.get_stage(stage_type)` returns the first stage matching a given type, or `None`. This is useful for post-hoc configuration:

```python
extractor = decoding.get_stage(SpikeCountExtractor)
if extractor is not None:
    extractor.duration = 1000
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
