# The test suite

Four tiers, one directory each. What you run is the directory; there is no
marker expression to remember and none to forget.

| tier | what belongs there | cost |
|---|---|---|
| `unit/` | no `Env`, no MPI, no simulation, no tracing | seconds |
| `contract/` | what a backend must do, asserted once and gated on declared capabilities | ~minutes |
| `concurrency/` | anything that needs more than one rank | minutes, `mpiexec` per test |
| `gradients/` | differentiability: gradients flow, nothing retraces, no stray numpy | minutes |

`gradients/` is dominated by jax compilation: `test_glif.py` alone compiles ~60
distinct solver configurations, and every one of them costs more than it takes
to run. Sharing modules across tests is the lever, not making the tests smaller.

```bash
tests/cli fast                      # unit, under no backend and under diffrax
tests/cli verify --backend neuron   # every tier, one backend
tests/cli verify                    # every tier, all three backends, serialized
tests/cli verify --slow             # including the long reference comparisons
tests/cli verify --backend brian2 --skip-tier concurrency
```

## Nothing downloads a system

Every tier runs on `./testing/culture.json`, a 40-cell `Monolayer` drawn at
init, so a checkout is enough to run the suite. `LIVN_TEST_SYSTEM` points at
that spec, and `--system` replaces it with any graph directory or JSON spec.

What a spec cannot be is an h5 file, and a handful of tests are about the h5
readers themselves -- the pyfive/neuroh5 differential is what lets the native
backend read a graph with no HDF5 library. Those ask for a *graph directory*
through `livn_test_h5_system()` (`LIVN_TEST_H5_SYSTEM`, default
`./systems/graphs/test`) and **skip when there is none**, rather than reading
`LIVN_TEST_SYSTEM` and failing on a spec where they wanted a directory.

Build the fixture instead of fetching one:

```bash
tests/cli generate-system           # ./testing/culture.json -> ./systems/graphs/test
```

Two seconds, 372 KB, and at the current graph format by construction. It needs
`neuroh5` and `mpi4py` (they are what writes it, and what the readers are
checked against), plus what `systems/generate_2d.py` imports. The spec carries
a seed, so the rebuild is deterministic; `--force` replaces an older copy, which
is worth doing after a graph format bump because a stale one still reads, with
only a warning.

`tests/cli download-system` still fetches a published system when you want the
real thing; nothing in the test suite requires it.

## Reproducing a CI failure

The backend suites and the long reference comparisons run weekly, not on every
push, so the first sight of a breakage is usually a red job in a run nobody
watched. `tests/cli ci` is that run, here:

```bash
tests/cli ci                      # every job, hours
tests/cli ci --job test-neuron    # the one that went red
tests/cli ci --list-jobs          # what each job stands for
```

It differs from CI in the two ways that matter for finding things out.

**It does not stop at the first failure.** CI fails a job and abandons the rest
of it; one pass here reports every job, and every step within a job, so a
morning of fixing starts from the whole list rather than from the first item on
it. `--fail-fast` restores the CI behaviour when you already know what you are
looking for.

**Every job's output is kept.** Each goes to `tests/tmp/ci/<timestamp>/<job>.log`
with the CI command it stands for in its first line, alongside an
`environment.txt` -- commit, interpreter, package versions, MPI flavour,
uncommitted files -- and a `summary.txt`. `tests/tmp/ci/latest` points at the
newest run. The summary repeats each failing job's `FAILED`/`ERROR` lines, which
is normally enough to pick the one test to iterate on. `--quiet` keeps the
output in the logs only.

What cannot be reproduced is reported rather than failed. A job whose dependency
this venv does not have is `SKIP`, naming the `uv sync` that would install it,
because a missing `brian2` is not a broken `brian2`. The three-OS matrix of
`test-native` becomes this OS alone. And `--oversubscribe` stays opt-in: CI runs
Open MPI, where the flag is correct, and a local MPICH rejects it and fails every
mpiexec test (`environment.txt` records which one you have).

Only one may run at a time -- a second is refused rather than queued. Two suites
at once produce false timeouts, a 137s test exceeding a 300s limit, and a
contention artefact reads exactly like a bug. Ctrl-C prints the summary of what
had run by then and names the job it stopped in the middle of.

The job list mirrors `.github/workflows/ci.yml`; a job added there needs one
here.

## The fast tier is enforced, not merely intended

`tests/unit/conftest.py` makes the expensive things fail there: constructing an
`Env`, tracing a jax function, or carrying the `mpiexec` marker. Each failure
names the tier the test should have gone to. A budget nobody enforces erodes —
one test builds an `Env` because it was convenient, the next copies it, and a
year later "fast" means four minutes.

The exception is `@pytest.mark.traces`, for the handful of tests whose subject
*is* the traced path (`Run` surviving `jit`, `Stimulus.to_array` under trace).
Their traces are tiny, and the marker keeps the exceptions countable.

## mpiexec tests are batched

Tests marked `mpiexec` with the same rank count run in *one* subprocess rather
than one each, which removes an interpreter start and a NEURON/jax import per
test. The batch is formed lazily, when the first of its members is reached, so
terminal ordering, `-x` and `-k` all behave as they did.

A test that mutates process-global state has to say so with `isolated=True`,
which gives it a subprocess to itself. Two do: `pc.subworlds()` partitions
NEURON's ParallelContext for the rest of the process and nothing puts it back,
and `test_recompile_and_smoke` wipes the compiled-mechanism cache. A
parametrization can carry the marker on its own, so only the case that needs
isolation pays for it.

`--mpi-batch=no` restores one subprocess per test. That is the first thing to
try when a test passes alone and fails in company.

**Silence is not assent.** A test passes only when every rank reported on it. A
rank parked in a collective while its peers finish writes no report, and
counting that as a pass is exactly how the bug this tier exists to catch would
go green. Each rank also arms a watchdog per test, so a genuine hang names the
test and dumps every rank's stack instead of timing out anonymously.

## The collective-symmetry checker

A collective only some ranks reach is the characteristic MPI bug, and waiting
for a deadlock is a poor way to find one: it does not reliably hang (an orphaned
`Barrier` will pair with a later one, leaving the ranks silently one collective
out of step), and when it does hang it hangs downstream of where it went wrong.

So each rank records the collectives it performed during a test, and they compare
at the end. Records are scoped to the communicator they were made on -- keyed by
the world ranks it spans *and* an ordinal, because `Split` followed by two `Dup`s
gives three communicators over the same ranks and conflating them makes
independent traffic look like a divergence. What is compared is the operation and
its root, in order: MPI matches collectives by their order on a communicator, not
by where they were called, so code reaching one collective from two branches is
correct. The call site is reported, not compared.

Runs in **strict** mode: a divergence fails the test that caused it.
`TEST_MPI_SYMMETRY=warn` reports without failing, `=off` disables it. Overhead is
about 4%. A test that is *about* ranks doing different things opts out with
`@pytest.mark.mpiexec(symmetry=False)`.

The finding that kept it out of strict is closed. `env/distributed.py` had the
broker call `local_comm.Free()` while the rest of its group did not, and
`MPI_Comm_free` is collective. It never hung -- implementations treat it as a
local refcount decrement -- and the workers went on holding a communicator the
broker had released: quietly wrong in exactly the way waiting for a deadlock
never finds. `local_comm` is scratch now, freed by every member as soon as the
two communicators the group actually runs on are duplicated from it.

## jax compilation is most of `gradients/`

The solving is milliseconds; building the graphs is minutes. Two things follow.

`jax.grad` is keyed on the *function object*, so a closure rebuilt per
parametrization recompiles the backward pass every time. Taking one gradient
over all the parameters and asserting per parameter took the finite-difference
tests from 117s to 58s and the escape-parameter ones from 77s to 49s. If you add
a parametrized gradient test, share the loss.

Compiled executables are also persisted to `.pytest_cache/jax`, so a second run
of the same tests skips compiling them (~16%). `LIVN_TEST_JAX_CACHE=0` turns
that off; anything else is used as the directory. The cache is keyed on the
compiled graph, so changing the model simply misses rather than going stale.

## Markers

- `mpiexec(n=, timeout=)` — re-run this test under `mpiexec -n N`; see
  `testing/mpiexec.py`.
- `needs("mpi", ...)` — skip unless the backend declares the capability
  (`livn.types.Capability`). Ask what a backend can do, not what it is called.
  Only *differences* are capabilities: something every backend does needs no
  entry, since asking could only return True. Adding a member is backward
  compatible (absent reads as unsupported); removing one is not, because
  `supports()` raises on an unknown name rather than skipping.
- `slow` — a long reference comparison. Deselected by default; `--slow` includes them.
- `traces` — fast-tier budget exemption, above.

## One NEURON simulation per process

NEURON keeps a single global simulation, and `Env.close()` does not give
everything back: a long enough chain of build/close cycles in one interpreter
segfaults inside `h.stdinit()`, somewhere unrelated to whatever built the last
env. Each module passes alone; the tier as one process does not.

So `tests/cli verify` runs each module of the contract tier in its own
interpreter. It costs a second of startup per module. The same constraint is why
a shared module-scoped `Env` fixture is not available: two live envs is already
too many.

## Why the backend is a process, not a parameter

`LIVN_BACKEND` is read at import and frozen (`livn/backend/config.py`), so one
pytest process tests exactly one backend. That is why `tests/cli verify` loops
over backends rather than parametrizing, and why the loop is serialized: the
`mpiexec` tiers would otherwise oversubscribe each other.

## Shared code

`testing/` (repo root, not shipped) holds the fixtures, the mpiexec plugin, the
capability helpers, and `REPO_ROOT`. Import from it rather than from `conftest`:
a conftest is only importable from its own directory, which stopped working once
the tiers appeared.
