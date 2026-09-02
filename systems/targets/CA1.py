from __future__ import annotations

import json
import logging

import numpy as np

from livn.decoding import (
    PopulationActiveFraction,
    PopulationFiringRates,
    PopulationSpikeDensity,
    Slice,
)
from livn.utils import P
from systems.targets.protocol import TuningTargets

logger = logging.getLogger(__name__)


GROUP = "|"
SOURCES = {
    "mean_rate": "density",
    "fraction_active": "density",
    "mean_fraction_active_per_bin": "density",
    "std_fraction_active_per_bin": "density",
    "rate_cv": "density",
    "rate_hz": "rates",
    "mean_active_fraction": "active_fraction",
    "std_active_fraction": "active_fraction",
    "tail_rate_hz": "tail_rates",
}

OBJECTIVES = {
    "target_rate": ("mean_rate", lambda x, target: (x - target) ** 2),
    "target_fraction_active": ("fraction_active", lambda x, target: (x - target) ** 2),
    "target_mean_fraction_active": (
        "mean_fraction_active_per_bin",
        lambda x, target: (x - target) ** 2,
    ),
    "target_std_fraction_active": (
        "std_fraction_active_per_bin",
        lambda x, target: (x - target) ** 2,
    ),
    "steady_firing": ("rate_cv", lambda x: x),
    "log_rate": (
        "rate_hz",
        lambda x, target, eps=1e-3: float(
            np.log((max(x, 0.0) + eps) / (target + eps)) ** 2
        ),
    ),
}

CONSTRAINTS = {
    "rate_bound": (
        "mean_rate",
        lambda x, min=0.0, max=float("inf"): np.minimum(x - min, max - x),
    ),
    "min_fraction_active": ("fraction_active", lambda x, min: x - min),
    "steady_firing_bound": (
        "rate_cv",
        lambda x, max: max - x,
        "mean_fraction_active_per_bin",
    ),
    "rate_floor": ("rate_hz", lambda x, min: 1.0 if x > min else -1.0 - (min - x)),
    "fraction_active_band": (
        "mean_active_fraction",
        lambda x, target, tolerance: tolerance - abs(x - target),
    ),
    "tail_rate_bound": (
        "tail_rate_hz",
        lambda x, max: 1.0 if x <= max else -1.0 - (x - max) / 10.0,
    ),
}

INACTIVE = -1.0

FACTOR_BANDS = {
    "mean_rate": lambda reference, k: {"min": reference / k, "max": reference * k},
    "rate_hz": lambda reference, k: {"min": reference / k, "max": reference * k},
    "rate_cv": lambda reference, k: {"max": reference * k},
}

TRANSFORMS = {
    "identity": "transform_identity",
    "log10": "transform_log10",
    "log1p": "transform_log1p",
}


class Scorer:
    def __init__(self, entry: dict, table: dict, kind: str):
        name = entry.get("kind")
        if name not in table:
            raise ValueError(
                f"{entry.get('name', entry)}: {name!r} is not a {kind}; livn "
                f"scores {sorted(table)}"
            )
        if not entry.get("population"):
            raise ValueError(f"{entry.get('name', entry)} names no 'population'")

        self.kind = name
        self.population = entry["population"]
        self.feature, self.fn, *guard = table[name]
        self.guard = guard[0] if guard else None
        self.name = entry.get("name") or f"{self.population}_{name}"
        self.binding = entry.get("binding")  # None -> decided by `tune`
        self.kwargs = {
            k: v
            for k, v in entry.items()
            if k not in ("name", "kind", "population", "binding")
        }

    def __call__(self, measured: float, active: float | None = None) -> float:
        if self.guard is not None and not active:
            return INACTIVE
        return float(self.fn(measured, **self.kwargs))

    def resolve(self, reference: dict) -> Scorer:
        if "factor" not in self.kwargs:
            return self
        build = FACTOR_BANDS.get(self.feature)
        if build is None:
            raise ValueError(
                f"{self.name}: 'factor' has no meaning for {self.feature}; it is "
                f"defined for {sorted(FACTOR_BANDS)}"
            )
        measured = (reference.get(self.population) or {}).get(self.feature)
        if measured is None:
            raise ValueError(
                f"{self.name} is stated as a factor of the reference activity, but "
                f"none is recorded for {self.population}.{self.feature}. Measure it "
                "with `livn systems tune ~ca1 reference --write`, or state the "
                "bounds outright."
            )
        rest = {k: v for k, v in self.kwargs.items() if k != "factor"}
        self.kwargs = {**build(float(measured), float(self.kwargs["factor"])), **rest}
        return self


def expand_grouped(params: dict) -> dict:
    expanded = {}
    for name, value in params.items():
        post, _, rest = name.partition("_")
        parts = rest.split("-")
        if len(parts) == 4 and GROUP in parts[1]:
            source, sections, mech, param = parts
            for section in sections.split(GROUP):
                expanded[f"{post}_{source}-{section}-{mech}-{param}"] = value
        else:
            expanded[name] = value
    return expanded


class CA1(TuningTargets):
    def __init__(
        self,
        config: str = "./systems/graphs/CA1/tuning.json",
        problem: str = "miv",
        populations: list[str] | None = None,
        tune: list[str] | None = None,
        size: int | float | dict[str, int | float] | None = None,
        selection: str | None = None,
        inputs: str | None = None,
        warmup: float | None = None,
        duration: float | None = None,
        activity_fraction: float | None = None,
    ):
        self.config_path = config
        with open(config) as fh:
            document = json.load(fh)

        problems = document.get("problems") or {}
        if problem not in problems:
            raise ValueError(
                f"{config} states no {problem!r} problem; it holds {sorted(problems)}"
            )
        self.problem = problem
        block = problems[problem]

        network = document.get("network") or {}
        self.weights: dict[str, list[dict]] = document.get("weights") or {}
        self.population_targets: dict[str, dict] = document.get("targets") or {}
        density = document.get("density") or {}

        self.density_resolution = float(density.get("temporal_resolution", 2.0))
        self.stability_resolution = float(density.get("stability_resolution", 2.0))
        self.baks_alpha = float(density.get("baks_alpha", 4.77))
        self.baks_beta = density.get("baks_beta")
        self.activity_fraction = float(
            density.get("activity_fraction", 0.5)
            if activity_fraction is None
            else activity_fraction
        )
        self.bin_size = float(density.get("bin_size", 50.0))
        self.tail_window = float(density.get("tail_window", 1000.0))

        self.objectives = [
            Scorer(e, OBJECTIVES, "objective") for e in block.get("objectives") or []
        ]
        if not self.objectives:
            raise ValueError(f"{config}: problem {problem!r} declares no objectives")
        constraints = [
            Scorer(e, CONSTRAINTS, "constraint") for e in block.get("constraints") or []
        ]

        self.scored = sorted({s.population for s in self.objectives + constraints})
        self.tuned = sorted(tune if tune is not None else block.get("tune") or [])
        unscored = [p for p in self.tuned if p not in self.scored]
        if unscored:
            raise ValueError(
                f"{unscored} would be tuned but nothing scores them, so their "
                f"weights would move without consequence; give them an objective "
                f"or a constraint in {config}, or drop them from 'tune'"
            )
        untunable = [p for p in self.tuned if p not in self.weights]
        if untunable:
            raise ValueError(
                f"{untunable} would be tuned but {config} states no weight ranges "
                "for them, so there is nothing to move"
            )

        self.reference = self._read_reference()
        constraints = [c.resolve(self.reference) for c in constraints]
        self.constraints = [
            c
            for c in constraints
            if c.binding is True or (c.binding is None and c.population in self.tuned)
        ]
        self.advisory = [c for c in constraints if c not in self.constraints]

        def scored_against_reference(c) -> float | None:
            recorded = self.reference.get(c.population) or {}
            if recorded.get(c.feature) is None:
                return None
            if c.guard is None:
                return c(float(recorded[c.feature]))
            if recorded.get(c.guard) is None:
                return None
            return c(float(recorded[c.feature]), float(recorded[c.guard]))

        unmeetable = [
            c.name
            for c in self.constraints
            if (score := scored_against_reference(c)) is not None and score <= 0
        ]
        if unmeetable:
            logger.warning(
                "%s would be violated by the graph's own recorded activity",
                unmeetable,
            )

        self.populations = sorted(
            populations if populations is not None else network.get("populations") or []
        )
        unsimulated = [p for p in self.scored if p not in self.populations]
        if unsimulated:
            raise ValueError(
                f"{unsimulated} are scored but not simulated; they would be replaying "
                f"their recorded spike trains, so their objectives would measure the "
                "recording rather than the network"
            )

        self.group_sections = bool(block.get("group_sections", True))
        transform = block.get("transform", "identity")
        if transform not in TRANSFORMS:
            raise ValueError(
                f"{config}: problem {problem!r} names transform {transform!r}; "
                f"livn has {sorted(TRANSFORMS)}"
            )
        self.transform = getattr(self, TRANSFORMS[transform])

        self.inputs = inputs or network["inputs"]
        self.input_namespace = network["input_namespace"]
        self.input_attribute = network.get("input_attribute", "Spike Train")
        self.dt = network.get("dt")
        self.v_init = network.get("v_init")

        self.size = size
        self.selection_name = selection
        if selection is not None and size is not None:
            raise ValueError(
                f"selection={selection!r} names cells the system has already "
                f"resolved and stored, so size={size!r} has nothing to scale; "
                "give one or the other"
            )
        if selection is not None and not isinstance(selection, str):
            raise ValueError(f"selection={selection!r} is not a name")

        self._durations: tuple[float, float] | None = None
        self._warmup_duration = float(
            network.get("warmup_ms", 250.0) if warmup is None else warmup
        )
        stop = network.get("stop_ms")
        if duration is not None:
            self._recording_duration = float(duration)
        elif stop is not None:
            self._recording_duration = float(stop) - self.warmup_duration
            if self._recording_duration <= 0:
                raise ValueError(
                    f"stop_ms={stop} leaves nothing to score after a "
                    f"{self.warmup_duration} ms warmup"
                )
        else:
            self._recording_duration = None

        self._graph: dict | None = None  # gids + input span, read from the h5
        self._reset_state()

    def _reset_state(self):
        self.response_data = None
        self.metrics: dict = {}
        self.measured: set[str] = set()
        self.tunable: set[str] = set()

    def _weight_space(self, model=None) -> dict[str, list]:
        space: dict[str, list] = {}
        for post in self.tuned:
            for row in self.weights[post]:
                lo, hi = row["range"]
                groups = (
                    [GROUP.join(row["sections"])]
                    if self.group_sections
                    else list(row["sections"])
                )
                for group in groups:
                    key = f"{post}_{row['source']}-{group}-{row['mechanism']}-weight"
                    space[key] = [lo, hi, self.transform]
        return space

    def decode_params(self, params: dict, model=None) -> dict:
        return expand_grouped(super().decode_params(params, model=model))

    def describe_params(self, decoded) -> dict:
        groups: dict[str, dict] = {p: {} for p in self.tuned}
        for name, value in decoded.items():
            groups.setdefault(name.split("_", 1)[0], {})[name] = value
        return {f"{post} weights": group for post, group in groups.items() if group}

    def objective_names(self) -> list[str]:
        return [s.name for s in self.objectives]

    def constraint_names(self) -> list[str]:
        return [s.name for s in self.constraints]

    def advisory_constraint_names(self) -> list[str]:
        return [s.name for s in self.advisory]

    def target_populations(self) -> list[str]:
        return list(self.tuned)

    def scored_populations(self) -> list[str]:
        return list(self.scored)

    def target_rates(self) -> dict[str, float]:
        return {
            pop: float(block["firing_rate"])
            for pop, block in self.population_targets.items()
            if "firing_rate" in block
        }

    def active_thresholds(self) -> dict[str, float]:
        return {
            pop: self.activity_fraction * rate
            for pop, rate in self.target_rates().items()
        }

    def read_graph(self, system, comm=None) -> dict:
        if self._graph is not None:
            return self._graph

        ranges = system.population_ranges

        graph = None
        if P.is_root(comm=comm):
            import h5py

            gids, span = {}, 0.0
            with h5py.File(self.inputs, "r") as fh:
                pops = fh["Populations"]
                for pop, (start, _count) in ranges.items():
                    group = pops.get(pop)
                    if group is None:
                        continue
                    if "Synapse Attributes" in group:
                        index = group["Synapse Attributes"]["syn_ids"]["Cell Index"][:]
                        gids[pop] = np.sort(np.asarray(index, dtype=np.int64)) + int(
                            start
                        )
                    trains = group.get(self.input_namespace)
                    if trains is None or self.input_attribute not in trains:
                        continue
                    attribute = trains[self.input_attribute]
                    pointer = attribute["Attribute Pointer"][:]
                    last = pointer[1:] - 1  # index of each cell's final spike
                    last = last[last >= pointer[:-1]]  # skip cells that never fire
                    if len(last):
                        span = max(
                            span, float(attribute["Attribute Value"][last].max())
                        )
            graph = {"gids": gids, "input_span": span}

        self._graph = P.broadcast(graph, comm=comm)
        return self._graph

    def graph_gids(self, system, comm=None) -> dict[str, np.ndarray]:
        return self.read_graph(system, comm=comm)["gids"]

    def durations(self, system, comm=None) -> tuple[float, float]:
        if self._durations is not None:
            return self._durations

        recording = self._recording_duration
        if recording is None:
            span = float(self.read_graph(system, comm=comm)["input_span"])
            recording = max(0.0, span - self.warmup_duration)
            if recording <= 0.0:
                raise ValueError(
                    f"the input trains in {self.inputs} span {span} ms, which "
                    f"leaves nothing to record after a {self.warmup_duration} ms "
                    "warmup"
                )

        self._durations = (self.warmup_duration, float(recording))
        return self._durations

    @property
    def warmup_duration(self) -> float:
        return self._warmup_duration

    @warmup_duration.setter
    def warmup_duration(self, value: float):
        self._warmup_duration = float(value)
        if self._durations is not None:
            self._durations = (self._warmup_duration, self._durations[1])

    @property
    def recording_duration(self) -> float:
        if self._durations is None:
            raise RuntimeError(
                "recording_duration is resolved from the input span; call "
                "durations() (build_env and record() both do) first"
            )
        return self._durations[1]

    @recording_duration.setter
    def recording_duration(self, value: float):
        self._recording_duration = float(value)
        self._durations = (self._warmup_duration, float(value))

    def selection(self, system, comm=None) -> dict[str, np.ndarray]:
        available = self.graph_gids(system, comm=comm)

        missing = [p for p in self.populations if p not in available]
        if missing:
            raise ValueError(
                f"{missing} have no cell data in {self.inputs}; the graph "
                f"provides {sorted(available)}"
            )
        pops = self.populations

        if self.size is None:
            return {p: available[p] for p in pops}

        counts: dict[str, int] = {}
        if isinstance(self.size, dict):
            unknown = [p for p in self.size if p not in pops]
            if unknown:
                raise ValueError(
                    f"size names {unknown}, which are not simulated; simulating {pops}"
                )
            for p in pops:
                want = self.size.get(p)
                if want is None:
                    counts[p] = len(available[p])
                elif isinstance(want, float):
                    counts[p] = max(1, round(want * len(available[p])))
                else:
                    counts[p] = max(1, int(want))
        elif isinstance(self.size, float):
            for p in pops:
                counts[p] = max(1, round(self.size * len(available[p])))
        else:
            total = sum(len(available[p]) for p in pops)
            for p in pops:
                share = int(self.size) * len(available[p]) / max(total, 1)
                counts[p] = max(1, round(share))

        return {p: available[p][: min(counts[p], len(available[p]))] for p in pops}

    def build_env(self, system, model, comm=None, subworld_size=None):
        from livn.env import Env

        env = Env(system, model=model, comm=comm, subworld_size=subworld_size)
        env.selection(
            self.selection_name
            if self.selection_name is not None
            else self.selection(env.system, comm=comm)
        )
        self.durations(env.system, comm=comm)
        env.init()
        if self.v_init is not None:
            env.v_init = float(self.v_init)
        return self.init(env)

    def init(self, env):
        declared = set(expand_grouped(dict.fromkeys(self._weight_space(env.model))))
        wired = set(env.weight_names)
        missing = declared - wired
        if missing and P.is_root(comm=getattr(env, "comm", None)):
            logger.warning(
                "%s: %d/%d tunable weights have no wired synapses "
                "under this selection (will be no-ops), e.g. %s",
                type(self).__name__,
                len(missing),
                len(declared),
                sorted(missing)[:8],
            )
        self.tunable = declared & wired
        return env

    def __call__(self, env, params=None, directory=None):
        self.record(env)
        return self.compute_objectives(env), self.compute_constraints(env)

    def record(self, env):
        self._reset_state()
        self.durations(env.system, comm=getattr(env, "comm", None))
        env.record_spikes()
        env.apply_stimulus_from_h5(
            self.inputs,
            self.input_namespace,
            attribute=self.input_attribute,
            equilibration_duration=0.0,
        )
        total = int(self.warmup_duration + self.recording_duration)
        self.response_data = env.run(total, dt=self.dt, root_only=False)

    def _window(self, env):
        return Slice(
            start=self.warmup_duration,
            stop=self.warmup_duration + self.recording_duration,
        )(self.response_data)

    def measure(self, feature: str, env) -> dict[str, float]:
        source = SOURCES.get(feature)
        if source is None:
            raise ValueError(
                f"no measurement produces {feature!r}; livn measures {sorted(SOURCES)}"
            )
        if source not in self.measured:
            self.measured.add(source)
            self.metrics.update(getattr(self, f"_measure_{source}")(env))
        return self.metrics.get(feature) or {}

    def _measure_density(self, env) -> dict:
        return PopulationSpikeDensity(
            duration=int(self.recording_duration),
            temporal_resolution=self.density_resolution,
            stability_resolution=self.stability_resolution,
            active_threshold=self.active_thresholds(),
            baks_alpha=self.baks_alpha,
            baks_beta=self.baks_beta,
        )(self._window(env), env)

    def _measure_rates(self, env) -> dict:
        rates = PopulationFiringRates(duration=int(self.recording_duration))(
            self._window(env), env
        )
        return {"rate_hz": rates["rates_hz"]}

    def _measure_active_fraction(self, env) -> dict:
        return PopulationActiveFraction(
            duration=int(self.recording_duration), bin_size=self.bin_size
        )(self._window(env), env)

    def _measure_tail_rates(self, env) -> dict:
        window = min(self.tail_window, self.recording_duration)
        stop = self.warmup_duration + self.recording_duration
        tail = Slice(start=stop - window, stop=stop)(self.response_data)
        rates = PopulationFiringRates(duration=int(window))(tail, env)
        return {"tail_rate_hz": rates["rates_hz"]}

    def _score(self, scorers: list[Scorer], env) -> dict:
        result = {}
        for scorer in scorers:
            measured = float(
                self.measure(scorer.feature, env).get(scorer.population, 0.0)
            )
            active = (
                float(self.measure(scorer.guard, env).get(scorer.population, 0.0))
                if scorer.guard
                else None
            )
            result[scorer.name] = (scorer(measured, active), measured)
        return result

    def compute_objectives(self, env) -> dict:
        return self._score(self.objectives, env)

    def compute_constraints(self, env) -> dict:
        advisory = self._score(self.advisory, env)
        self.metrics["advisory"] = advisory
        violated = {k: v for k, v in advisory.items() if v[0] <= 0}
        if violated and P.is_root(comm=getattr(env, "comm", None)):
            logger.warning(
                "%d/%d advisory constraints violated, e.g. %s. These populations "
                "are simulated but not tuned, so the search cannot answer for "
                "them; add them to `tune` to bind these bounds.",
                len(violated),
                len(advisory),
                {k: round(v[1], 2) for k, v in list(violated.items())[:5]},
            )
        return self._score(self.constraints, env)

    def _read_reference(self) -> dict:
        return {
            pop: block["reference"]
            for pop, block in self.population_targets.items()
            if block.get("reference")
        }

    def write_reference(self, measured: dict, config: str | None = None) -> str:
        path = config or self.config_path
        with open(path) as fh:
            document = json.load(fh)

        targets = document.setdefault("targets", {})
        for pop, features in measured.items():
            targets.setdefault(pop, {})["reference"] = {
                key: float(value) for key, value in sorted(features.items())
            }
            self.population_targets.setdefault(pop, {})["reference"] = targets[pop][
                "reference"
            ]
        self.reference = self._read_reference()

        with open(path, "w") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")
        return path

    def reference_targets(
        self,
        system,
        populations: list[str] | None = None,
        sample: int = 1500,
        comm=None,
    ) -> dict[str, dict[str, float]]:
        import h5py

        from livn.run import Run

        pops = list(populations or self.scored)
        ranges = system.population_ranges
        start, recording = self.durations(system, comm=comm)
        stop = start + recording

        ids: list[int] = []
        times: list[float] = []
        cells: dict[str, dict] = {}
        with h5py.File(self.inputs, "r") as fh:
            for pop in pops:
                trains = fh["Populations"][pop][self.input_namespace][
                    self.input_attribute
                ]
                pointer = trains["Attribute Pointer"][:]
                values = trains["Attribute Value"]
                index = trains["Cell Index"][:]
                first = int(ranges[pop][0])

                rng = np.random.default_rng(0)
                picks = (
                    np.sort(rng.choice(len(index), sample, replace=False))
                    if len(index) > sample
                    else np.arange(len(index))
                )
                gids = []
                for k in picks:
                    lo, hi = int(pointer[k]), int(pointer[k + 1])
                    gid = int(index[k]) + first
                    gids.append(gid)
                    if hi <= lo:
                        continue
                    train = values[lo:hi]
                    train = train[(train >= start) & (train < stop)]
                    ids.extend([gid] * len(train))
                    times.extend(train - start)
                cells[pop] = dict.fromkeys(gids)

        class Recording:
            def __init__(self, cells, comm):
                self.cells = cells
                self.comm = comm
                self.system = type("system", (), {"population_ranges": ranges})

        metrics = PopulationSpikeDensity(
            duration=int(stop - start),
            temporal_resolution=self.density_resolution,
            stability_resolution=self.stability_resolution,
            active_threshold=self.active_thresholds(),
            baks_alpha=self.baks_alpha,
            baks_beta=self.baks_beta,
        )(
            Run(duration=stop - start).add_spikes(
                np.asarray(ids, dtype=np.int64), np.asarray(times, dtype=np.float64)
            ),
            Recording(cells, comm),
        )

        return {
            pop: {feature: values[pop] for feature, values in metrics.items()}
            for pop in pops
        }
