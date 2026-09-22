import contextlib
import json
import os
import time
from collections.abc import Mapping
from typing import Literal

from machinable import Interface, get
from machinable.config import Field as ConfigField
from machinable.config import import_ref, to_dict
from machinable.errors import ConfigurationError, ExpansionError
from machinable.interface import extract
from pydantic import BaseModel, ConfigDict

from livn.utils import P
from systems.targets.protocol import Target
from systems.tune.report import (
    anatomy_table,
    as_json,
    band_legend,
    front_table,
    in_band_counts,
    layout_report,
    wide,
)
from systems.tune.report import gist as gist_report
from systems.tune.sizing import min_ranks_per_worker, plan_execution


def resolve_target(spec) -> Target | None:
    if spec is None:
        return None
    module, version = extract(to_dict(spec) if not isinstance(spec, str) else spec)

    tried = []
    for candidate in _modules(module):
        tried.append(candidate)
        if _names_a_target(candidate):
            return Interface.make(candidate, version, base_class=Target)
    raise ModuleNotFoundError(f"{module!r} names no target; looked for one in {tried}")


def _names_a_target(module: str) -> bool:
    from machinable.project import Project, import_interface

    try:
        return import_interface(Project.get().path(), module, Target) is not None
    except (ModuleNotFoundError, ConfigurationError):
        return False


def _modules(module: str):
    head, _, _tail = module.rpartition(".")
    for name in (module, head):
        if not name:
            continue
        yield name
        if not name.startswith("systems."):
            yield f"systems.{name}"


def _system_label(spec) -> str:
    if not isinstance(spec, Mapping) or "cls" not in spec:
        return str(spec)
    with contextlib.suppress(Exception):
        from livn.system import resolve

        system = resolve(spec)
        return f"{system!r} uuid {system.uuid}"
    return str(spec.get("cls"))


def _described(spec, built=None) -> dict:
    if isinstance(spec, Mapping) and "cls" in spec:
        from livn.types import _plain

        return {"cls": spec["cls"], "kwargs": _plain(spec.get("kwargs") or {})}
    if isinstance(spec, (list, tuple)):
        cls, kwargs = [*list(spec), {}][:2]
        return {"cls": cls, "kwargs": dict(kwargs)}
    if isinstance(spec, str) and not spec.endswith(".json") and "/" not in spec:
        return {"cls": spec, "kwargs": {}}

    from livn.types import _describe

    return _describe(built if built is not None else spec)


def _or_default(value, default):
    return default if value is None else value


def _parse_overrides(params) -> dict:
    """`"a=1,b=2"` or a mapping -> `{"a": 1.0, "b": 2.0}`."""
    if not params:
        return {}
    if isinstance(params, dict):
        return {str(k): float(v) for k, v in params.items()}
    pairs = [part for part in str(params).split(",") if part.strip()]
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{pair!r} is not `name=value`")
        name, value = pair.split("=", 1)
        out[name.strip()] = float(value)
    return out


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        target: str | list = ConfigField("targets.EI", identifying=False)
        system: str | int | dict | None = None
        model: str | list | None = None
        selection: str | None = None
        trials: int = 1

        n_initial: int | None = None
        n_epochs: int | None = None
        nprocs_per_worker: int | None = ConfigField(None, identifying=False)
        population_size: int = 100
        num_generations: int = 10

        optimizer: Literal["nsga2", "age", "smpso", "cmaes", "trs"] = "nsga2"

        class SurrogateConfig(BaseModel):
            method_name: (
                str
                | Literal[
                    "gpr",
                    "egp",
                    "megp",
                    "mdgp",
                    "mdspp",
                    "vgp",
                    "svgp",
                    "spv",
                    "siv",
                    "crv",
                ]
                | None
            ) = None
            method_kwargs: dict = {}
            custom_training: str | None = "dmosopt.model_transformer.joint"
            custom_training_kwargs: dict | None = {}

        surrogate: SurrogateConfig = SurrogateConfig()

        autosize: bool = ConfigField(True, identifying=False)
        worker_memory_max: float | None = ConfigField(None, identifying=False)
        cores_per_node: int | None = ConfigField(None, identifying=False)
        max_nodes: int | None = ConfigField(None, identifying=False)
        min_ranks_per_worker: int | None = ConfigField(None, identifying=False)

    def target(self) -> Target:
        self._assert_expanded("resolve the target of")
        if "target" not in self._cache:
            self._cache["target"] = resolve_target(self.config.target)
        return self._cache["target"]

    def system(self):
        if self.config.system is not None:
            return to_dict(self.config.system)
        return self.target().system_spec()

    def build_model(self):
        spec = self.config.model or self.target().model_spec()
        return import_ref(to_dict(spec) if not isinstance(spec, str) else spec)

    def model_ref(self):
        spec = self.config.model or self.target().model_spec()
        return to_dict(spec) if not isinstance(spec, str) else spec

    def on_compute_predicate(self):
        predicate = {}
        with contextlib.suppress(Exception):
            from livn.system import resolve

            uuid = getattr(resolve(self.system()), "uuid", None)
            if uuid:
                predicate["system"] = uuid
        with contextlib.suppress(Exception):
            target = self.target()
            predicate["target"] = target.module
            predicate.update(target.on_compute_predicate() or {})
        return predicate

    def version_fit(self, observation: str, **options):
        return {"target": ["targets.EI", {"observation": observation, **options}]}

    def version_ca1(self, problem: str = "uniform", selection: str | None = None):
        options = {"problem": problem}
        if selection is not None:
            options["selection"] = selection
        return {
            "system": "./systems/graphs/CA1",
            "model": "livn.models.ca1.PinskyRinzel",
            "target": ["targets.CA1", options],
        }

    def version_cell(self, config: str):
        """One cell specification under `systems/targets/cells`."""
        from systems.targets.cells.SingleCell import SingleCellOptConfig

        parsed = SingleCellOptConfig.from_yaml(config)
        return {
            "system": {parsed.Population: 1},
            "model": "livn.models.rcsd.ReducedCalciumSomaDendrite",
            "target": ["targets.cells.SingleCell", f"~yaml('{config}')"],
        }

    def version_motoneuron(self):
        return self.version_cell("systems/targets/cells/motoneuron.yaml")

    def version_renshaw_perry(self):
        return self.version_cell("systems/targets/cells/rc_v1in_perry.yaml")

    def version_renshaw_invitro(self):
        return self.version_cell("systems/targets/cells/rc_v1in_invitro.yaml")

    def version_renshaw(self):
        return self.version_renshaw_perry()

    @staticmethod
    def axis_cells(directory: str = "./systems/targets/cells"):
        from glob import glob

        from systems.targets.cells.SingleCell import SingleCellOptConfig

        return [
            {
                "system": {SingleCellOptConfig.from_yaml(path).Population: 1},
                "model": "livn.models.rcsd.ReducedCalciumSomaDendrite",
                "target": ["targets.cells.SingleCell", f"~yaml('{path}')"],
            }
            for path in sorted(glob(os.path.join(directory, "*.yaml")))
            if not SingleCellOptConfig.from_yaml(path).Retired
        ]

    OBSERVATIONS = (
        "./systems/targets/miv/processed/targets",
        "./systems/targets/miv/processed/recording-sep-2026/targets",
    )

    @staticmethod
    def axis_cultures(directory: str | None = None, per_arm: int | None = None):
        from glob import glob

        from systems.targets.observation import composition_of

        directories = [directory] if directory else list(Tune.OBSERVATIONS)
        arms: dict[str, list[str]] = {}
        for path in sorted(
            p for d in directories for p in glob(os.path.join(d, "*.json"))
        ):
            with open(path) as f:
                document = json.load(f)

            if "conditions" not in document:
                continue
            arms.setdefault(composition_of(document) or "unstated", []).append(path)

        if not arms:
            raise FileNotFoundError(
                f"no target documents in {directories}; name the directory "
                "the observations were extracted to"
            )

        return [
            {"target": ["targets.EI", {"observation": path}]}
            for arm in sorted(arms)
            for path in (arms[arm] if per_arm is None else arms[arm][:per_arm])
        ]

    def _sopt_config(self, target, model, layout: dict | None = None) -> dict:
        surrogate_config = {}
        for k, v in self.config.surrogate.items():
            surrogate_config["surrogate_" + k] = v

        sizing = target.sizing
        return {
            "system": self.system(),
            "dopt_params": {
                "space": target.search_space(model),
                "obj_fun_init_args": {
                    "model": self.model_ref(),
                    "target": self.target_ref(),
                    "trials": self.config.trials,
                    "selection": self.config.selection,
                },
                "optimizer_name": self.config.optimizer,
                "n_epochs": _or_default(self.config.n_epochs, sizing.n_epochs),
                "n_initial": _or_default(self.config.n_initial, sizing.n_initial),
                "population_size": self.config.population_size,
                "num_generations": self.config.num_generations,
                **surrogate_config,
            },
            **(layout or {}),
        }

    def target_ref(self) -> list:
        target = self.target()
        return [target.module, target.settings]

    def _plan(self, target=None, model=None) -> dict | None:
        if not self.config.autosize:
            return None

        target = target or self.target()
        model = model or self.build_model()
        worker_memory = getattr(target, "worker_memory", None)
        if worker_memory is None:
            return None

        sopt = get("interface.sopt", self._sopt_config(target, model))
        system = self.system()
        selection = target.selection_name or self.config.selection
        floor = self.config.min_ranks_per_worker or min_ranks_per_worker(
            target.sizing.min_ranks_per_worker
        )
        asked = target.sizing.min_ranks_per_worker

        plan = plan_execution(
            lambda ranks: worker_memory(system, ranks, selection),
            sopt.num_evals_per_epoch,
            node_gib=self.config.worker_memory_max,
            cores_per_node=self.config.cores_per_node,
            max_nodes=self.config.max_nodes,
            min_ranks_per_worker=floor,
        )
        plan["system"] = _system_label(system)
        plan["selection"] = selection
        plan["space"] = sopt.num_parameters
        plan["initial_evals"] = sopt.num_initial_samples
        plan["total_evals"] = sopt.num_evals_total
        plan["floor"] = floor
        plan["floor_asked"] = asked
        plan["n_initial"] = _or_default(self.config.n_initial, target.sizing.n_initial)
        plan["n_epochs"] = _or_default(self.config.n_epochs, target.sizing.n_epochs)
        return plan

    def sizing(self):
        if self.is_unexpanded():
            for run in self.interfaces:
                run.sizing()
            return

        target = self.target()
        plan = self._plan(target, self.build_model())
        if plan is None:
            why = (
                "the target cannot price its own network"
                if self.config.autosize
                else "autosize=False"
            )
            print(
                f"\n  autosize      off ({why}) -- nprocs_per_worker="
                f"{self.config.nprocs_per_worker or target.sizing.nprocs_per_worker}"
                ", and the rank count is whatever the caller passes\n"
            )
            return

        stated = bool(
            self.config.worker_memory_max or os.environ.get("LIVN_WORKER_MEMORY_MAX")
        )
        print(layout_report(plan, self.config, stated))
        return

    def launch(self):
        if self.is_unexpanded():
            return super().launch()

        target, model = self.target(), self.build_model()
        plan = self._plan(target, model)
        layout = (
            {
                "nprocs_per_worker": plan["ranks_per_worker"],
                "nodes": str(plan["nodes"]),
                "ranks": plan["ranks_per_node"],
            }
            if plan
            else {
                "nprocs_per_worker": (
                    self.config.nprocs_per_worker or target.sizing.nprocs_per_worker
                )
            }
        )

        get("interface.sopt", self._sopt_config(target, model, layout)).launch()

        return self

    def observation(self) -> str | None:
        ref = to_dict(self.config.target)
        if isinstance(ref, str):
            return None
        for item in ref[1:]:
            if isinstance(item, Mapping) and item.get("observation"):
                return str(item["observation"])
        return None

    def headline(self) -> str:
        ref = to_dict(self.config.target)
        if isinstance(ref, str):
            return ref
        module, *version = ref
        observation = self.observation()
        if observation:
            return f"{module}  {os.path.basename(observation)}"
        for item in version:
            if isinstance(item, str):
                return f"{module}  {item}"
        return str(module)

    def _for_each(self, report: str, *args, **kwargs):
        results = []
        for run in self.interfaces:
            print(f"\n=== {run.headline()}")
            results.append(getattr(run, report)(*args, **kwargs))
        return results if any(r is not None for r in results) else None

    def _one_of(self, command: str) -> None:
        if not self.is_unexpanded():
            return
        runs = "\n".join(
            f"    {run.observation() or run.headline()}" for run in self.interfaces
        )
        raise ExpansionError(
            f"`{command}` acts on one solution of one run, and this sweep "
            f"denotes several:\n\n{runs}\n\n"
            'Name one of them -- `~fit(observation="...")` for a culture -- or '
            "`--export` the sweep, which writes a front per run.\n"
        )

    def _ranked_best(self, optimization, target):
        best = optimization.get_best()
        if hasattr(target, "rank_solutions"):
            best = target.rank_solutions(best)
        return best

    def reference(self, populations=None, sample: int = 1500, write: bool = False):
        """Measure the graph's own stored spike trains.

        livn systems tune ~ca1 '--reference(write=True)'
        """
        if self.is_unexpanded():
            return self._for_each(
                "reference", populations=populations, sample=sample, write=write
            )

        import pandas as pd

        from livn.system import resolve

        target = self.target()
        if not hasattr(target, "reference_targets"):
            print(f"{self.config.target} states no reference activity")
            return None

        measured = target.reference_targets(
            resolve(self.system()),
            populations=(
                None
                if populations is None
                else [p.strip() for p in populations.split(",")]
            ),
            sample=sample,
        )

        if write:
            print("wrote", target.write_reference(measured))

        rows = []
        for pop, features in measured.items():
            row = {"population": pop, "cells": int(features.get("n_total", 0))}
            row.update(
                {
                    k: round(float(v), 4)
                    for k, v in features.items()
                    if k not in ("n_total", "n_active")
                }
            )
            rows.append(row)
        table = pd.DataFrame(rows)
        print(wide(table))
        return table

    def summary(self, params=None, sort=True):
        if self.is_unexpanded():
            return self._for_each("summary", params=params, sort=sort)

        optimization = self._optimization()
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self.target(), self.build_model()
        best = self._ranked_best(optimization, target)
        f = best.get("f")
        if f is None or len(f) == 0:
            print("No solutions")
            return None

        bands = target.bands()
        wanted = [p.strip() for p in (params or "").split(",") if p.strip()]

        decoded = {}
        if wanted:
            for i in range(len(f)):
                values = target.decode_params(
                    optimization.parameter_vector_to_dict(
                        list(map(float, best["x"].to_numpy()[i]))
                    ),
                    model=model,
                    strict=True,
                )
                decoded[i] = {name: values.get(name) for name in wanted}

        table = front_table(f, bands, decoded)
        if sort and "_n" in table:
            table = table.sort_values("_n", ascending=False)
        table = table.drop(columns=[c for c in ("_n",) if c in table])

        print(wide(table))
        print(band_legend(bands))
        return table

    def _solution(self, loc: int, front: str | None = None) -> dict:
        from livn.system import resolve

        if front:
            with open(front) as f:
                document = json.load(f)
            solutions = {int(s["loc"]): s for s in document["solutions"]}
            if loc not in solutions:
                raise ValueError(
                    f"no solution loc={loc} on this front; it has "
                    f"{', '.join(str(k) for k in sorted(solutions))}"
                )
            solution = solutions[loc]
            selection = document.get("selection")
            spec = document["system"]
            target = self._front_target(document)
            model = import_ref(document["model"])
            decoded = dict(solution["params"])
            meta = {
                "loc": loc,
                "ranked_by": "rank_solutions",
                "objectives": dict(solution.get("objectives") or {}),
                "features": dict(solution.get("features") or {}),
                "feasible": bool(solution.get("feasible")),
                "space": sorted(decoded),
                "source": document.get("source"),
                "target": document["target"],
                "model": document["model"],
            }
        else:
            selection = self.config.selection
            spec = self.system()

            optimization = self._optimization()
            if not optimization.is_materialized() or not os.path.isfile(
                optimization.output_filepath
            ):
                raise ValueError("no data yet; nothing has been evaluated")

            target, model = self.target(), self.build_model()
            best = self._ranked_best(optimization, target)
            if best.get("f") is None or len(best["f"]) == 0:
                raise ValueError("no solutions on this front")

            decoded = target.decode_params(
                optimization.parameter_vector_to_dict(
                    list(map(float, best["x"].to_numpy()[loc]))
                ),
                model=model,
                strict=True,
            )

            constraints = best.get("c")
            meta = {
                "loc": loc,
                "ranked_by": "rank_solutions",
                "objectives": {k: float(v) for k, v in best["y"].iloc[loc].items()},
                "features": {k: float(v) for k, v in best["f"].iloc[loc].items()},
                "feasible": bool(
                    constraints is None or (constraints.iloc[loc] > 0).all()
                ),
                "space": sorted(self._recorded_space() or decoded),
                "source": optimization.output_filepath,
                "target": self.target_ref(),
                "model": self.model_ref(),
            }
        return {
            "loc": loc,
            "params": decoded,
            "meta": meta,
            "target": target,
            "model": model,
            "system": resolve(spec),
            "spec": spec,
            "selection": selection,
        }

    def _front_target(self, document: dict) -> Target:
        try:
            return resolve_target(document["target"])
        except Exception as _ex:
            target = self.target()
            print(
                f"NOTE: this front was written by an older `tune` and its "
                f"target cannot be rebuilt as it was ({_ex}). Reading it "
                f"against {target.module!r} as this config states it; "
                "`--check` would measure a different problem."
            )
            return target

    def promote(
        self,
        group: str = "default",
        loc: int | None = None,
        selection: str | None = None,
        force: bool = False,
        front: str | None = None,
        directory: str | None = None,
    ):
        self._one_of("promote")

        if loc is None:
            loc = int(os.environ.get("LOC", 0))

        picked = self._solution(int(loc), front)
        loc = picked["loc"]
        decoded, meta = picked["params"], picked["meta"]
        system = picked["system"]
        if selection is None:
            selection = picked["selection"]

        target = picked["target"]
        spec = picked["spec"]
        if hasattr(target, "set_params"):
            decoded = target.set_params(dict(decoded))
            spec = getattr(target, "system_for", lambda _p: None)(decoded) or spec
        meta["fitted"] = sorted(decoded)

        if selection:
            meta["selection"] = selection

        name = "-".join(
            part for part in (selection or "", group) if part and part != "default"
        )
        filename = f"env-{name}.json" if name else "env.json"
        if directory is None:
            directory = (
                os.path.join(os.path.dirname(os.path.abspath(front)), "promoted")
                if front
                else self.local_directory("promoted")
            )
        path = os.path.join(directory, filename)

        if os.path.isfile(path) and not force:
            raise FileExistsError(
                f"{path!r} already exists, and runs refer to it by name; pass "
                "force=True to rebind it"
            )

        document = {
            "system": _described(spec, system),
            "model": _described(meta.get("model") or self.model_ref()),
            "io": None,
            "selection": selection or None,
            "params": {k: float(v) for k, v in decoded.items()},
            "meta": meta,
        }

        if not P.is_root():
            return path
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(document, f, indent=2, sort_keys=True)
        print(f"promoted loc={loc} to {path}\n  use with: Env.from_json({path!r})")
        return path

    def check(
        self,
        loc: int | None = None,
        seeds: int = 3,
        front: str | None = None,
        duration: float | None = None,
        params=None,
        raster: bool = False,
        directory: str | None = None,
    ):
        """Re-evaluate one front point on `seeds` noise streams and read its bursts.

            LIVN_BACKEND=neuron mpiexec -n 2 livn systems tune \
                '~fit(observation="…/E_E-sample2_15.json")' '--check(loc=3, seeds=3)'

        Args:
            loc: Solution on the front; `LOC` in the environment otherwise.
            seeds: Independent noise streams to run. Each one costs a full
                recording -- tens of minutes on the replica -- so three is the
                cheapest answer to "does it burst on another stream", not a
                statistic.
            front: An exported `front.json`; the stored run otherwise.
            duration: Recording per repeat, ms; the target's otherwise.
            params: Overrides on the solution, as `"name=value,..."` or a
                mapping, for asking what one parameter does without a tune.
            raster: Also draw `systems.plots.BurstRaster` for every repeat.
            directory: Where to write; beside the front document, or the
                run's own storage, otherwise.
        """
        self._one_of("check")

        import numpy as np

        import livn

        if loc is None:
            loc = int(os.environ.get("LOC", 0))

        picked = self._solution(int(loc), front)
        target, model = picked["target"], picked["model"]
        decoded = {**picked["params"], **_parse_overrides(params)}

        if duration is not None:
            target.recording_duration = float(duration)

        if directory is None:
            directory = (
                os.path.join(os.path.dirname(os.path.abspath(front)), "check")
                if front
                else self.local_directory("check")
            )
        comm = P.comm()
        root = P.is_root(comm=comm)
        if root:
            os.makedirs(directory, exist_ok=True)

        env = livn.make(
            {
                "system": picked["spec"],
                "model": model,
                "io": target.io() if hasattr(target, "io") else None,
                "selection": picked["selection"] or None,
            },
            comm=comm,
        )
        env = target.init(env)

        rows = []
        try:
            for stream in range(int(seeds)):
                started = time.time()
                env.clear()
                env.reseed_noise(stream)
                env.set_params(target.set_params(dict(decoded)))
                target(env, params=None)
                rows.append(dict(target.metrics))

                data = target.response_data
                gathered = None if data is None else data.gather(comm=comm, root=0)
                if root:
                    print(
                        f"  stream {stream}: {target.simulated_ms:.0f} ms in "
                        f"{time.time() - started:.0f} s",
                        flush=True,
                    )
                    path = os.path.join(directory, f"loc{loc}-stream{stream}.npz")
                    np.savez_compressed(
                        path,
                        spike_ids=np.asarray(gathered.spike_ids, dtype=np.int64),
                        spike_times=np.asarray(gathered.spike_times, dtype=np.float64),
                        coordinates=np.asarray(env.system.neuron_coordinates),
                        meta=json.dumps(
                            {
                                "loc": loc,
                                "stream": stream,
                                "system": str(picked["spec"]),
                                "warmup_ms": float(target.warmup_duration),
                                "duration_ms": float(target.recording_duration),
                                "parameters": {k: float(v) for k, v in decoded.items()},
                                "metrics": {
                                    k: float(v)
                                    for k, v in target.metrics.items()
                                    if isinstance(v, (int, float))
                                },
                            }
                        ),
                    )
                    if raster:
                        from systems.plots import BurstRaster

                        BurstRaster(
                            warmup=float(target.warmup_duration),
                            duration=float(target.recording_duration),
                            title=f"loc {loc}, stream {stream}",
                        )(
                            gathered,
                            path.replace(".npz", ".png"),
                            env=env,
                        )
        finally:
            env.close()

        if not root:
            return None

        bands = target.bands()
        print(f"\nloc={loc} over {len(rows)} noise streams, against the culture:\n")
        table = anatomy_table(rows, target.targets(), bands)
        print(table)
        print(f"\nwrote {len(rows)} repeats to {directory}")
        return table

    def _recorded_space(self) -> list[str] | None:
        """The search space the run actually used, as machinable stored it."""
        try:
            return list(self._optimization().config.dopt_params.space.keys())
        except (AttributeError, KeyError, IndexError, TypeError):
            return None

    def export(self, path: str | None = None, feasible_only: bool = False):
        if self.is_unexpanded():
            return self._for_each("export", path=path, feasible_only=feasible_only)

        import numpy as np

        optimization = self._optimization()
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self.target(), self.build_model()

        h5 = optimization.load_h5()
        n_rows, n_evals, n_epochs = self._evaluation_counts(h5)

        best = self._ranked_best(optimization, target)
        print(f"Front: {len(best['x'])} solutions over {n_evals} evaluations")

        c = np.asarray(best["c"]) if best.get("c") is not None else None

        solutions = []
        for loc in range(len(best["x"])):
            if feasible_only and c is not None and (c[loc] <= 0).any():
                continue
            decoded = target.decode_params(
                optimization.parameter_vector_to_dict(
                    list(map(float, best["x"].to_numpy()[loc]))
                ),
                model=model,
                strict=True,
            )
            solutions.append(
                {
                    "loc": loc,
                    "objectives": {k: float(v) for k, v in best["y"].iloc[loc].items()},
                    "features": {k: float(v) for k, v in best["f"].iloc[loc].items()},
                    "constraints": (
                        {k: float(v) for k, v in best["c"].iloc[loc].items()}
                        if best.get("c") is not None
                        else None
                    ),
                    "feasible": bool(c is None or (c[loc] > 0).all()),
                    "params": {k: float(v) for k, v in decoded.items()},
                }
            )

        document = {
            "evaluations": n_evals,
            "table_rows": n_rows,
            "epoch_entries": n_epochs,
            "truncated": n_evals != n_rows,
            "executions": len(list(self.interfaces)),
            "system": self.system(),
            "selection": self.config.selection,
            "model": self.model_ref(),
            "target": self.target_ref(),
            "source": optimization.output_filepath,
            "feature_bands": {k: list(v) for k, v in target.bands().items()},
            "solutions": solutions,
        }

        document = to_dict(document)
        if path is None:
            path = optimization.save_file("front.json", document)
        else:
            with open(path, "w") as f:
                json.dump(document, f, indent=2)
        n_feasible = sum(1 for s in solutions if s["feasible"])
        print(f"wrote {len(solutions)} solutions ({n_feasible} feasible) to {path}")
        return path

    @staticmethod
    def _evaluation_counts(h5) -> tuple:
        import numpy as np

        objectives = h5.get("objectives")
        if objectives is None:
            return 0, 0, len(h5["epochs"])
        values = objectives.to_numpy()
        usable = int(np.logical_not(np.any(np.isnan(values), axis=1)).sum())
        return len(values), usable, len(h5["epochs"])

    def _completed(self, optimization) -> int:
        try:
            if not optimization.is_materialized() or not os.path.isfile(
                optimization.output_filepath
            ):
                return -1
            return self._evaluation_counts(optimization.load_h5())[1]
        except Exception:
            return -1

    def _optimization(self):
        interfaces = list(self.interfaces)
        if len(interfaces) == 1:
            return interfaces[0]
        if not interfaces:
            raise ValueError("no optimization has been launched for this config")

        try:
            wanted = set(self.target().search_space(self.build_model()))
        except Exception:
            wanted = None

        candidates = []
        for candidate in interfaces:
            try:
                space = set(candidate.config.dopt_params.space.keys())
            except Exception:
                space = set()
            candidates.append((candidate, space, self._completed(candidate)))

        agreeing = [c for c in candidates if wanted is None or c[1] == wanted]
        if wanted is not None and not agreeing:
            print(
                f"WARNING: none of the {len(candidates)} stored runs searches "
                "this target's space, so every front below describes a "
                "different problem than the one this target states. Reading "
                "the largest anyway; re-run rather than trust it."
            )

        pool = sorted(agreeing or candidates, key=lambda c: -c[2])
        chosen, _chosen_space, chosen_n = pool[0]

        print(
            f"NOTE: {len(candidates)} runs are stored under this config; "
            f"reading the one with {chosen_n} completed evaluations."
        )
        for other, space, n in candidates:
            if other is chosen:
                continue
            if wanted is not None and space != wanted:
                missing = sorted(wanted - space)
                extra = sorted(space - wanted)
                why = "different space"
                if missing:
                    why += f", missing {missing}"
                if extra:
                    why += f", also searches {extra}"
            else:
                why = "same space, fewer evaluations"
            print(f"        {getattr(other, 'output_filepath', '?')}")
            print(f"          {n} evaluations -- skipped: {why}")

        return chosen

    def inspect(self, loc=None, params=None, verbose=0, gist=False):
        if self.is_unexpanded():
            return self._for_each(
                "inspect", loc=loc, params=params, verbose=verbose, gist=gist
            )

        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        optimization = self._optimization()
        print(f"System: {_system_label(self.system())}")
        if not optimization.is_materialized():
            print("No data yet (nothing launched for this config)")
            return None

        if not os.path.isfile(optimization.output_filepath):
            print("No data yet")
            return None

        h5 = optimization.load_h5()
        n_rows, n_evals, _n_epochs = self._evaluation_counts(h5)

        if n_evals != n_rows:
            print(f"  WARNING: the table has {n_rows} rows but only {n_evals} ")

        target, model = self.target(), self.build_model()

        best = self._ranked_best(optimization, target)
        n_front = 0 if best.get("y") is None else len(best["y"])

        bands = target.bands()
        if gist:
            print(gist_report(best, bands, n_evals))
            return None

        if bands:
            print("\nPer-solution band membership (`--summary` for parameters):")
            counts = in_band_counts(best["f"], bands)
            shown = ", ".join(f"loc={i} ({n}/{len(bands)})" for n, i in counts[:6])
            print(f"    best: {shown}")
            if counts and counts[0][1] != loc:
                print(
                    f"    note: loc={counts[0][1]} has {counts[0][0]}/{len(bands)} "
                    f"in band; the selected loc={loc} has "
                    f"{ {i: n for n, i in counts}[loc] }/{len(bands)}"
                )

        print(f"\nSelected solution (loc={loc}):")
        print("  y:", dict(best["y"].iloc[loc]))
        print("  f:", dict(best["f"].iloc[loc]))

        decoded = target.decode_params(
            optimization.parameter_vector_to_dict(
                list(map(float, best["x"].to_numpy()[loc]))
            ),
            model=model,
            strict=True,
        )

        groups = (
            target.describe_params(decoded)
            if hasattr(target, "describe_params")
            else {"params": decoded}
        )
        if verbose > 0:
            for name, group in groups.items():
                if group:
                    print(f"\n{name}:")
                    print(as_json(group))

        wfn = optimization.save_file("params.json", decoded)
        print("\nSaved to", wfn)

        print(f"Front: {n_front} solutions over {n_evals} evaluations")

        print("\nFeatures (f):")
        print(wide(best["f"]))
        print("\nObjectives (y):")
        print(wide(best["y"]))
        if best.get("c") is not None:
            print("\nConstraints (c):")
            if (best["c"] > 0).all(axis=None):
                print("All constraints satisfied")
            else:
                print(wide(best["c"]))
                import numpy as np

                c = np.asarray(best["c"])
                infeasible = int((c <= 0).any(axis=1).sum())
                if infeasible:
                    print(
                        f"{infeasible} of {len(c)} solutions on this front violate "
                        "at least one constraint, so this is an infeasible front."
                    )

        print(
            "Epochs",
            h5["epochs"][-1],
            " Evals ",
            n_evals,
            " n_i: ",
            optimization.num_initial_samples,
        )
        print("Cached:", optimization.cached())

        print(optimization.output_filepath)

        print(optimization.execution.output_filepath())
        return None
