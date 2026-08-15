import json
import os

from machinable import Interface, get
from pydantic import BaseModel, ConfigDict
from livn.utils import ObjSpec, import_instance
import pandas as pd


def _pj(p):
    return json.dumps(p, indent=4, sort_keys=True)


def decode_strict(target, model, space, raw: dict) -> dict:
    unknown = sorted(set(raw) - set(space))
    if unknown:
        raise ValueError(
            f"{unknown} are not in this target's search space, so they have no "
            "inverse transform and would be emitted at their encoded value. "
            "The target is probably built differently from the run that "
            f"produced them as this offers {sorted(space)}."
        )
    return target.decode_params(raw, model=model)


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str | int = "./systems/graphs/EI1"
        model: ObjSpec = "livn.models.rcsd.ReducedCalciumSomaDendrite"
        target: ObjSpec = "systems.targets.EI.Spontaneous"
        trials: int = 1
        nprocs_per_worker: int = 1
        n_initial: int = 100
        population_size: int = 100
        num_generations: int = 10
        n_epochs: int = 10
        surrogate: dict = {}

    def version_cell(self, config: str):
        from systems.targets.cells.SingleCell import SingleCellOptConfig

        cfg = SingleCellOptConfig.from_yaml(config).model_dump()
        return {
            "system": 1,
            "model": "livn.models.rcsd.ReducedCalciumSomaDendrite",
            "target": ["systems.targets.cells.SingleCell.SingleCell", {"config": cfg}],
        }

    def version_spontaneous_recording(
        self,
        file: str,
        readout: str = "channels",
        duration: float | None = None,
        warmup: float | None = None,
    ):
        with open(file) as f:
            document = json.load(f)["pooled"]
        measured = document["ei_targets"]
        features = document.get("summary", {}).get("features") or {}

        mea = None
        metadata = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(file))), "metadata.json"
        )
        if os.path.isfile(metadata):
            with open(metadata) as f:
                geometry = json.load(f)["geometry"]
            mea = {
                "electrode_coordinates": [
                    [float(i), float(x), float(y), 5.0] for i, x, y in geometry["pos"]
                ],
                "input_radius": 50.0,
                "output_radius": 50.0,
            }
        elif readout == "channels":
            raise FileNotFoundError(
                f"readout='channels' needs the recording set's metadata.json, but "
                f"{file!r} has none two levels up (looked in {metadata!r}). Pass "
                "readout='neurons' to tune without an array."
            )

        widen_factor = 2.0
        widen = {
            "BRANCHING_RATIO_BAND": ("branching_ratio", "band"),
            "POP_TAU_BAND_MS": ("pop_autocorr_tau", "band"),
            "SYNCHRONY_BAND": ("mean_channel_correlation", "band"),
            "MAX_BURST_RATE_HZ": ("burst_rate", "max"),
            "MAX_MEAN_RATE_HZ": ("mfr", "max"),
            "MIN_MEAN_RATE_HZ": ("mfr", "min"),
            "MAX_NEURON_RATE_HZ": ("max_neuron_firing_rate", "max"),
            "MAX_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "max"),
            "MIN_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "min"),
            "MAX_SYNC_PEAK": ("max_synchronous_peak", "max"),
        }
        gates = dict(measured)
        for key, (feature, kind) in widen.items():
            spec = features.get(feature)
            if not spec or key not in gates:
                continue
            lo, hi = spec.get("q_lo"), spec.get("q_hi")
            if lo is None or hi is None:
                continue
            slack = (hi - lo) / 2.0 * (widen_factor - 1.0)
            if kind == "band":
                gates[key] = [float(lo - slack), float(hi + slack)]
            elif kind == "max":
                gates[key] = float(hi + slack)
            else:
                gates[key] = float(max(0.0, lo - slack))

        overrides = {
            name: value
            for name, value in gates.items()
            if name != "MIN_ACTIVE_FRACTION"
        }
        overrides["targets"] = {
            name: value
            for name, value in measured["targets"].items()
            if name != "active_fraction"
        }

        skip_constraints = ["avalanche_r2", "active_fraction_floor"]
        synchrony_detection_floor = 0.01
        sync_lo, sync_hi = measured.get("SYNCHRONY_BAND", (0.0, 1.0))
        if sync_lo <= 0.0 or sync_hi < synchrony_detection_floor:
            skip_constraints.append("synchrony")

        for feature, constraint in (("pop_autocorr_tau", "pop_autocorr_tau_band"),):
            spec = features.get(feature)
            if spec and spec.get("q_lo") is not None and spec.get("min") is not None:
                if abs(spec["q_lo"] - spec["min"]) <= 1e-9:
                    skip_constraints.append(constraint)

        options = {
            "overrides": overrides,
            "feature_bands": {
                name: (spec["q_lo"], spec["q_hi"])
                for name, spec in features.items()
                if spec is not None and spec.get("q_lo") is not None
            },
            "readout": readout,
            "mea": mea,
            "skip_objectives": ["active_fraction"],
            "skip_constraints": skip_constraints,
        }
        if duration is not None:
            options["duration"] = float(duration)
        if warmup is not None:
            options["warmup"] = float(warmup)

        return {"target": ["systems.targets.EI.Spontaneous", options]}

    def version_rcsd(
        self, implicit_inhibition: bool = False, short_term_depression: bool = False
    ):
        options = {"implicit_inhibition": implicit_inhibition}
        if short_term_depression:
            options["short_term_depression"] = True
        return {"model": ["livn.models.rcsd.ReducedCalciumSomaDendrite", options]}

    def version_E_only(self):
        return self.version_rcsd(implicit_inhibition=True, short_term_depression=True)

    def version_EI(self):
        return self.version_rcsd(implicit_inhibition=False, short_term_depression=False)

    def version_motoneuron(self):
        return self.version_cell("systems/targets/cells/motoneuron.yaml")

    def version_renshaw_perry(self):
        return self.version_cell("systems/targets/cells/rc_v1in_perry.yaml")

    def version_renshaw_invitro(self):
        return self.version_cell("systems/targets/cells/rc_v1in_invitro.yaml")

    def version_renshaw(self):
        return self.version_renshaw_perry()

    def _target_and_model(self):
        target = import_instance(self.config.target)
        model = import_instance(self.config.model)

        if getattr(target, "system", None) is None and hasattr(target, "system"):
            target.system = self.config.system

        self._check_composition(model)

        return target, model

    def _check_composition(self, model) -> None:
        spec = self.config.model
        if spec is None or isinstance(spec, str):
            return
        try:
            options = spec[1] or {}
            if "implicit_inhibition" not in options:
                return
        except (IndexError, KeyError, TypeError):
            return

        system = self.config.system
        if not isinstance(system, (str, os.PathLike)):
            return
        graph = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            str(system).lstrip("./"),
            "graph.json",
        )
        if not os.path.exists(graph):
            return
        with open(graph) as f:
            counts = json.load(f)["architecture"]["config"]["cell_counts"]

        has_inhibition = int(counts.get("INH", 0)) > 0
        implicit = bool(options["implicit_inhibition"])
        if implicit and has_inhibition:
            raise ValueError(
                f"{os.path.basename(str(system))} has {counts.get('INH')} INH cells "
                "but the model folds inhibition into the cell dynamics "
                "(`~E_only`); they would be built and then ignored. Use `~EI`, "
                "or an excitatory-only graph."
            )
        if not implicit and not has_inhibition:
            raise ValueError(
                f"{os.path.basename(str(system))} has no INH cells but the model "
                "simulates them explicitly (`~EI`); their weights would select "
                "no synapses. Use `~E_only`, or a mixed graph."
            )

    def launch(self):
        target, model = self._target_and_model()

        surrogate_config = {}
        for k, v in self.config.surrogate.items():
            surrogate_config["surrogate_" + k] = v

        get(
            "interface.sopt",
            {
                "system": self.config.system,
                "dopt_params": {
                    "space": target.search_space(model),
                    "obj_fun_init_args": {
                        "model": self.config.model,
                        "target": self.config.target,
                        "trials": self.config.trials,
                    },
                    "n_epochs": self.config.n_epochs,
                    "n_initial": self.config.n_initial,
                    "population_size": self.config.population_size,
                    "num_generations": self.config.num_generations,
                    **surrogate_config,
                },
                "nprocs_per_worker": self.config.nprocs_per_worker,
            },
        ).launch()

        return self

    @staticmethod
    def _feature_bands(target) -> dict:
        bands = getattr(target, "feature_bands", None)
        if callable(bands):
            bands = bands()
        return dict(bands or {})

    def _ranked_best(self, optimization, target):
        best = optimization.get_best()
        if hasattr(target, "rank_solutions"):
            best = target.rank_solutions(best)
        return best

    def summary(self, params=None, sort=True):
        optimization = self.interfaces[0]
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self._target_and_model()
        best = self._ranked_best(optimization, target)
        f = best.get("f")
        if f is None or len(f) == 0:
            print("No solutions")
            return None

        bands = self._feature_bands(target)
        wanted = [p.strip() for p in (params or "").split(",") if p.strip()]
        space = target.search_space(model) if wanted else {}

        rows = []
        for i in range(len(f)):
            row = {"loc": i}
            n_in = 0
            for name in f.columns:
                value = float(f[name].iloc[i])
                lo_hi = bands.get(name)
                inside = lo_hi is not None and lo_hi[0] <= value <= lo_hi[1]
                n_in += bool(inside)
                row[name] = f"{value:.4g}{'*' if inside else ''}"
            if bands:
                row["in_band"] = f"{n_in}/{len(bands)}"
                row["_n"] = n_in
            if wanted:
                decoded = decode_strict(
                    target,
                    model,
                    space,
                    optimization.parameter_vector_to_dict(
                        list(map(float, best["x"].to_numpy()[i]))
                    ),
                )
                for name in wanted:
                    value = decoded.get(name)
                    row[name] = "--" if value is None else f"{float(value):.4g}"
            rows.append(row)

        table = pd.DataFrame(rows)
        if sort and "_n" in table:
            table = table.sort_values("_n", ascending=False)
        table = table.drop(columns=[c for c in ("_n",) if c in table])

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(table.to_string(index=False))
        if bands:
            print("\n* = inside the measured band:")
            for name, (lo, hi) in bands.items():
                print(f"    {name:<24} {lo:>10.4g} - {hi:<10.4g}")
        else:
            print("\n(the target states no feature bands, so nothing is marked)")
        return table

    def export(self, path: str | None = None, feasible_only: bool = False):
        import numpy as np

        optimization = self.interfaces[0]
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self._target_and_model()
        best = self._ranked_best(optimization, target)

        space = target.search_space(model)
        c = np.asarray(best["c"]) if best.get("c") is not None else None

        solutions = []
        for loc in range(len(best["x"])):
            if feasible_only and c is not None and (c[loc] <= 0).any():
                continue
            raw = optimization.parameter_vector_to_dict(
                list(map(float, best["x"].to_numpy()[loc]))
            )
            decoded = decode_strict(target, model, space, raw)
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
            "system": self.config.system,
            "model": self.config.model,
            "target": self.config.target,
            "source": optimization.output_filepath,
            "feature_bands": {
                k: list(v) for k, v in self._feature_bands(target).items()
            },
            "solutions": solutions,
        }

        path = path or "front.json"
        with open(path, "w") as f:
            json.dump(document, f, indent=2)
        n_feasible = sum(1 for s in solutions if s["feasible"])
        print(f"wrote {len(solutions)} solutions ({n_feasible} feasible) to {path}")
        return path

    def inspect(self, loc=None, params=None):
        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        optimization = self.interfaces[0]
        print(f"System: {self.config.system}")
        if not optimization.is_materialized():
            print("No data yet (nothing launched for this config)")
            return
        print(optimization.output_filepath)
        if not os.path.isfile(optimization.output_filepath):
            print("No data yet")
            return

        h5 = optimization.load_h5()
        print(
            "Epochs",
            h5["epochs"][-1],
            " Evals ",
            len(h5["epochs"]),
            " n_i: ",
            optimization.num_initial_samples,
        )
        print("Cached:", optimization.cached())

        target, model = self._target_and_model()

        best = self._ranked_best(optimization, target)

        with pd.option_context("display.max_columns", None):
            print("\nObjectives (y):")
            print(best["y"])
            print("\nFeatures (f):")
            print(best["f"])
            if best.get("c") is not None:
                print("\nConstraints (c):")
                if (best["c"] > 0).all(axis=None):
                    print("All constraints satisfied")
                else:
                    print(best["c"])
                    import numpy as np

                    c = np.asarray(best["c"])
                    infeasible = int((c <= 0).any(axis=1).sum())
                    if infeasible:
                        print(
                            f"{infeasible} of {len(c)} solutions on this front violate at "
                            "least one constraint, so this is an infeasible front."
                        )

        bands = self._feature_bands(target)
        if bands:
            print("\nPer-solution band membership (`--summary` for parameters):")
            counts = []
            for i in range(len(best["f"])):
                n_in = sum(
                    lo <= float(best["f"][name].iloc[i]) <= hi
                    for name, (lo, hi) in bands.items()
                    if name in best["f"].columns
                )
                counts.append((n_in, i))
            counts.sort(reverse=True)
            shown = ", ".join(f"loc={i} ({n}/{len(bands)})" for n, i in counts[:6])
            print(f"    best: {shown}")
            if counts and counts[0][1] != loc:
                print(
                    f"    note: loc={counts[0][1]} has {counts[0][0]}/{len(bands)} "
                    f"in band; the selected loc={loc} has "
                    f"{dict((i, n) for n, i in counts)[loc]}/{len(bands)}"
                )

        print(f"\nSelected solution (loc={loc}):")
        print("  y:", dict(best["y"].iloc[loc]))
        print("  f:", dict(best["f"].iloc[loc]))

        raw_params = optimization.parameter_vector_to_dict(
            list(map(float, best["x"].to_numpy()[loc]))
        )
        decoded = decode_strict(target, model, target.search_space(model), raw_params)

        groups = (
            target.describe_params(decoded)
            if hasattr(target, "describe_params")
            else {"params": decoded}
        )
        for name, group in groups.items():
            if group:
                print(f"\n{name}:")
                print(_pj(group))

        wfn = optimization.save_file("params.json", decoded)
        print("\nSaved to", wfn)
