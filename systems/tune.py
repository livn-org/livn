import json
import os

from machinable import Interface, get
from pydantic import BaseModel, ConfigDict
from livn.utils import ObjSpec, import_instance
import pandas as pd


def _pj(p):
    return json.dumps(p, indent=4, sort_keys=True)


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "./systems/graphs/EI1"
        model: ObjSpec = "livn.models.rcsd.ReducedCalciumSomaDendrite"
        target: ObjSpec = "systems.targets.EI.Spontaneous"
        trials: int = 1
        nprocs_per_worker: int = 1
        n_initial: int = 100
        population_size: int = 100
        num_generations: int = 10
        n_epochs: int = 10

    def launch(self):
        target = import_instance(self.config.target)
        model = import_instance(self.config.model)

        get(
            "interface.sopt",
            {
                "dopt_params": {
                    "space": target.search_space(model),
                    "obj_fun_init_args": {
                        "system": self.config.system,
                        "model": self.config.model,
                        "target": self.config.target,
                        "trials": self.config.trials,
                    },
                    "n_epochs": self.config.n_epochs,
                    "n_initial": self.config.n_initial,
                    "population_size": self.config.population_size,
                    "num_generations": self.config.num_generations,
                },
                "nprocs_per_worker": self.config.nprocs_per_worker,
            },
        ).launch()

        return self

    def inspect(self, loc=None):
        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        optimization = self.components[0]
        print(f"Network: {self.config.system}")
        print(optimization.output_filepath)
        print(optimization.execution.output_filepath())
        print("num initial", optimization.num_initial_samples)
        if os.path.isfile(optimization.output_filepath):
            h5 = optimization.load_h5()
            print("Epochs", h5["epochs"][-1], len(h5["epochs"]))
            print("Cached: ", optimization.cached())
            best = optimization.get_best()

            features_df = best.get("f")
            if isinstance(features_df, pd.DataFrame) and {
                "mfr",
                "branching_ratio",
            }.issubset(features_df.columns):
                sortable = features_df.copy()
                c_df = best.get("c")
                y_df = best.get("y")

                def _feat(name, default=0.0):
                    if name in sortable.columns:
                        return sortable[name].astype(float).fillna(default)
                    return pd.Series(default, index=sortable.index)

                def _con(name, default=0.0):
                    if isinstance(c_df, pd.DataFrame) and name in c_df.columns:
                        return c_df[name].astype(float).fillna(default)
                    return pd.Series(default, index=sortable.index)

                import numpy as _np

                mfr = _feat("mfr", 0.0)
                eps = 1e-3
                mfr_component = pd.Series(
                    _np.log((mfr.clip(lower=0.0).to_numpy() + eps) / (1.0 + eps)) ** 2,
                    index=sortable.index,
                )
                isi_component = _feat("isi_cv", 0.0).sub(1.2).abs()
                active_obj_component = (1.0 - _feat("active_fraction", 0.0)).clip(
                    lower=0.0
                )

                branching_component = _feat("branching_ratio", 0.0).sub(1.0).abs()
                active_floor_component = -_con("active_fraction_floor", -1.0)
                synchrony_component = -_con("synchrony", -1.0)
                peak_sync_component = -_con("max_synchronous_peak", -1.0)
                tau_band_component = -_con("pop_autocorr_tau_band", -1.0)
                burst_cap_component = -_con("burst_rate_cap", -1.0)
                branching_band_component = -_con("branching_ratio_band", -1.0)
                avalanche_r2_component = -_con("avalanche_r2", -1.0)

                if isinstance(y_df, pd.DataFrame):
                    obj_component = y_df.sum(axis=1).fillna(1e6)
                else:
                    obj_component = pd.Series(0.0, index=sortable.index)

                score = (
                    5.0 * mfr_component
                    + 2.0 * isi_component
                    + 5.0 * active_obj_component
                    + 2.0 * branching_component
                    + 2.0 * active_floor_component
                    + 4.0 * synchrony_component
                    + 5.0 * peak_sync_component
                    + 2.0 * tau_band_component
                    + 2.0 * burst_cap_component
                    + 3.0 * branching_band_component
                    + 3.0 * avalanche_r2_component
                    + 0.5 * obj_component
                ).sort_values()

                order = score.index
                for key, value in best.items():
                    if isinstance(value, pd.DataFrame):
                        best[key] = value.loc[order].reset_index(drop=True)
            print(
                "Best solution",
            )
            print("y\n\n", best["y"].iloc[loc], "f\n\n", best["f"].iloc[loc])
            print(
                "Features:",
            )
            with pd.option_context("display.max_columns", None):
                print(best["f"])

            print(
                "Obj:",
            )
            print(best["y"])
            print(
                "Const:",
            )
            print(best["c"])

            raw_params = optimization.parameter_vector_to_dict(
                list(map(float, best["x"].to_numpy()[loc]))
            )

            target = import_instance(self.config.target)
            model = import_instance(self.config.model)

            all_decoded = target.decode_params(raw_params, model=model)

            env_params = target.set_params(all_decoded.copy())

            weight_params = {
                k: v
                for k, v in env_params.items()
                if "-weight" in k and not k.startswith("noise-")
            }
            noise_keys = {"std_e", "std_i", "g_e0", "g_i0", "tau_e", "tau_i"}
            noise_params = {
                k.replace("noise-", "", 1): v
                for k, v in env_params.items()
                if k.startswith("noise-") and k.replace("noise-", "", 1) in noise_keys
            }

            protocol_params = {
                k: v for k, v in all_decoded.items() if k not in env_params
            }

            print("All decoded params", _pj(all_decoded))
            if weight_params:
                print("\nWeights for neuron_default_weights:")
                print(_pj(weight_params))
            if noise_params:
                print("\nNoise for neuron_default_noise:")
                print(_pj(noise_params))
            if protocol_params:
                print("\nProtocol-specific params:")
                print(_pj(protocol_params))

            print("Saved to")
            wfn = optimization.save_file("params.json", all_decoded)
            print(wfn)
        else:
            print("No data yet")
