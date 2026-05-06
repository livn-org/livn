import json
import os

from machinable import Interface, get
from pydantic import BaseModel, ConfigDict
from livn.utils import import_object_by_path
import pandas as pd


def _pj(p):
    return json.dumps(p, indent=4, sort_keys=True)


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "./systems/graphs/EI2"
        model: str | None = None
        target: str = "systems.targets.EI.Spontaneous"
        trials: int = 1
        nprocs_per_worker: int = 1
        n_initial: int = 10
        population_size: int = 100
        num_generations: int = 10
        n_epochs: int = 25

    def launch(self):
        target = import_object_by_path(self.config.target)()

        get(
            "interface.sopt",
            {
                "dopt_params": {
                    "space": target.search_space(),
                    "obj_fun_init_args": {
                        "system": self.config.system,
                        "model": self.config.model,
                        "target": self.config.target,
                        "trials": self.config.trials,
                    },
                    "n_epochs": self.config.n_epochs,
                    "n_initial": self.config.initial,
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

                # Emphasize high, healthy MFR first, pushing sub-0.3 Hz solutions down.
                mfr_component = (
                    sortable["mfr"].where(
                        sortable["mfr"] >= 0.3, sortable["mfr"] - 10.0
                    )
                ).fillna(-1e9)

                # Keep branching ratio close to 1.0; normalize deviations.
                branching_component = (
                    sortable["branching_ratio"].sub(1.0).abs() / 0.05
                ).fillna(1e6)

                # Encourage burst cadence near 0.1 Hz when available.
                if "burst_rate" in sortable.columns:
                    burst_component = sortable["burst_rate"].sub(0.1).abs().fillna(1e3)
                else:
                    burst_component = pd.Series(0.0, index=sortable.index)

                # Prefer delta dominance (ratio ~1.6) if metric exists.
                if "delta_theta_ratio" in sortable.columns:
                    delta_theta_component = (
                        sortable["delta_theta_ratio"].sub(1.6).abs().fillna(1e3)
                    )
                else:
                    delta_theta_component = pd.Series(0.0, index=sortable.index)

                if "avalanche_power_law" in sortable.columns:
                    avalanche_component = (
                        ((1.0 - sortable["avalanche_power_law"]) * 10.0)
                        .clip(lower=0.0)
                        .fillna(1e3)
                    )
                else:
                    avalanche_component = pd.Series(0.0, index=sortable.index)

                # Promote low synchrony solutions (higher constraint = better)
                c_df = best.get("c")
                if isinstance(c_df, pd.DataFrame) and "synchrony" in c_df.columns:
                    # Negate because higher constraint values are better (low sync)
                    synchrony_component = (-c_df["synchrony"]).fillna(1e3)
                else:
                    synchrony_component = pd.Series(0.0, index=sortable.index)

                if (
                    isinstance(c_df, pd.DataFrame)
                    and "max_synchronous_peak" in c_df.columns
                ):
                    peak_sync_component = (-c_df["max_synchronous_peak"]).fillna(1e3)
                else:
                    peak_sync_component = pd.Series(0.0, index=sortable.index)

                y_df = best.get("y")
                if isinstance(y_df, pd.DataFrame):
                    obj_component = y_df.sum(axis=1).fillna(1e6)
                else:
                    obj_component = pd.Series(0.0, index=sortable.index)

                score = (
                    -10.0 * mfr_component
                    + 5.0 * branching_component
                    + 2.0 * burst_component
                    + 1.0 * delta_theta_component
                    + 3.0 * avalanche_component
                    + 4.0 * synchrony_component
                    + 5.0 * peak_sync_component
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

            # Decode parameters: apply inverse transforms to all params first,
            # then use set_params to learn which keys are protocol-specific.
            target = import_object_by_path(self.config.target)()

            all_decoded = {}
            for name, value in raw_params.items():
                if name in target._transforms:
                    all_decoded[name] = target._transforms[name](
                        float(value), inverse=True
                    )
                else:
                    all_decoded[name] = float(value)

            # set_params consumes protocol-specific keys and returns env params
            env_params = target.set_params(all_decoded.copy())

            # Classify env params into weights and noise
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

            # Everything in all_decoded that set_params consumed is protocol-specific
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
