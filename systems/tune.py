import json
import os
import sys

import numpy as np
from machinable import Interface, get
from machinable.utils import load_file
from pydantic import BaseModel, ConfigDict, Field

from livn.backend import backend


def _pj(p):
    return json.dumps(p, indent=4, sort_keys=True)


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        stimulate: bool = False
        weights: str | bool = False
        noise: dict | bool = False
        trials: int = 1

        system: str = "./systems/data/S1"
        model: str | None = None
        nprocs_per_worker: int = 1

    def launch(self):
        with open(os.path.join(self.config.system, "graph.json")) as f:
            graph = json.load(f)

        # Parameters

        space = {}
        problem_parameters = {}

        if isinstance(self.config.weights, str):
            # fixed weights
            problem_parameters.update(load_file(self.config.weights))
        elif self.config.weights is True:
            # find weights
            for pre, val in graph["connections"]["EXC"]["config"]["synapses"].items():
                for post, p in val.items():
                    if backend() != "neuron":
                        space[f"{pre}_{post}"] = [0, 10.0]
                    else:
                        for section in p["sections"]:
                            for mech in p["mechanisms"]["default"].keys():
                                space[f"{pre}_{post}-{section}-{mech}-weight"] = [
                                    0,
                                    10.0,
                                ]

        if isinstance(self.config.noise, dict):
            problem_parameters.update(self.config.noise)
        elif self.config.noise is True:
            space.update(
                {
                    "noise_exc": [0.001, 10.0],
                    "noise_inh": [0.001, 10.0],
                }
            )

        num_channels = len(
            load_file([self.config.system, "mea.json"])["electrode_coordinates"]
        )

        objective_names = ["rate", "active", "isi"]

        if self.config.stimulate is True:
            objective_names = ["corr", "isi"] + [
                f"{q}{i}" for i in range(num_channels) for q in ("a", "s")
            ]

        targets = {
            "rate": 3.0,
            "active": 0.6,
            "cell_count": {
                p: sum([i for i in dist.values()])
                for p, dist in graph["architecture"]["config"][
                    "cell_distributions"
                ].items()
            },
            "objective_names": objective_names,
        }

        get(
            "interface.sopt",
            {
                "dopt_params": {
                    "space": space,
                    "obj_fun_init_args": {
                        "system": self.config.system,
                        "model": self.config.model,
                        "stimulate": self.config.stimulate,
                        "trials": self.config.trials,
                    },
                    "reduce_fun_args": (targets,),
                    "problem_parameters": problem_parameters,
                    "objective_names": objective_names,
                    "surrogate_method_name": "gpr",
                    "surrogate_method_kwargs": {"top_k": 250},
                    "n_epochs": 10,
                    "population_size": 100,
                    "num_generations": 10,
                    "n_initial": 100,
                },
                "nprocs_per_worker": self.config.nprocs_per_worker,
            },
        ).launch()

        return self

    @property
    def weights_filepath(self):
        return self.components[-1].local_directory("weights.json")

    def best(self, loc=-1):
        opt = self.components[0]
        best = opt.get_best(sort_by="np.max(y,axis=1)")
        return opt.parameter_vector_to_dict(list(map(float, best["x"].to_numpy()[loc])))

    def inspect(self, loc=None):
        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        optimization = self.components[0]
        print(f"Network: {self.config.system}")

        if os.path.isfile(optimization.output_filepath):
            print(optimization.output_filepath)
            h5 = optimization.load_h5()
            print("Epochs", h5["epochs"][-1], len(h5["epochs"]))
            print("Cached: ", optimization.cached())
            best = optimization.get_best(sort_by="np.max(y,axis=1)")
            print(
                "Best solution",
            )
            print("y\n\n", best["y"].iloc[loc], "f\n\n", best["f"].iloc[loc])
            print(
                "Features:",
            )

            print(best["f"])
            weights = optimization.parameter_vector_to_dict(
                list(map(float, best["x"].to_numpy()[loc]))
            )
            noise = dict(
                exc=weights.pop("noise_exc", 0.0), inh=weights.pop("noise_inh", 0.0)
            )

            print("Weights", _pj(weights))
            print('noise="', noise, '"')

            print("Saved to")
            wfn = optimization.save_file("weights.json", weights)
            print(wfn)
            # save_file([self.config.directory, "weights.json"], weights)

            print("Command to continue")
            cmd = " ".join(sys.argv[: sys.argv.index("tune")])
            print(
                f"{cmd} tune system={self.config.system} model={self.config.model} nprocs_per_worker={self.config.nprocs_per_worker} weights={wfn} noise=1 --launch"
            )
        else:
            print("No data yet")

    def eval(self, loc=-1):
        c = self.components[0]
        data = c.get_best()  # load_h5()

        print(data["f"])

        print("Selected:")
        print(data["f"].iloc[loc])

        x = list(map(float, data["x"].to_numpy()[loc]))

        w = c.parameter_vector_to_dict(x)

        print(c.evaluate_objective_at(w))

        print(c.output_filepath)
        print(w)
