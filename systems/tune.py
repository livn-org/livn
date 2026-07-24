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

    def version_cell(self, config: str):
        from systems.targets.cells.SingleCell import SingleCellOptConfig

        cfg = SingleCellOptConfig.from_yaml(config).model_dump()
        return {
            "system": "./systems/graphs/mn_single",
            "model": "livn.models.rcsd.ReducedCalciumSomaDendrite",
            "target": ["systems.targets.cells.SingleCell.SingleCell", {"config": cfg}],
        }

    def version_motoneuron(self):
        return self.version_cell("systems/targets/cells/motoneuron.yaml")

    def version_renshaw_perry(self):
        return self.version_cell("systems/targets/cells/rc_v1in_perry.yaml")

    def version_renshaw_invitro(self):
        return self.version_cell("systems/targets/cells/rc_v1in_invitro.yaml")

    def version_renshaw(self):
        return self.version_renshaw_perry()

    def launch(self):
        target = import_instance(self.config.target)
        model = import_instance(self.config.model)

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
                },
                "nprocs_per_worker": self.config.nprocs_per_worker,
            },
        ).launch()

        return self

    def inspect(self, loc=None):
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
        print("Epochs", h5["epochs"][-1], " Evals ", len(h5["epochs"]))
        print("Cached:", optimization.cached())

        target = import_instance(self.config.target)
        model = import_instance(self.config.model)

        best = optimization.get_best()
        if hasattr(target, "rank_solutions"):
            best = target.rank_solutions(best)

        with pd.option_context("display.max_columns", None):
            print("\nObjectives (y):")
            print(best["y"])
            print("\nFeatures (f):")
            print(best["f"])
            if best.get("c") is not None:
                print("\nConstraints (c):")
                print(best["c"])

        print(f"\nSelected solution (loc={loc}):")
        print("  y:", dict(best["y"].iloc[loc]))
        print("  f:", dict(best["f"].iloc[loc]))

        raw_params = optimization.parameter_vector_to_dict(
            list(map(float, best["x"].to_numpy()[loc]))
        )
        decoded = target.decode_params(raw_params, model=model)

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
