import json
import os
import shutil

from machinable import Interface, get

from livn.io import electrode_array_coordinates


class GenerateSystem(Interface):
    class Config:
        config: str = "systems/config/S1.yml"

    def launch(self):
        get(
            "miv_simulator.interface.network",
            f"~from_config('{self.config.config}')",
        ).launch()

        return self

    @property
    def output_directory(self):
        return self.components[-1].local_directory()

    def inspect(self):
        print("Directory:")
        d = self.output_directory
        print(d)
        print(os.system(f"ls -la {d}"))

    def mea(self):
        fn = os.path.join(self.output_directory, "mea.json")
        if os.path.isfile(fn):
            raise FileExistsError("mea.json already exists.")

        with open(fn, "w") as f:
            json.dump(
                {
                    "electrode_coordinates": electrode_array_coordinates().tolist(),
                    "input_radius": 250,
                    "output_radius": 250,
                },
                f,
            )

    def export(self):
        name = (
            os.path.basename(self.config.config)
            .replace(".yml", "")
            .replace(".yaml", "")
        )
        t = os.path.join(os.path.dirname(__file__), "data", name)
        d = self.output_directory
        print(f"Export: {d} -> {t}")

        if os.path.isdir(t):
            raise FileExistsError("The target directory already exists.")

        if not os.path.isfile(os.path.join(d, "graph.json")):
            raise FileNotFoundError(
                "The generation appears incomplete. Please (re)generate the system."
            )

        files = ["cells.h5", "connections.h5", "graph.json"]
        os.makedirs(t, exist_ok=True)
        for file in files:
            print(f"Copying {file} ...")
            shutil.copy2(os.path.join(d, file), t)

        if os.path.isfile(os.path.join(d, "mea.json")):
            print(f"Copying mea.json ...")
            shutil.copy2(os.path.join(d, "mea.json"), t)

        print(f"Export to {t} complete.")
