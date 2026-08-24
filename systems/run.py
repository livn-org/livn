import json

from machinable import Interface
from pydantic import BaseModel, ConfigDict

from livn.env import Env
from livn.utils import ObjSpec, P, import_instance


class Run(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "systems/graphs/EI"
        model: ObjSpec = None
        io: ObjSpec = None
        selection: str | int | float | dict | None = None
        params: dict | None = None
        encoding: ObjSpec = None
        decoding: ObjSpec = ("livn.decoding.GatherAndMerge", {"duration": 60_000})
        figure: ObjSpec = None

    def version_front(self, file: str, loc: int = 0):
        """One solution's decoded parameters, from a `tune --export` document."""
        with open(file) as handle:
            document = json.load(handle)

        solutions = {int(s["loc"]): s for s in document["solutions"]}
        if loc not in solutions:
            raise ValueError(
                f"no solution loc={loc} in {file}; it has "
                f"{', '.join(str(k) for k in sorted(solutions))}"
            )

        return {"params": dict(solutions[loc]["params"])}

    def version_params(self, file: str):
        with open(file) as handle:
            document = json.load(handle)
        return {"params": dict(document.get("params", document))}

    def __call__(self):
        env = Env(
            self.config.system,
            model=import_instance(self.config.model),
            io=import_instance(self.config.io),
            comm=P.comm(),
        )
        if self.config.selection is not None:
            env.selection(self.config.selection)

        env.init()

        if self.config.params is None:
            env.apply_default_params()
        else:
            env.set_params(dict(self.config.params))

        decoding = import_instance(self.config.decoding)
        encoding = import_instance(self.config.encoding)
        figure = import_instance(self.config.figure)

        response = env(decoding=decoding, encoding=encoding)

        if response is not None:
            print(self.save_file(f"response_{P.rank()}.p", response), flush=True)

        if figure is not None and P.is_root(comm=env.comm):
            print(
                figure(
                    response,
                    self.local_directory("figure.png"),
                    env=env,
                    encoding=encoding,
                ),
                flush=True,
            )

    def on_write_meta_data(self):
        return P.rank() == 0

    def on_commit(self):
        if P.rank() != 0:
            return False
        return None
