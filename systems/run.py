from livn.io import MEA
from machinable import Component
from pydantic import BaseModel, ConfigDict

from livn.system import System
from livn.utils import P
from livn.env import Env
from dmosopt.config import import_object_by_path


class Run(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "systems/data/S1"
        model: str | None = None
        duration: float = 100
        inputs: str | None = None
        inputs_namespace: str = ""
        noise: bool = False
        save: bool = False

    def __call__(self):
        model = self.config.model
        if model is not None:
            model = import_object_by_path(model)()

        system = System(self.config.system)

        mea = MEA.from_directory(system.uri)

        env = Env(system, model, mea).init()

        env.apply_model_defaults(noise=self.config.noise)
        env.record_spikes()

        if self.config.inputs is not None:
            env.apply_stimulus_from_h5(self.config.inputs, self.config.inputs_namespace)

        it, t, *_ = env.run(self.config.duration)

        it, t = P.gather(it, t)

        if P.is_root():
            it, t = P.merge(it, t)

            if self.config.save:
                self.save_file("results.p", {"it": it, "t": t})

            print("Simulation finished: ", it[:3], t[:3], flush=True)

    def on_write_meta_data(self):
        return P.rank() == 0

    def on_commit(self):
        if P.rank() != 0:
            return False
