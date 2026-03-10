from machinable import Component
from pydantic import BaseModel, ConfigDict

from livn.utils import P, ObjSpec, import_instance
from livn.env import Env


class Run(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "systems/graphs/S1"
        model: ObjSpec = None
        decoding: ObjSpec = ("livn.types.Decoding", {"duration": 100})
        encoding: ObjSpec = None

    def __call__(self):
        model = import_instance(self.config.model)

        env = Env(self.config.system, model).init()
        env.apply_model_defaults()

        decoding = import_instance(self.config.decoding)
        encoding = import_instance(self.config.encoding)

        response = env(decoding=decoding, encoding=encoding)
        if response is not None:
            self.save_file(f"response_{P.rank()}.p", response)

    def on_write_meta_data(self):
        return P.rank() == 0

    def on_commit(self):
        if P.rank() != 0:
            return False
