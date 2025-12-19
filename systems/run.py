from machinable import Component
from pydantic import BaseModel, ConfigDict

from livn.utils import P, import_object_by_path
from livn.env import Env


class Run(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str = "systems/graphs/S1"
        model: str | None = None
        decoding: str = "livn.types.Decoding"
        decoding_kwargs: dict = {"duration": 100}
        encoding: str | None = None
        encoding_kwargs: dict = {}

    def __call__(self):
        model = self.config.model
        if model is not None:
            model = import_object_by_path(model)()

        env = Env(self.config.system, model).init()
        env.apply_model_defaults()

        decoding = import_object_by_path(self.config.decoding)(
            **self.config.decoding_kwargs
        )
        encoding = self.config.encoding
        if encoding is not None:
            encoding = import_object_by_path(self.config.encoding)(
                **self.config.encoding_kwargs
            )

        response = env(decoding=decoding, encoding=encoding)
        if response is not None:
            self.save_file(f"response_{P.rank()}.p", response)

    def on_write_meta_data(self):
        return P.rank() == 0

    def on_commit(self):
        if P.rank() != 0:
            return False
