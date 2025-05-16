__all__ = ["Env", "Pass"]


from typing import TYPE_CHECKING

from livn.backend import Env

if TYPE_CHECKING:
    from livn.types import Decoding, Encoding


class Pass:
    def __init__(
        self,
        env: "Env",
        encoding: "Encoding",
        decoding: "Decoding",
        run_kwargs,
    ):
        self.env = env
        self.encoding = encoding
        self.decoding = decoding
        self.run_kwargs = run_kwargs

    def __call__(self, features):
        channel_inputs, duration = self.encoding(features)

        stimulus = self.env.cell_stimulus(channel_inputs)

        response = self.env.run(duration, stimulus, **self.run_kwargs)

        return self.decoding(duration, *response)
