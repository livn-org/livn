from livn.types import Encoding


class H5Inputs(Encoding):
    filepath: str | None = None
    namespace: str = ""
    attribute: str = "Spike Train"
    onset: int = 0
    io_size: int = 1
    microcircuit_inputs: bool = True
    n_trials: int = 1
    equilibration_duration: float = 250.0

    def __call__(self, env, t_end, inputs):
        filepath = self.filepath
        if filepath is None:
            filepath = inputs
        env.apply_stimulus_from_h5(
            filepath,
            self.namespace,
            self.attribute,
            self.onset,
            self.io_size,
            self.microcircuit_inputs,
            self.n_trials,
            self.equilibration_duration,
        )

